from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .comparison import compare_fused_to_model, compare_ref_delta
from .spectrum_interop import spectrum_bridge
from .telemetry import TelemetryWriter, flatten_record
from .trajectory import StreamLayout


CALIBRATION_SCHEMA_VERSION = 2
COMPARISON_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_calibration_id(value: str) -> str:
    safe = _SAFE_ID.sub("_", str(value)).strip("._")
    if not safe:
        raise ValueError("calibration_id must contain at least one filename-safe character")
    return safe


def safe_comparison_label(value: str) -> str:
    safe = safe_calibration_id(value).lower()
    if safe in {"capture", "comparisons"}:
        raise ValueError(f"reserved comparison_label: {safe}")
    return safe


def invocation_key(seed: int | None, steps: int, shape: tuple[int, ...]) -> str:
    seed_part = "unknown" if seed is None else str(int(seed))
    shape_part = "x".join(str(int(value)) for value in shape)
    return f"seed{seed_part}-steps{int(steps)}-shape{shape_part}"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {
            key: value.detach().to("cpu").contiguous()
            for key, value in tensors.items()
        },
        str(temporary),
    )
    os.replace(temporary, path)


class CalibrationCaptureWriter:
    """Persist one sampler invocation without retaining trajectory tensors in RAM."""

    def __init__(
        self,
        output_directory: Path,
        calibration_id: str,
        seed: int | None,
        steps: int,
        shape: tuple[int, ...],
        sigmas: torch.Tensor,
        layout: StreamLayout,
    ) -> None:
        self.calibration_id = safe_calibration_id(calibration_id)
        self.seed = None if seed is None else int(seed)
        self.steps = int(steps)
        self.shape = tuple(int(value) for value in shape)
        self.key = invocation_key(self.seed, self.steps, self.shape)
        self.directory = output_directory / "refdelta_calibration" / self.calibration_id / self.key
        if self.directory.exists():
            if (self.directory / "manifest.json").is_file():
                raise FileExistsError(
                    "completed RefDelta calibration capture already exists for "
                    f"{self.key}; choose a new calibration_id instead of overwriting it"
                )
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sigmas = [float(value) for value in sigmas.detach().float().cpu()]
        self.video_elements = layout.video_elements
        self._records: list[dict[str, Any] | None] = [None] * self.steps
        self._actual: list[bool | None] = [None] * self.steps
        self._closed = False

    def write_step(
        self,
        step: int,
        state: torch.Tensor,
        fused_x0: torch.Tensor,
        *,
        actual: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError("calibration capture is already closed")
        if not 0 <= int(step) < self.steps:
            raise IndexError("calibration capture step is outside the invocation")
        if tuple(state.shape) != self.shape or tuple(fused_x0.shape) != self.shape:
            raise ValueError("calibration capture tensor shape changed during one invocation")
        _atomic_safetensors(
            self.directory / f"step-{int(step):04d}.safetensors",
            {"state": state, "fused_x0": fused_x0},
        )
        self._actual[int(step)] = bool(actual)

    def write_record(self, step: int, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("calibration capture is already closed")
        self._records[int(step)] = flatten_record(record)

    def close(self, final_sample: torch.Tensor) -> None:
        if self._closed:
            return
        missing_steps = [index for index, value in enumerate(self._actual) if value is None]
        missing_records = [index for index, value in enumerate(self._records) if value is None]
        if missing_steps or missing_records:
            raise RuntimeError(
                "incomplete calibration capture: "
                f"missing tensor steps={missing_steps}, telemetry steps={missing_records}"
            )
        if tuple(final_sample.shape) != self.shape:
            raise ValueError("calibration capture final sample shape changed")
        _atomic_safetensors(
            self.directory / "final.safetensors",
            {"sample": final_sample},
        )
        _atomic_json(
            self.directory / "manifest.json",
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "complete": True,
                "calibration_id": self.calibration_id,
                "invocation_key": self.key,
                "seed": self.seed,
                "steps": self.steps,
                "shape": list(self.shape),
                "sigmas": self.sigmas,
                "video_elements": self.video_elements,
                "actual_model_evaluations": self._actual,
                "records": self._records,
            },
        )
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        marker = self.directory / "INCOMPLETE"
        marker.write_text(
            "Sampler invocation did not complete; this capture is not replayable.\n",
            encoding="utf-8",
        )
        self._closed = True


class CalibrationReplay:
    """Read one completed fused capture and its labeled comparison passes."""

    def __init__(
        self,
        output_directory: Path,
        calibration_id: str,
        seed: int | None,
        steps: int,
        shape: tuple[int, ...],
        sigmas: torch.Tensor,
    ) -> None:
        safe_id = safe_calibration_id(calibration_id)
        self.key = invocation_key(seed, steps, shape)
        self.directory = output_directory / "refdelta_calibration" / safe_id / self.key
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"completed RefDelta calibration capture not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != CALIBRATION_SCHEMA_VERSION or not manifest.get("complete"):
            raise RuntimeError("RefDelta calibration capture is incomplete or uses an unsupported schema")
        if int(manifest.get("steps", -1)) != int(steps):
            raise ValueError("RefDelta calibration replay step count does not match the capture")
        if tuple(manifest.get("shape", ())) != tuple(int(value) for value in shape):
            raise ValueError("RefDelta calibration replay latent shape does not match the capture")
        expected_sigmas = torch.tensor(manifest.get("sigmas", ()), dtype=torch.float64)
        received_sigmas = sigmas.detach().to("cpu", dtype=torch.float64)
        if expected_sigmas.shape != received_sigmas.shape or not torch.allclose(
            expected_sigmas,
            received_sigmas,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("RefDelta calibration replay sigma schedule does not match the capture")
        records = manifest.get("records")
        actual = manifest.get("actual_model_evaluations")
        if not isinstance(records, list) or len(records) != steps:
            raise RuntimeError("RefDelta calibration capture has invalid telemetry records")
        if not isinstance(actual, list) or len(actual) != steps:
            raise RuntimeError("RefDelta calibration capture has invalid actual-step metadata")
        self.records: list[dict[str, Any]] = records
        self.actual: list[bool] = [bool(value) for value in actual]
        self.steps = int(steps)
        self.shape = tuple(int(value) for value in shape)
        self.capture_fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._comparison_manifests: dict[str, dict[str, Any]] = {}
        video_elements = manifest.get("video_elements")
        self.layout = StreamLayout(None if video_elements is None else int(video_elements))

    def load_step(self, step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.directory / f"step-{int(step):04d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"RefDelta calibration capture is missing {path.name}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"state", "fused_x0"}:
            raise RuntimeError(f"RefDelta calibration step file has unexpected tensors: {path.name}")
        return tensors["state"].to(device), tensors["fused_x0"].to(device)

    def load_final(self, device: torch.device) -> torch.Tensor:
        path = self.directory / "final.safetensors"
        if not path.is_file():
            raise FileNotFoundError("RefDelta calibration capture is missing final.safetensors")
        tensors = load_file(str(path), device="cpu")
        sample = tensors.get("sample")
        if sample is None:
            raise RuntimeError("RefDelta calibration final file does not contain sample")
        return sample.to(device)

    def comparison_manifest(self, comparison_label: str) -> dict[str, Any]:
        label = safe_comparison_label(comparison_label)
        cached = self._comparison_manifests.get(label)
        if cached is not None:
            return cached
        path = self.directory / "comparisons" / label / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"completed RefDelta comparison pass '{label}' not found: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("schema_version") != COMPARISON_SCHEMA_VERSION
            or not manifest.get("complete")
        ):
            raise RuntimeError(
                f"RefDelta comparison pass '{label}' is incomplete or uses an unsupported schema"
            )
        if manifest.get("comparison_label") != label:
            raise RuntimeError(f"RefDelta comparison manifest label mismatch for '{label}'")
        if manifest.get("capture_fingerprint") != self.capture_fingerprint:
            raise RuntimeError(f"RefDelta comparison pass '{label}' belongs to another capture")
        if manifest.get("invocation_key") != self.key or int(manifest.get("steps", -1)) != self.steps:
            raise RuntimeError(f"RefDelta comparison pass '{label}' invocation metadata is stale")
        if manifest.get("actual_model_evaluations") != self.actual:
            raise RuntimeError(f"RefDelta comparison pass '{label}' actual-step metadata is stale")
        self._comparison_manifests[label] = manifest
        return manifest

    def load_comparison_step(
        self,
        comparison_label: str,
        step: int,
        device: torch.device,
    ) -> torch.Tensor:
        label = safe_comparison_label(comparison_label)
        self.comparison_manifest(label)
        if not self.actual[int(step)]:
            raise ValueError(f"comparison output requested for non-actual capture step {step}")
        path = self.directory / "comparisons" / label / f"step-{int(step):04d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"RefDelta comparison pass '{label}' is missing {path.name}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"comparison_x0"}:
            raise RuntimeError(f"RefDelta comparison step file has unexpected tensors: {path}")
        value = tensors["comparison_x0"]
        if tuple(value.shape) != self.shape:
            raise RuntimeError(f"RefDelta comparison step shape is stale: {path}")
        return value.to(device)


class ComparisonPassWriter:
    """Persist one labeled comparison pass without retaining per-step x0 tensors."""

    def __init__(self, replay: CalibrationReplay, comparison_label: str) -> None:
        self.replay = replay
        self.label = safe_comparison_label(comparison_label)
        self.directory = replay.directory / "comparisons" / self.label
        if self.directory.exists():
            if (self.directory / "manifest.json").is_file():
                raise FileExistsError(
                    f"completed RefDelta comparison pass '{self.label}' already exists for "
                    f"{replay.key}; choose another comparison_label or calibration_id"
                )
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any] | None] = [None] * replay.steps
        self._written: list[bool] = [False] * replay.steps
        self._closed = False

    def write_step(
        self,
        step: int,
        comparison_x0: torch.Tensor,
        record: dict[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("comparison pass is already closed")
        step = int(step)
        if not 0 <= step < self.replay.steps or not self.replay.actual[step]:
            raise ValueError("comparison output can only be written for captured actual steps")
        if tuple(comparison_x0.shape) != self.replay.shape:
            raise ValueError("comparison output shape does not match the fused capture")
        _atomic_safetensors(
            self.directory / f"step-{step:04d}.safetensors",
            {"comparison_x0": comparison_x0},
        )
        self._written[step] = True
        self._records[step] = flatten_record(record)

    def close(self) -> None:
        if self._closed:
            return
        missing = [
            step
            for step, expected in enumerate(self.replay.actual)
            if expected and not self._written[step]
        ]
        if missing:
            raise RuntimeError(f"incomplete comparison pass: missing actual steps={missing}")
        _atomic_json(
            self.directory / "manifest.json",
            {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "complete": True,
                "comparison_label": self.label,
                "capture_fingerprint": self.replay.capture_fingerprint,
                "invocation_key": self.replay.key,
                "steps": self.replay.steps,
                "shape": list(self.replay.shape),
                "actual_model_evaluations": self.replay.actual,
                "records": self._records,
            },
        )
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        (self.directory / "INCOMPLETE").write_text(
            "Comparison sampler invocation did not complete; this pass is not reusable.\n",
            encoding="utf-8",
        )
        self._closed = True


@torch.no_grad()
def sample_refdelta_comparison_replay(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    calibration_id: str,
    comparison_label: str,
    telemetry_prefix: str = "refdelta_comparison_replay",
):
    """Evaluate one labeled MODEL on states captured from a previous fused run.

    The captured fused final sampler value is returned unchanged so Continuum's
    later chunks are conditioned on the original fused trajectory rather than a
    newly generated reference-model trajectory.
    """
    del disable
    extra_args = {} if extra_args is None else extra_args
    comparison_label = safe_comparison_label(comparison_label)
    if spectrum_bridge(extra_args) is not None:
        raise RuntimeError(
            "RefDelta Comparison Replay requires Spectrum to be disabled on the comparison run"
        )

    from comfy.k_diffusion import sampling as k_sampling
    import folder_paths

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    sigmas = k_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    steps = len(sigmas) - 1
    seed = extra_args.get("seed")
    replay = CalibrationReplay(
        Path(folder_paths.get_output_directory()),
        calibration_id,
        seed,
        steps,
        tuple(x.shape),
        sigmas,
    )
    if comparison_label == "ref2va":
        # Ref2VA-from-FL2VA decomposition is only meaningful against the exact
        # FL2VA pass belonging to this immutable fused capture.
        replay.comparison_manifest("fl2va")
    comparison_writer = ComparisonPassWriter(replay, comparison_label)
    writer = TelemetryWriter(
        Path(folder_paths.get_output_directory()) / "refdelta_telemetry",
        f"{telemetry_prefix}-{comparison_label}",
        seed,
    )
    s_in = x.new_ones([x.shape[0]])
    comparison_evaluations = 0
    all_comparison_identical = True
    completed = False
    try:
        for step in range(steps):
            state, fused_x0 = replay.load_step(step, x.device)
            baseline = dict(replay.records[step])
            baseline["comparison_label"] = comparison_label
            baseline["comparison_model_evaluated"] = replay.actual[step]
            if replay.actual[step]:
                comparison_x0 = model(state, sigmas[step] * s_in, **extra_args)
                comparison_evaluations += 1
                all_comparison_identical = all_comparison_identical and torch.equal(
                    comparison_x0,
                    fused_x0,
                )
                comparison_metrics = compare_fused_to_model(
                    state,
                    sigmas[step],
                    fused_x0,
                    comparison_x0,
                    replay.layout,
                )
                baseline["comparison"] = {comparison_label: comparison_metrics}
                if comparison_label == "ref2va":
                    fl2va_x0 = replay.load_comparison_step("fl2va", step, x.device)
                    baseline["ref_delta"] = compare_ref_delta(
                        fused_x0,
                        fl2va_x0,
                        comparison_x0,
                        replay.layout,
                    )
                comparison_writer.write_step(
                    step,
                    comparison_x0,
                    {
                        "comparison": {comparison_label: comparison_metrics},
                        "ref_delta": baseline.get("ref_delta", {}),
                    },
                )
                if callback is not None:
                    callback(
                        {
                            "x": state,
                            "i": step,
                            "sigma": sigmas[step],
                            "sigma_hat": sigmas[step],
                            "denoised": comparison_x0,
                        }
                    )
            elif callback is not None:
                callback(
                    {
                        "x": state,
                        "i": step,
                        "sigma": sigmas[step],
                        "sigma_hat": sigmas[step],
                        "denoised": fused_x0,
                    }
                )
            writer.write(baseline)
        if comparison_evaluations == 0:
            raise RuntimeError("RefDelta calibration capture contains no actual model evaluations")
        if all_comparison_identical:
            raise ValueError(
                f"RefDelta comparison '{comparison_label}' was bit-identical to the fused capture "
                "at every actual step; verify that the existing MODEL loader was switched"
            )
        writer.close()
        comparison_writer.close()
        completed = True
        return replay.load_final(x.device)
    finally:
        if not completed:
            comparison_writer.abort()


def sample_refdelta_reference_replay(*args, **kwargs):
    """Deprecated saved-workflow alias for a labeled Ref2VA comparison pass."""
    kwargs.setdefault("comparison_label", "ref2va")
    return sample_refdelta_comparison_replay(*args, **kwargs)


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "CalibrationCaptureWriter",
    "CalibrationReplay",
    "ComparisonPassWriter",
    "invocation_key",
    "safe_calibration_id",
    "safe_comparison_label",
    "sample_refdelta_comparison_replay",
    "sample_refdelta_reference_replay",
]
