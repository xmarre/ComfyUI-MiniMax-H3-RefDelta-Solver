from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .diagnostics import compare_same_state
from .spectrum_interop import spectrum_bridge
from .telemetry import TelemetryWriter, flatten_record
from .trajectory import StreamLayout


CALIBRATION_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_calibration_id(value: str) -> str:
    safe = _SAFE_ID.sub("_", str(value)).strip("._")
    if not safe:
        raise ValueError("calibration_id must contain at least one filename-safe character")
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
    """Read one completed capture and evaluate a reference model on its exact states."""

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


@torch.no_grad()
def sample_refdelta_reference_replay(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    calibration_id: str,
    telemetry_prefix: str = "refdelta_reference_replay",
):
    """Evaluate the current MODEL on states captured from a previous fused run.

    The captured fused final sampler value is returned unchanged so Continuum's
    later chunks are conditioned on the original fused trajectory rather than a
    newly generated reference-model trajectory.
    """
    del disable
    extra_args = {} if extra_args is None else extra_args
    if spectrum_bridge(extra_args) is not None:
        raise RuntimeError(
            "RefDelta Reference Replay requires Spectrum to be disabled on the reference run"
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
    writer = TelemetryWriter(
        Path(folder_paths.get_output_directory()) / "refdelta_telemetry",
        telemetry_prefix,
        seed,
    )
    s_in = x.new_ones([x.shape[0]])
    reference_evaluations = 0
    all_reference_identical = True
    try:
        for step in range(steps):
            state, fused_x0 = replay.load_step(step, x.device)
            baseline = dict(replay.records[step])
            baseline["replay_reference_evaluated"] = replay.actual[step]
            if replay.actual[step]:
                reference_x0 = model(state, sigmas[step] * s_in, **extra_args)
                reference_evaluations += 1
                all_reference_identical = all_reference_identical and torch.equal(
                    reference_x0,
                    fused_x0,
                )
                baseline["reference"] = compare_same_state(
                    state,
                    sigmas[step],
                    fused_x0,
                    reference_x0,
                    replay.layout,
                )
                if callback is not None:
                    callback(
                        {
                            "x": state,
                            "i": step,
                            "sigma": sigmas[step],
                            "sigma_hat": sigmas[step],
                            "denoised": reference_x0,
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
        if reference_evaluations == 0:
            raise RuntimeError("RefDelta calibration capture contains no actual model evaluations")
        if all_reference_identical:
            raise ValueError(
                "RefDelta Reference Replay was bit-identical to the fused capture at every actual step; "
                "verify that the existing MODEL loader was switched to genuine Ref2VA"
            )
        return replay.load_final(x.device)
    finally:
        writer.close()


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationCaptureWriter",
    "CalibrationReplay",
    "invocation_key",
    "safe_calibration_id",
    "sample_refdelta_reference_replay",
]
