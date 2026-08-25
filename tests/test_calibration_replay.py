from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.calibration_replay import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationCaptureWriter,
    CalibrationReplay,
    ComparisonPassWriter,
    invocation_key,
    safe_calibration_id,
    safe_comparison_label,
    sample_refdelta_comparison_replay,
    sample_refdelta_reference_replay,
)
from comfyui_refdelta_solver.spectrum_interop import (
    SPECTRUM_BRIDGE_KEY,
    SPECTRUM_INTEROP_CONTRACT,
)
from comfyui_refdelta_solver.trajectory import StreamLayout


def _write_capture(tmp_path, *, actual=(True, True)):
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    shape = (1, 4)
    writer = CalibrationCaptureWriter(
        tmp_path,
        "unit-test",
        17,
        2,
        shape,
        sigmas,
        StreamLayout(2),
    )
    states = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[0.5, 1.5, 2.5, 3.5]]),
    ]
    fused = [state * 0.25 for state in states]
    for step in range(2):
        writer.write_step(step, states[step], fused[step], actual=actual[step])
        writer.write_record(
            step,
            {
                "step": step,
                "video": {"denoised_rms": torch.tensor(float(step + 1))},
            },
        )
    final = torch.tensor([[9.0, 8.0, 7.0, 6.0]])
    writer.close(final)
    return sigmas, states, fused, final


def _install_replay_runtime(tmp_path, monkeypatch):
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    sampling = ModuleType("comfy.k_diffusion.sampling")
    sampling.offset_first_sigma_for_snr = lambda values, model_sampling: values
    k_diffusion = ModuleType("comfy.k_diffusion")
    k_diffusion.sampling = sampling
    comfy = sys.modules.get("comfy") or ModuleType("comfy")
    comfy.k_diffusion = k_diffusion
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", k_diffusion)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)


class _ModelPatcher:
    @staticmethod
    def get_model_object(name):
        assert name == "model_sampling"
        return object()


class _ReferenceModel:
    def __init__(self, scale=0.5):
        self.inner_model = SimpleNamespace(model_patcher=_ModelPatcher())
        self.calls = []
        self.scale = scale

    def __call__(self, state, sigma, **extra_args):
        self.calls.append((state.detach().clone(), sigma.detach().clone(), dict(extra_args)))
        return state * self.scale


def test_safe_calibration_id_and_invocation_key_are_deterministic():
    assert safe_calibration_id("  a/b:c  ") == "a_b_c"
    assert invocation_key(12, 3, (1, 4)) == "seed12-steps3-shape1x4"
    with pytest.raises(ValueError, match="filename-safe"):
        safe_calibration_id("///")
    assert safe_comparison_label(" Ref2VA/run ") == "ref2va_run"
    with pytest.raises(ValueError, match="reserved"):
        safe_comparison_label("comparisons")


def test_capture_round_trip_persists_steps_records_and_final(tmp_path):
    sigmas, states, fused, final = _write_capture(tmp_path, actual=(True, False))

    replay = CalibrationReplay(
        tmp_path,
        "unit-test",
        17,
        2,
        (1, 4),
        sigmas,
    )

    assert replay.actual == [True, False]
    assert replay.records[0]["video_denoised_rms"] == pytest.approx(1.0)
    assert replay.records[1]["video_denoised_rms"] == pytest.approx(2.0)
    for step in range(2):
        state, fused_x0 = replay.load_step(step, torch.device("cpu"))
        assert torch.equal(state, states[step])
        assert torch.equal(fused_x0, fused[step])
    assert torch.equal(replay.load_final(torch.device("cpu")), final)


def test_replay_rejects_schedule_mismatch(tmp_path):
    _write_capture(tmp_path)
    with pytest.raises(ValueError, match="sigma schedule"):
        CalibrationReplay(
            tmp_path,
            "unit-test",
            17,
            2,
            (1, 4),
            torch.tensor([1.0, 0.4, 0.0]),
        )


@pytest.mark.parametrize(
    "calibration_id,seed,steps,shape,sigmas",
    (
        ("wrong-id", 17, 2, (1, 4), torch.tensor([1.0, 0.5, 0.0])),
        ("unit-test", 18, 2, (1, 4), torch.tensor([1.0, 0.5, 0.0])),
        ("unit-test", 17, 3, (1, 4), torch.tensor([1.0, 0.7, 0.3, 0.0])),
        ("unit-test", 17, 2, (1, 5), torch.tensor([1.0, 0.5, 0.0])),
    ),
)
def test_replay_rejects_wrong_capture_identity(
    tmp_path,
    calibration_id,
    seed,
    steps,
    shape,
    sigmas,
):
    _write_capture(tmp_path)
    with pytest.raises(FileNotFoundError, match="capture not found"):
        CalibrationReplay(tmp_path, calibration_id, seed, steps, shape, sigmas)


def test_replay_rejects_schema_mismatch(tmp_path):
    _write_capture(tmp_path)
    manifest = next((tmp_path / "refdelta_calibration").glob("**/manifest.json"))
    payload = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = CALIBRATION_SCHEMA_VERSION - 1
    manifest.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported schema"):
        CalibrationReplay(
            tmp_path,
            "unit-test",
            17,
            2,
            (1, 4),
            torch.tensor([1.0, 0.5, 0.0]),
        )


def test_new_capture_replaces_stale_incomplete_invocation(tmp_path):
    sigmas = torch.tensor([1.0, 0.0])
    first = CalibrationCaptureWriter(
        tmp_path,
        "replace",
        5,
        1,
        (1, 2),
        sigmas,
        StreamLayout(None),
    )
    first.write_step(0, torch.ones(1, 2), torch.zeros(1, 2), actual=True)
    first.abort()
    assert (first.directory / "INCOMPLETE").is_file()
    with pytest.raises(FileNotFoundError, match="capture not found"):
        CalibrationReplay(
            tmp_path,
            "replace",
            5,
            1,
            (1, 2),
            sigmas,
        )

    second = CalibrationCaptureWriter(
        tmp_path,
        "replace",
        5,
        1,
        (1, 2),
        sigmas,
        StreamLayout(None),
    )
    assert not (second.directory / "INCOMPLETE").exists()
    second.write_step(0, torch.ones(1, 2), torch.zeros(1, 2), actual=True)
    second.write_record(0, {"step": 0})
    second.close(torch.full((1, 2), 3.0))
    assert (second.directory / "manifest.json").is_file()


def test_completed_capture_key_is_never_silently_overwritten(tmp_path):
    sigmas = torch.tensor([1.0, 0.0])
    writer = CalibrationCaptureWriter(
        tmp_path,
        "collision",
        5,
        1,
        (1, 2),
        sigmas,
        StreamLayout(None),
    )
    writer.write_step(0, torch.ones(1, 2), torch.zeros(1, 2), actual=True)
    writer.write_record(0, {"step": 0})
    writer.close(torch.full((1, 2), 3.0))

    with pytest.raises(FileExistsError, match="new calibration_id"):
        CalibrationCaptureWriter(
            tmp_path,
            "collision",
            5,
            1,
            (1, 2),
            sigmas,
            StreamLayout(None),
        )


def test_comparison_replay_evaluates_only_actual_steps_and_returns_fused_final(tmp_path, monkeypatch):
    sigmas, states, _fused, final = _write_capture(tmp_path, actual=(True, False))
    _install_replay_runtime(tmp_path, monkeypatch)

    model = _ReferenceModel(scale=0.5)
    callbacks = []
    output = sample_refdelta_comparison_replay(
        model,
        torch.zeros(1, 4),
        sigmas,
        extra_args={"seed": 17},
        callback=callbacks.append,
        calibration_id="unit-test",
        comparison_label="fl2va",
        telemetry_prefix="replay-test",
    )

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], states[0])
    assert len(callbacks) == 2
    assert torch.equal(output, final)

    telemetry = sorted((tmp_path / "refdelta_telemetry").glob("replay-test-fl2va-*.jsonl"))
    assert len(telemetry) == 1
    rows = telemetry[0].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert '"comparison_model_evaluated": true' in rows[0]
    assert '"comparison_model_evaluated": false' in rows[1]
    assert '"comparison_fl2va_video_x0_cosine"' in rows[0]
    assert '"comparison_fl2va_video_x0_cosine"' not in rows[1]
    comparison_manifest = next(
        (tmp_path / "refdelta_calibration").glob("**/comparisons/fl2va/manifest.json")
    )
    assert comparison_manifest.is_file()


def test_comparison_replay_rejects_spectrum_on_comparison_run(tmp_path, monkeypatch):
    sigmas, _states, _fused, _final = _write_capture(tmp_path)
    _install_replay_runtime(tmp_path, monkeypatch)

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            return True

        @staticmethod
        def publish_stochastic_increment(step_id, increment):
            return None

    extra_args = {
        "seed": 17,
        "model_options": {
            "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
        },
    }
    with pytest.raises(RuntimeError, match="Spectrum.*disabled"):
        sample_refdelta_comparison_replay(
            _ReferenceModel(),
            torch.zeros(1, 4),
            sigmas,
            extra_args=extra_args,
            calibration_id="unit-test",
            comparison_label="fl2va",
        )


def test_comparison_replay_rejects_bit_identical_model_output(tmp_path, monkeypatch):
    sigmas, _states, _fused, _final = _write_capture(tmp_path, actual=(True, False))
    _install_replay_runtime(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="MODEL loader was switched"):
        sample_refdelta_comparison_replay(
            _ReferenceModel(scale=0.25),
            torch.zeros(1, 4),
            sigmas,
            extra_args={"seed": 17},
            calibration_id="unit-test",
            comparison_label="fl2va",
        )

    incomplete = next(
        (tmp_path / "refdelta_calibration").glob("**/comparisons/fl2va/INCOMPLETE")
    )
    assert incomplete.is_file()


def test_ref2va_replay_requires_and_combines_completed_fl2va_pass(tmp_path, monkeypatch):
    sigmas, _states, _fused, final = _write_capture(tmp_path, actual=(True, False))
    _install_replay_runtime(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="fl2va"):
        sample_refdelta_comparison_replay(
            _ReferenceModel(scale=0.75),
            torch.zeros(1, 4),
            sigmas,
            extra_args={"seed": 17},
            calibration_id="unit-test",
            comparison_label="ref2va",
        )

    fl2va_output = sample_refdelta_comparison_replay(
        _ReferenceModel(scale=0.5),
        torch.zeros(1, 4),
        sigmas,
        extra_args={"seed": 17},
        calibration_id="unit-test",
        comparison_label="fl2va",
    )
    ref2va_output = sample_refdelta_comparison_replay(
        _ReferenceModel(scale=0.75),
        torch.zeros(1, 4),
        sigmas,
        extra_args={"seed": 17},
        calibration_id="unit-test",
        comparison_label="ref2va",
    )
    assert torch.equal(fl2va_output, final)
    assert torch.equal(ref2va_output, final)

    telemetry = sorted((tmp_path / "refdelta_telemetry").glob("*ref2va-*.jsonl"))
    rows = [__import__("json").loads(line) for line in telemetry[-1].read_text().splitlines()]
    assert rows[0]["comparison_ref2va_video_x0_cosine"] == pytest.approx(1.0)
    assert rows[0]["ref_delta_video_cosine"] == pytest.approx(-1.0)
    assert rows[0]["ref_delta_video_magnitude_ratio"] == pytest.approx(1.0)
    assert rows[0]["ref_delta_video_projection_fraction"] == pytest.approx(-1.0)
    assert rows[0]["ref_delta_video_relative_residual"] == pytest.approx(2.0)
    assert rows[0]["ref_delta_video_orthogonal_residual"] == pytest.approx(0.0)


def test_comparison_labels_do_not_overwrite_and_completed_passes_collide(tmp_path):
    sigmas, _states, _fused, _final = _write_capture(tmp_path)
    replay = CalibrationReplay(tmp_path, "unit-test", 17, 2, (1, 4), sigmas)
    first = ComparisonPassWriter(replay, "fl2va")
    second = ComparisonPassWriter(replay, "ref2va-control")
    for step in range(2):
        first.write_step(step, torch.ones(1, 4), {"step": step})
        second.write_step(step, torch.full((1, 4), 2.0), {"step": step})
    first.close()
    second.close()

    assert replay.load_comparison_step("fl2va", 0, torch.device("cpu")).eq(1.0).all()
    assert replay.load_comparison_step("ref2va-control", 0, torch.device("cpu")).eq(2.0).all()
    with pytest.raises(FileExistsError, match="already exists"):
        ComparisonPassWriter(replay, "fl2va")


def test_incomplete_comparison_pass_is_replaceable(tmp_path):
    sigmas, _states, _fused, _final = _write_capture(tmp_path)
    replay = CalibrationReplay(tmp_path, "unit-test", 17, 2, (1, 4), sigmas)
    first = ComparisonPassWriter(replay, "fl2va")
    first.write_step(0, torch.ones(1, 4), {"step": 0})
    first.abort()
    assert (first.directory / "INCOMPLETE").is_file()

    second = ComparisonPassWriter(replay, "fl2va")
    assert not (second.directory / "INCOMPLETE").exists()
    for step in range(2):
        second.write_step(step, torch.ones(1, 4), {"step": step})
    second.close()
    assert (second.directory / "manifest.json").is_file()


def test_legacy_reference_replay_alias_uses_ref2va_label(tmp_path, monkeypatch):
    sigmas, _states, _fused, _final = _write_capture(tmp_path)
    _install_replay_runtime(tmp_path, monkeypatch)
    replay = CalibrationReplay(tmp_path, "unit-test", 17, 2, (1, 4), sigmas)
    fl2va = ComparisonPassWriter(replay, "fl2va")
    for step in range(2):
        fl2va.write_step(step, torch.zeros(1, 4), {"step": step})
    fl2va.close()
    output = sample_refdelta_reference_replay(
        _ReferenceModel(scale=0.75),
        torch.zeros(1, 4),
        sigmas,
        extra_args={"seed": 17},
        calibration_id="unit-test",
    )
    assert output.shape == (1, 4)
