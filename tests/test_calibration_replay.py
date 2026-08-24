from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.calibration_replay import (
    CalibrationCaptureWriter,
    CalibrationReplay,
    invocation_key,
    safe_calibration_id,
    sample_refdelta_reference_replay,
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


def test_safe_calibration_id_and_invocation_key_are_deterministic():
    assert safe_calibration_id("  a/b:c  ") == "a_b_c"
    assert invocation_key(12, 3, (1, 4)) == "seed12-steps3-shape1x4"
    with pytest.raises(ValueError, match="filename-safe"):
        safe_calibration_id("///")


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


def test_reference_replay_evaluates_only_actual_steps_and_returns_fused_final(tmp_path, monkeypatch):
    sigmas, states, _fused, final = _write_capture(tmp_path, actual=(True, False))

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

    class ModelPatcher:
        @staticmethod
        def get_model_object(name):
            assert name == "model_sampling"
            return object()

    class ReferenceModel:
        def __init__(self):
            self.inner_model = SimpleNamespace(model_patcher=ModelPatcher())
            self.calls = []

        def __call__(self, state, sigma, **extra_args):
            self.calls.append((state.detach().clone(), sigma.detach().clone(), dict(extra_args)))
            return state * 0.5

    model = ReferenceModel()
    callbacks = []
    output = sample_refdelta_reference_replay(
        model,
        torch.zeros(1, 4),
        sigmas,
        extra_args={"seed": 17},
        callback=callbacks.append,
        calibration_id="unit-test",
        telemetry_prefix="replay-test",
    )

    assert len(model.calls) == 1
    assert torch.equal(model.calls[0][0], states[0])
    assert len(callbacks) == 2
    assert torch.equal(output, final)

    telemetry = sorted((tmp_path / "refdelta_telemetry").glob("replay-test-*.jsonl"))
    assert len(telemetry) == 1
    rows = telemetry[0].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert '"replay_reference_evaluated": true' in rows[0]
    assert '"replay_reference_evaluated": false' in rows[1]
    assert '"reference_video_x0_cosine"' in rows[0]
    assert '"reference_video_x0_cosine"' not in rows[1]
