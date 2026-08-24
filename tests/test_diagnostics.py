from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from comfyui_refdelta_solver.diagnostics import (
    RefDeltaReferenceGuiderMixin,
    compare_same_state,
    spectrum_step_is_forecast,
)
from comfyui_refdelta_solver.trajectory import StreamLayout


def test_same_state_reference_metrics_are_stream_separate():
    state = torch.tensor([[4.0, 2.0, 1.0, 8.0, 4.0]])
    reference = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0]])
    fused = reference.clone()
    fields = compare_same_state(state, torch.tensor(0.5), fused, reference, StreamLayout(3))
    assert fields["video_x0_cosine"] == pytest.approx(1.0)
    assert fields["audio_x0_cosine"] == pytest.approx(1.0)
    assert fields["video_x0_relative_error"] == pytest.approx(0.0)
    assert fields["audio_velocity_relative_error"] == pytest.approx(0.0)


def test_same_state_reference_metrics_detect_error():
    state = torch.ones((1, 4))
    reference = torch.zeros((1, 4))
    fused = torch.tensor([[0.0, 1.0, 0.0, -1.0]])
    fields = compare_same_state(state, torch.tensor(0.25), fused, reference, StreamLayout(2))
    assert fields["video_x0_relative_error"] > 0.0
    assert fields["audio_velocity_relative_error"] > 0.0


def test_same_state_reference_rejects_shape_device_or_dtype_mismatch():
    state = torch.ones((1, 4))
    with pytest.raises(ValueError, match="shapes"):
        compare_same_state(state, torch.tensor(1.0), state, torch.ones((1, 5)), StreamLayout(2))
    with pytest.raises(ValueError, match="dtype"):
        compare_same_state(state, torch.tensor(1.0), state, state.double(), StreamLayout(2))


def test_spectrum_forecast_marker_gates_reference_evaluation_only_when_explicit():
    assert not spectrum_step_is_forecast(None)
    assert not spectrum_step_is_forecast({})
    assert not spectrum_step_is_forecast({"transformer_options": {}})
    assert not spectrum_step_is_forecast(
        {"transformer_options": {"spectrum_h3_actual": True}}
    )
    assert spectrum_step_is_forecast(
        {"transformer_options": {"spectrum_h3_actual": False}}
    )
    assert not spectrum_step_is_forecast(
        {"transformer_options": {"spectrum_h3_actual": 0}}
    )


def test_predict_noise_suppresses_forecast_reference_and_preserves_actual_sequence(monkeypatch):
    fused_calls = []
    reference_calls = []

    class BaseGuider:
        def predict_noise(self, x, timestep, model_options=None, seed=None):
            fused_calls.append((model_options, seed))
            return x + 1.0

    class DiagnosticGuider(RefDeltaReferenceGuiderMixin, BaseGuider):
        pass

    def sampling_function(
        inner_model,
        x,
        timestep,
        negative,
        positive,
        cfg,
        model_options=None,
        seed=None,
    ):
        reference_calls.append(
            (inner_model, x.clone(), timestep.clone(), negative, positive, cfg, model_options, seed)
        )
        return x - 1.0

    comfy_module = ModuleType("comfy")
    samplers_module = ModuleType("comfy.samplers")
    samplers_module.sampling_function = sampling_function
    comfy_module.samplers = samplers_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers_module)

    guider = DiagnosticGuider()
    guider.reference_inner_model = object()
    guider.reference_conds = {"negative": "negative", "positive": "positive"}
    guider.reference_runtime_model_options = {"reference": True}
    guider.cfg = 1.0
    guider._refdelta_reference_result = None
    guider._refdelta_reference_call_index = 0

    x = torch.tensor([[0.25, -0.5]])
    forecast_sigma = torch.tensor([0.8])
    actual_sigma = torch.tensor([0.6])
    forecast_options = {"transformer_options": {"spectrum_h3_actual": False}}
    actual_options = {"transformer_options": {"spectrum_h3_actual": True}}

    forecast = guider.predict_noise(x, forecast_sigma, model_options=forecast_options, seed=17)
    torch.testing.assert_close(forecast, x + 1.0)
    assert len(fused_calls) == 1
    assert reference_calls == []
    assert guider._refdelta_reference_result is None
    assert guider._refdelta_reference_call_index == 0

    actual = guider.predict_noise(x, actual_sigma, model_options=actual_options, seed=17)
    torch.testing.assert_close(actual, x + 1.0)
    assert len(fused_calls) == 2
    assert len(reference_calls) == 1
    assert reference_calls[0][2].item() == pytest.approx(0.6)
    assert guider._refdelta_reference_result is not None
    assert guider._refdelta_reference_result.call_index == 0
    torch.testing.assert_close(guider._refdelta_reference_result.sigma, actual_sigma)
    torch.testing.assert_close(guider._refdelta_reference_result.denoised, x - 1.0)
    assert guider._refdelta_reference_call_index == 1
