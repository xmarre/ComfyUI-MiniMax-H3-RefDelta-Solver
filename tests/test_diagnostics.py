from __future__ import annotations

import pytest
import torch

from comfyui_refdelta_solver.diagnostics import compare_same_state
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

