from __future__ import annotations

import math

import pytest
import torch

from comfyui_refdelta_solver.comparison import compare_ref_delta
from comfyui_refdelta_solver.trajectory import StreamLayout


def test_ref_delta_metrics_match_known_video_and_audio_vectors():
    fl2va = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
    ref2va = torch.tensor([[2.0, 2.0, 10.0, 22.0]])
    fused = torch.tensor([[1.5, 2.0, 10.0, 21.0]])
    fields = compare_ref_delta(fused, fl2va, ref2va, StreamLayout(2))

    for stream in ("video", "audio"):
        assert fields[f"{stream}_defined"] is True
        assert fields[f"{stream}_cosine"] == pytest.approx(1.0)
        assert fields[f"{stream}_magnitude_ratio"] == pytest.approx(0.5)
        assert fields[f"{stream}_relative_residual"] == pytest.approx(0.5)
        assert fields[f"{stream}_projection_fraction"] == pytest.approx(0.5)
        assert fields[f"{stream}_orthogonal_residual"] == pytest.approx(0.0)


def test_ref_delta_zero_true_delta_is_explicitly_undefined_without_nan_or_inf():
    fl2va = torch.ones(1, 4)
    ref2va = fl2va.clone()
    fused = torch.tensor([[1.0, 2.0, 1.0, 0.0]])
    fields = compare_ref_delta(fused, fl2va, ref2va, StreamLayout(2))

    for stream in ("video", "audio"):
        assert fields[f"{stream}_defined"] is False
        for metric in (
            "cosine",
            "magnitude_ratio",
            "relative_residual",
            "projection_fraction",
            "orthogonal_residual",
        ):
            assert fields[f"{stream}_{metric}"] is None
    assert all(
        not isinstance(value, float) or math.isfinite(value)
        for value in fields.values()
    )


def test_ref_delta_rejects_shape_and_dtype_mismatch():
    value = torch.ones(1, 4)
    with pytest.raises(ValueError, match="shapes"):
        compare_ref_delta(value, value, torch.ones(1, 5), StreamLayout(2))
    with pytest.raises(ValueError, match="dtype"):
        compare_ref_delta(value, value.double(), value, StreamLayout(2))
