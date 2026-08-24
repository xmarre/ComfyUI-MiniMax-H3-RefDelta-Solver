from __future__ import annotations

import pytest
import torch

from comfyui_refdelta_solver.coordinates import divided_difference, second_divided_difference
from comfyui_refdelta_solver.trajectory import (
    StreamLayout,
    TrajectoryHistory,
    adaptive_order_gates,
    bounded_trajectory_correction,
    cosine,
    rms,
    stochastic_multiplier,
    stochastic_pressure_from_ratio,
)


def test_cosine_handles_zero_vectors_without_hiding_one_sided_error():
    zero = torch.zeros(4)
    nonzero = torch.ones(4)
    assert cosine(zero, zero).item() == 1.0
    assert cosine(zero, nonzero).item() == 0.0


def test_nonuniform_divided_differences_are_exact_for_quadratic():
    coordinates = (7.0, 3.5, 1.25)
    values = [torch.tensor([coordinate**2]) for coordinate in coordinates]
    first_previous = divided_difference(values[1], values[0], coordinates[1], coordinates[0])
    first_current = divided_difference(values[2], values[1], coordinates[2], coordinates[1])
    assert first_previous.item() == pytest.approx(coordinates[1] + coordinates[0])
    assert first_current.item() == pytest.approx(coordinates[2] + coordinates[1])
    second = second_divided_difference(first_current, first_previous, coordinates[2], coordinates[0])
    assert second.item() == pytest.approx(2.0)


def test_near_zero_coordinate_span_fails_safely():
    value = torch.ones(2)
    assert divided_difference(value, value, 1.0, 1.0 + 1e-14) is None
    assert second_divided_difference(value, value, 1.0, 1.0 + 1e-14) is None


def test_adaptive_order_is_bounded_smooth_and_deterministic():
    risks = torch.linspace(0.0, 1.0, 101)
    stage2, stage3 = adaptive_order_gates(risks, True)
    assert torch.all((0.0 <= stage2) & (stage2 <= 1.0))
    assert torch.all((0.0 <= stage3) & (stage3 <= 1.0))
    assert torch.all(stage2[1:] <= stage2[:-1])
    assert torch.all(stage3[1:] <= stage3[:-1])
    assert torch.equal((stage2, stage3)[0], adaptive_order_gates(risks, True)[0])
    one2, one3 = adaptive_order_gates(risks, False)
    assert torch.all(one2 == 1.0)
    assert torch.all(one3 == 1.0)


def test_stochastic_multiplier_bounds_and_terminal_inputs():
    risk = torch.linspace(0.0, 1.0, 100)
    multiplier = stochastic_multiplier(risk, 0.75, 0.2)
    assert torch.all(multiplier >= 0.2)
    assert torch.all(multiplier <= 1.0)
    assert torch.all(multiplier[1:] <= multiplier[:-1])
    assert torch.all(stochastic_multiplier(risk, 0.0, 0.0) == 1.0)


def test_stochastic_pressure_is_smooth_bounded_and_does_not_hard_saturate():
    ratios = torch.tensor([0.0, 0.1, 1.0, 3.0, 10.0])
    pressure = stochastic_pressure_from_ratio(ratios)
    expected = ratios / (1.0 + ratios)
    torch.testing.assert_close(pressure, expected)
    assert torch.all(pressure[1:] > pressure[:-1])
    assert pressure[-1] < 1.0


def test_correction_insufficient_history_returns_raw_identity():
    raw = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    corrected, norms = bounded_trajectory_correction(
        raw, None, None, None, -1.0, StreamLayout(5), 0.5, 0.5, 0.5, torch.tensor(1.0)
    )
    assert corrected is raw
    assert norms.keys() == {"video", "audio"}
    assert all(value.item() == 0.0 for value in norms.values())


def test_correction_is_bounded_separate_and_does_not_mutate_raw_history():
    raw = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0]], dtype=torch.float32)
    previous = torch.zeros_like(raw)
    first = torch.full_like(raw, 100.0)
    second = torch.full_like(raw, 50.0)
    raw_copy = raw.clone()
    previous_copy = previous.clone()
    corrected, norms = bounded_trajectory_correction(
        raw,
        previous,
        first,
        second,
        -0.5,
        StreamLayout(3),
        video_strength=0.5,
        audio_strength=0.1,
        bound=0.25,
        gate=torch.tensor(1.0),
    )
    assert torch.equal(raw, raw_copy)
    assert torch.equal(previous, previous_copy)
    assert corrected.data_ptr() != raw.data_ptr()
    video_allowed = rms(raw[..., :3] - previous[..., :3]) * 0.25
    audio_allowed = rms(raw[..., 3:] - previous[..., 3:]) * 0.25
    assert norms["video"] <= video_allowed + 1e-6
    assert norms["audio"] <= audio_allowed + 1e-6
    assert torch.isfinite(corrected).all()


def test_history_reset_shape_dtype_and_interruption_safety():
    history = TrajectoryHistory()
    raw = torch.ones((1, 6), dtype=torch.float32)
    history.commit(raw, 5.0, None)
    assert history.previous_raw is not raw
    history.previous_stochastic_ratios = {"combined": torch.tensor(0.5)}
    history.reset()
    assert history.previous_raw is None
    assert history.previous_first is None
    assert history.previous_stochastic_ratios is None

    history.commit(raw, 4.0, None)
    changed = torch.ones((1, 7), dtype=torch.float64)
    history.commit(changed, 3.0, None)
    assert history.previous_raw.shape == changed.shape
    assert history.previous_raw.dtype == changed.dtype
    assert history.previous_first is None


def test_stream_layout_rejects_stale_shape():
    with pytest.raises(ValueError, match="shorter"):
        StreamLayout(8).split(torch.zeros((1, 8)))


def test_stochastic_history_is_stream_specific_and_separate_from_trajectory_risk():
    history = TrajectoryHistory()
    history.previous_stochastic_ratios = {
        "video": torch.tensor(0.1),
        "audio": torch.tensor(0.9),
    }
    raw = torch.ones((1, 4))
    observation = history.observe(raw, 2.0, 1.0, StreamLayout(2), None, None, raw, 1.0)

    assert observation.trajectory_risk.item() == 0.0
    assert observation.stream_trajectory_risks["video"].item() == 0.0
    assert observation.stream_trajectory_risks["audio"].item() == 0.0
    assert observation.stream_stochastic_pressures["video"].item() == pytest.approx(0.1 / 1.1)
    assert observation.stream_stochastic_pressures["audio"].item() == pytest.approx(0.9 / 1.9)
    assert observation.stream_risks["video"].item() == pytest.approx(0.1 / 1.1)
    assert observation.stream_risks["audio"].item() == pytest.approx(0.9 / 1.9)
    assert observation.risk.item() == pytest.approx(0.9 / 1.9)
    assert observation.components["audio"]["previous_native_stochastic_ratio"].item() == pytest.approx(0.9)


def test_missing_stochastic_ratio_does_not_create_frozen_stream_risk_floor():
    history = TrajectoryHistory()
    history.previous_stochastic_ratios = {"video": torch.tensor(2.0)}
    raw = torch.ones((1, 4))
    observation = history.observe(raw, 2.0, 1.0, StreamLayout(2), None, None, raw, 1.0)

    assert observation.stream_risks["video"].item() == pytest.approx(2.0 / 3.0)
    assert observation.stream_risks["audio"].item() == 0.0
    assert "previous_stochastic_pressure" not in observation.components["audio"]
