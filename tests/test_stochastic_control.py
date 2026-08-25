from __future__ import annotations

import pytest
import torch

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler import (
    _stochastic_control_record,
    _write_stability_maps,
)
from comfyui_refdelta_solver.stochastic_control import (
    StochasticStabilityController,
    diffusion_change_ratio,
    smooth_stability_map,
    temporal_motion_ratio,
)
from comfyui_refdelta_solver.trajectory import (
    StreamLayout,
    TrajectoryObservation,
    stochastic_multiplier,
)
from comfyui_refdelta_solver.telemetry import TelemetryWriter


def _observation(video_risk: float, audio_risk: float) -> TrajectoryObservation:
    scalar = torch.tensor(max(video_risk, audio_risk), dtype=torch.float32)
    zero = torch.tensor(0.0)
    return TrajectoryObservation(
        first=None,
        second=None,
        first_direction_cosine=None,
        movement_rms=zero,
        risk=scalar,
        stream_risks={
            "video": torch.tensor(video_risk),
            "audio": torch.tensor(audio_risk),
        },
        trajectory_risk=zero,
        stream_trajectory_risks={"video": zero, "audio": zero},
        stochastic_pressure=zero,
        stream_stochastic_pressures={},
        components={"video": {}, "audio": {}},
    )


def _layout() -> StreamLayout:
    return StreamLayout(12, (1, 3, 1, 4))


def _packed(video: torch.Tensor, audio_elements: int = 2) -> torch.Tensor:
    return torch.cat((video.reshape(video.shape[0], -1), video.new_zeros((video.shape[0], audio_elements))), dim=-1)


def _spatial_config(**overrides) -> RefDeltaSamplerConfig:
    values = {
        "stochastic_control_mode": "spatiotemporal_stability",
        "stochastic_adaptation_strength": 0.75,
        "minimum_stochastic_multiplier": 0.1,
        "static_video_stochastic_adaptation_strength": 0.50,
        "video_stability_motion_low": 0.02,
        "video_stability_motion_high": 0.10,
        "video_stability_diffusion_low": 0.02,
        "video_stability_diffusion_high": 0.15,
        "video_stability_diffusion_weight": 0.5,
        "video_stability_spatial_radius": 0,
        "video_stability_temporal_radius": 0,
        "video_stability_ema": 0.0,
        "video_stability_start_fraction": 0.0,
        "video_stability_full_fraction": 0.0,
    }
    values.update(overrides)
    return RefDeltaSamplerConfig(**values)


def test_stream_layout_round_trip_and_validation():
    layout = _layout()
    packed = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    streams = layout.split(packed)
    latent = layout.video_to_latent(streams["video"])
    assert latent.shape == (2, 1, 3, 1, 4)
    torch.testing.assert_close(layout.latent_to_video(latent), streams["video"])
    torch.testing.assert_close(layout.combine(streams["video"], streams["audio"]), packed)

    packed_3d = packed.unsqueeze(1)
    streams_3d = layout.split(packed_3d)
    latent_3d = layout.video_to_latent(streams_3d["video"])
    torch.testing.assert_close(latent_3d, latent)
    torch.testing.assert_close(
        layout.latent_to_video(latent_3d, packed_like=streams_3d["video"]),
        streams_3d["video"],
    )

    with pytest.raises(ValueError, match="video shape"):
        StreamLayout(12).video_to_latent(streams["video"])
    with pytest.raises(ValueError, match="does not match"):
        StreamLayout(12, (1, 2, 2, 2))


def test_legacy_global_reproduces_pre_feature_gate_and_increment():
    config = RefDeltaSamplerConfig(
        stochastic_control_mode="legacy_global",
        stochastic_adaptation_strength=0.75,
        minimum_stochastic_multiplier=0.2,
    )
    controller = StochasticStabilityController(config, _layout())
    observation = _observation(0.4, 0.8)
    native = torch.linspace(-1.0, 1.0, 14).reshape(1, 14)
    endpoint = torch.tensor(0.6)
    result = controller.apply(native, observation, endpoint, 2, 10)

    adapted = stochastic_multiplier(observation.risk, 0.75, 0.2)
    expected_gate = 1.0 + endpoint * (adapted - 1.0)
    torch.testing.assert_close(result.increment, native * expected_gate)
    torch.testing.assert_close(result.compatibility_gate, expected_gate)


def test_streamwise_risks_are_independent():
    config = RefDeltaSamplerConfig(
        stochastic_control_mode="streamwise",
        stochastic_adaptation_strength=0.75,
        minimum_stochastic_multiplier=0.1,
    )
    native = torch.ones((1, 14))
    first = StochasticStabilityController(config, _layout()).apply(
        native, _observation(0.7, 0.2), torch.tensor(1.0), 1, 5
    )
    audio_changed = StochasticStabilityController(config, _layout()).apply(
        native, _observation(0.7, 0.9), torch.tensor(1.0), 1, 5
    )
    video_changed = StochasticStabilityController(config, _layout()).apply(
        native, _observation(0.3, 0.2), torch.tensor(1.0), 1, 5
    )
    torch.testing.assert_close(first.video_applied_gate, audio_changed.video_applied_gate)
    assert first.audio_applied_gate != audio_changed.audio_applied_gate
    torch.testing.assert_close(first.audio_applied_gate, video_changed.audio_applied_gate)
    assert first.video_applied_gate != video_changed.video_applied_gate


def test_temporal_motion_separates_static_and_moving_regions():
    video = torch.ones((1, 2, 4, 1, 4))
    video[:, :, :, :, 2:] = torch.arange(4).reshape(1, 1, 4, 1, 1)
    ratio = temporal_motion_ratio(video, 0.1)
    assert ratio[:, :, :, :, :2].amax().item() == pytest.approx(0.0)
    assert ratio[:, :, :, :, 2:].mean() > 0.5


def test_mixed_static_moving_mask_and_gate_targets():
    current = torch.ones((1, 1, 3, 1, 4))
    current[:, :, :, :, 2:] = torch.arange(3).reshape(1, 1, 3, 1, 1)
    packed = _packed(current)
    controller = StochasticStabilityController(_spatial_config(), _layout())
    controller.update_actual(packed, packed.clone(), 2)
    result = controller.apply(
        torch.ones_like(packed),
        _observation(1.0, 0.0),
        torch.tensor(1.0),
        2,
        5,
    )
    static_restore = result.restore_mask[:, :, :, :, :2].mean()
    moving_restore = result.restore_mask[:, :, :, :, 2:].mean()
    assert static_restore > 0.99
    assert moving_restore < 0.01
    dynamic = result.video_dynamic_target_gate
    static = result.video_static_target_gate
    static_gate = result.video_applied_gate[:, :, :, :, :2].mean()
    moving_gate = result.video_applied_gate[:, :, :, :, 2:].mean()
    assert torch.abs(static_gate - static) < torch.abs(static_gate - dynamic)
    assert torch.abs(moving_gate - dynamic) < torch.abs(moving_gate - static)


def test_diffusion_instability_suppresses_restoration_and_weight_zero_ignores_it():
    current = torch.ones((1, 1, 3, 1, 4))
    previous = torch.full_like(current, -4.0)
    current_packed = _packed(current)
    previous_packed = _packed(previous)

    weighted = StochasticStabilityController(
        _spatial_config(video_stability_diffusion_weight=1.0),
        _layout(),
    )
    weighted.update_actual(current_packed, previous_packed, 2)
    weighted_result = weighted.apply(
        torch.ones_like(current_packed), _observation(1.0, 0.0), torch.tensor(1.0), 2, 5
    )
    assert weighted_result.restore_mask.max() < 0.01

    temporal_only = StochasticStabilityController(
        _spatial_config(video_stability_diffusion_weight=0.0),
        _layout(),
    )
    temporal_only.update_actual(current_packed, previous_packed, 2)
    temporal_result = temporal_only.apply(
        torch.ones_like(current_packed), _observation(1.0, 0.0), torch.tensor(1.0), 2, 5
    )
    assert temporal_result.restore_mask.min() > 0.99


@pytest.mark.parametrize(
    "overrides",
    (
        {"video_stability_restore_strength": 0.0},
        {"static_video_stochastic_adaptation_strength": 0.75},
    ),
)
def test_spatial_mask_is_irrelevant_when_policies_collapse(overrides):
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    spatial_config = _spatial_config(**overrides)
    spatial = StochasticStabilityController(spatial_config, _layout())
    spatial.update_actual(packed, packed.clone(), 2)
    spatial_result = spatial.apply(
        torch.ones_like(packed), _observation(0.9, 0.6), torch.tensor(1.0), 2, 5
    )
    streamwise = StochasticStabilityController(
        RefDeltaSamplerConfig(
            stochastic_control_mode="streamwise",
            stochastic_adaptation_strength=0.75,
            minimum_stochastic_multiplier=0.1,
        ),
        _layout(),
    ).apply(torch.ones_like(packed), _observation(0.9, 0.6), torch.tensor(1.0), 2, 5)
    torch.testing.assert_close(spatial_result.increment, streamwise.increment)


def test_zero_base_strength_keeps_all_new_modes_mathematically_inactive():
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    controller = StochasticStabilityController(
        _spatial_config(stochastic_adaptation_strength=0.0),
        _layout(),
    )
    controller.update_actual(packed, packed.clone(), 2)
    native = torch.randn_like(packed)
    result = controller.apply(native, _observation(1.0, 1.0), torch.tensor(1.0), 2, 5)
    torch.testing.assert_close(result.increment, native)


def test_spatial_control_preserves_packed_rank_and_bfloat16_dtype():
    current = torch.ones((1, 1, 3, 1, 4), dtype=torch.bfloat16)
    packed = _packed(current).unsqueeze(1)
    controller = StochasticStabilityController(_spatial_config(), _layout())
    controller.update_actual(packed, packed.clone(), 2)
    native = torch.ones_like(packed)
    result = controller.apply(
        native,
        _observation(1.0, 1.0),
        torch.tensor(1.0, dtype=torch.bfloat16),
        2,
        5,
    )
    assert result.increment.shape == native.shape
    assert result.increment.dtype == torch.bfloat16
    assert result.video_applied_gate.dtype == torch.bfloat16


def test_ratios_and_masks_are_finite_and_bounded_for_pathological_inputs():
    values = torch.tensor(
        [0.0, 1e-30, 1e30, float("nan"), float("inf"), -float("inf")],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 1, 2)
    motion = temporal_motion_ratio(values, 0.1)
    diffusion = diffusion_change_ratio(values, torch.zeros_like(values), 0.1)
    assert torch.isfinite(motion).all() and torch.all((0 <= motion) & (motion <= 2))
    assert torch.isfinite(diffusion).all() and torch.all((0 <= diffusion) & (diffusion <= 2))


def test_separable_smoothing_preserves_shape_edges_and_axis_identity():
    ones = torch.ones((1, 1, 2, 3, 4))
    smoothed = smooth_stability_map(ones, spatial_radius=2, temporal_radius=2)
    assert smoothed.shape == ones.shape
    torch.testing.assert_close(smoothed, ones)
    torch.testing.assert_close(
        smooth_stability_map(ones, spatial_radius=0, temporal_radius=0),
        ones,
    )


def test_actual_only_ema_is_reused_by_forecast_without_mutation():
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    controller = StochasticStabilityController(
        _spatial_config(video_stability_ema=0.7),
        _layout(),
    )
    controller.update_actual(packed, packed.clone(), 1)
    cached = controller.evidence.ema_restore_mask.clone()
    source = controller.evidence.source_actual_step
    controller.apply(
        torch.ones_like(packed),
        _observation(0.9, 0.9),
        torch.tensor(1.0),
        2,
        5,
    )
    torch.testing.assert_close(controller.evidence.ema_restore_mask, cached)
    assert controller.evidence.source_actual_step == source


def test_wild_forecast_cannot_contaminate_actual_stability_evidence():
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    controller = StochasticStabilityController(_spatial_config(), _layout())
    controller.update_actual(packed, packed.clone(), 1)
    cached = controller.evidence.ema_restore_mask.clone()
    wild_forecast_increment = torch.full_like(packed, 1e20)
    controller.apply(
        wild_forecast_increment,
        _observation(1.0, 1.0),
        torch.tensor(1.0),
        2,
        5,
    )
    torch.testing.assert_close(controller.evidence.ema_restore_mask, cached)


def test_shape_change_resets_cached_mask_and_slew_state():
    first_layout = _layout()
    first = torch.ones((1, 1, 3, 1, 4))
    controller = StochasticStabilityController(
        _spatial_config(stochastic_gate_slew_limit=0.1),
        first_layout,
    )
    first_packed = _packed(first)
    controller.update_actual(first_packed, first_packed.clone(), 1)
    controller.apply(first_packed, _observation(1.0, 1.0), torch.tensor(1.0), 1, 5)
    assert controller.evidence is not None
    assert controller._previous_video_gate is not None

    controller.layout = StreamLayout(8, (1, 2, 1, 4))
    changed = torch.ones((1, 1, 2, 1, 4))
    changed_packed = _packed(changed)
    controller.update_actual(changed_packed, None, 2)
    assert controller.evidence.source_actual_step == 2
    assert controller.evidence.ema_restore_mask.shape[2] == 2
    assert controller._previous_video_gate is None


def test_slew_limit_bounds_adapted_change_and_endpoint_still_restores_native():
    config = RefDeltaSamplerConfig(
        stochastic_control_mode="streamwise",
        stochastic_adaptation_strength=0.75,
        minimum_stochastic_multiplier=0.1,
        stochastic_gate_slew_limit=0.1,
    )
    controller = StochasticStabilityController(config, _layout())
    native = torch.ones((1, 14))
    controller.apply(native, _observation(0.0, 0.0), torch.tensor(1.0), 0, 5)
    limited = controller.apply(
        native, _observation(1.0, 1.0), torch.tensor(1.0), 1, 5, collect_stats=True
    )
    assert limited.video_applied_gate.item() == pytest.approx(0.9)
    assert limited.audio_applied_gate.item() == pytest.approx(0.9)
    assert limited.slew_applied.item() == 1.0

    endpoint = controller.apply(
        native, _observation(1.0, 1.0), torch.tensor(0.0), 2, 5
    )
    torch.testing.assert_close(endpoint.increment, native)


def test_endpoint_restoration_applies_after_spatial_and_audio_control():
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    controller = StochasticStabilityController(_spatial_config(), _layout())
    controller.update_actual(packed, packed.clone(), 2)
    result = controller.apply(
        torch.ones_like(packed), _observation(1.0, 1.0), torch.tensor(0.25), 2, 5
    )
    expected_video = 1.0 + 0.25 * (
        torch.lerp(
            result.video_dynamic_target_gate,
            result.video_static_target_gate,
            result.restore_mask,
        )
        - 1.0
    )
    torch.testing.assert_close(result.video_applied_gate, expected_video)
    expected_audio = 1.0 + 0.25 * (result.audio_target_gate - 1.0)
    torch.testing.assert_close(result.audio_applied_gate, expected_audio)


def test_telemetry_fields_are_present_finite_and_bounded():
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    config = _spatial_config()
    controller = StochasticStabilityController(config, _layout())
    controller.update_actual(packed, packed.clone(), 2)
    result = controller.apply(
        torch.ones_like(packed),
        _observation(1.0, 0.5),
        torch.tensor(1.0),
        2,
        5,
        collect_stats=True,
    )
    record = _stochastic_control_record(config, controller, result)
    required = {
        "stochastic_control_mode",
        "video_dynamic_target_multiplier",
        "video_static_target_multiplier",
        "audio_target_multiplier",
        "video_applied_gate_min",
        "video_applied_gate_p50",
        "video_applied_gate_p95",
        "video_restore_active_fraction",
        "video_temporal_motion_ratio_mean",
        "video_diffusion_change_ratio_p95",
        "video_stability_source_actual_step",
        "stochastic_gate_slew_fraction",
    }
    assert required <= record.keys()
    for value in record.values():
        if torch.is_tensor(value):
            assert value.numel() == 1 and torch.isfinite(value)


def test_debug_maps_emit_for_actual_evidence_and_not_forecasts(tmp_path):
    current = torch.ones((1, 1, 3, 1, 4))
    packed = _packed(current)
    config = _spatial_config(debug_stability_maps=True, telemetry=True)
    controller = StochasticStabilityController(config, _layout())
    controller.update_actual(packed, packed.clone(), 1)
    actual = controller.apply(
        torch.ones_like(packed), _observation(1.0, 1.0), torch.tensor(1.0), 1, 5
    )
    forecast = controller.apply(
        torch.ones_like(packed), _observation(1.0, 1.0), torch.tensor(1.0), 2, 5
    )
    writer = TelemetryWriter(tmp_path, "actual-only", 7)
    _write_stability_maps(writer, config, forecast, 2, torch.tensor(0.5), 7)
    assert not writer.stability_map_directory.exists()
    _write_stability_maps(writer, config, actual, 1, torch.tensor(0.75), 7)
    assert sorted(path.name for path in writer.stability_map_directory.glob("*.npz")) == [
        "step-0001.npz"
    ]


def test_invalid_layout_fails_cleanly_for_spatial_control():
    controller = StochasticStabilityController(_spatial_config(), StreamLayout(12))
    with pytest.raises(ValueError, match="video shape"):
        controller.update_actual(torch.ones((1, 14)), None, 0)
