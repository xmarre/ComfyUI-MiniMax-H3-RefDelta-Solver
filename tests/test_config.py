from __future__ import annotations

import pytest

from comfyui_refdelta_solver.config import (
    PRODUCTION_STABILITY_DEFAULTS,
    PRODUCTION_STABILITY_PRESET_ID,
    RefDeltaSamplerConfig,
    production_stability_config,
)


def test_native_equivalence_mode_requires_every_adaptive_path_off():
    config = RefDeltaSamplerConfig(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )
    assert config.is_native_equivalence_mode
    assert not RefDeltaSamplerConfig().is_native_equivalence_mode

    spatial_values = RefDeltaSamplerConfig(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
        stochastic_control_mode="spatiotemporal_stability",
        static_video_stochastic_adaptation_strength=0.5,
    )
    assert spatial_values.is_native_equivalence_mode


def test_production_stability_defaults_are_named_tuned_and_separate_from_legacy_defaults():
    assert PRODUCTION_STABILITY_PRESET_ID == "r1024_int8_convrot_stability_v1"
    config = production_stability_config()
    config.validate()

    assert RefDeltaSamplerConfig().stochastic_control_mode == "legacy_global"
    assert RefDeltaSamplerConfig().trajectory_correction is False
    assert RefDeltaSamplerConfig().minimum_stochastic_multiplier == pytest.approx(0.25)

    assert config.stochastic_control_mode == "spatiotemporal_stability"
    assert config.stochastic_adaptation_strength == pytest.approx(0.50)
    assert config.minimum_stochastic_multiplier == pytest.approx(0.50)
    assert config.static_video_stochastic_adaptation_strength == pytest.approx(0.25)
    assert config.trajectory_correction is True
    assert config.video_stability_motion_low == pytest.approx(0.15)
    assert config.video_stability_motion_high == pytest.approx(0.60)
    assert config.video_stability_diffusion_low == pytest.approx(0.05)
    assert config.video_stability_diffusion_high == pytest.approx(0.50)
    assert config.video_stability_diffusion_weight == pytest.approx(0.20)
    assert config.video_stability_gamma == pytest.approx(1.00)
    assert config.calibration_capture is False
    assert PRODUCTION_STABILITY_DEFAULTS["debug_stability_maps"] is False


def test_production_stability_defaults_accept_explicit_tuning_overrides():
    config = production_stability_config(
        stochastic_adaptation_strength=0.55,
        static_video_stochastic_adaptation_strength=0.30,
        video_stability_gamma=1.25,
    )
    assert config.stochastic_adaptation_strength == pytest.approx(0.55)
    assert config.static_video_stochastic_adaptation_strength == pytest.approx(0.30)
    assert config.video_stability_gamma == pytest.approx(1.25)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"risk_sensitivity": 4.1},
        {"stochastic_adaptation_strength": -0.1},
        {"minimum_stochastic_multiplier": 1.1},
        {"correction_bound": 2.1},
        {"endpoint_fidelity_fraction": 0.51},
        {"stochastic_control_mode": "invalid"},
        {"video_stochastic_strength_scale": 2.1},
        {"audio_stochastic_strength_scale": -0.1},
        {"static_video_stochastic_adaptation_strength": 1.1},
        {"video_stability_motion_low": 0.1, "video_stability_motion_high": 0.1},
        {"video_stability_diffusion_low": 0.2, "video_stability_diffusion_high": 0.1},
        {"video_stability_normalization_floor": 0.0},
        {"video_stability_spatial_radius": 9},
        {"video_stability_temporal_radius": True},
        {"video_stability_ema": 1.0},
        {"video_stability_start_fraction": 0.5, "video_stability_full_fraction": 0.4},
        {"stochastic_gate_slew_limit": 1.1},
    ),
)
def test_invalid_config_fails(kwargs):
    with pytest.raises(ValueError):
        RefDeltaSamplerConfig(**kwargs).validate()
