from __future__ import annotations

from dataclasses import dataclass


PRODUCTION_STABILITY_PRESET_ID = "r1024_int8_convrot_stability_v1"


@dataclass(frozen=True, slots=True)
class RefDeltaSamplerConfig:
    adaptive_order: bool = True
    risk_sensitivity: float = 1.0
    stochastic_adaptation_strength: float = 0.50
    minimum_stochastic_multiplier: float = 0.25
    trajectory_correction: bool = False
    video_correction_strength: float = 0.15
    audio_correction_strength: float = 0.05
    correction_bound: float = 0.50
    endpoint_fidelity_fraction: float = 0.15
    telemetry: bool = False
    telemetry_prefix: str = "refdelta_trajectory"
    calibration_capture: bool = False
    calibration_id: str = "refdelta_calibration"
    stochastic_control_mode: str = "legacy_global"
    video_stochastic_strength_scale: float = 1.0
    audio_stochastic_strength_scale: float = 1.0
    static_video_stochastic_adaptation_strength: float = 0.50
    video_stability_restore_strength: float = 1.0
    video_stability_motion_low: float = 0.02
    video_stability_motion_high: float = 0.10
    video_stability_diffusion_low: float = 0.02
    video_stability_diffusion_high: float = 0.15
    video_stability_diffusion_weight: float = 0.50
    video_stability_normalization_floor: float = 0.10
    video_stability_gamma: float = 1.0
    video_stability_spatial_radius: int = 2
    video_stability_temporal_radius: int = 2
    video_stability_ema: float = 0.70
    video_stability_start_fraction: float = 0.10
    video_stability_full_fraction: float = 0.30
    stochastic_gate_slew_limit: float = 0.0
    debug_stability_maps: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.risk_sensitivity <= 4.0:
            raise ValueError("risk_sensitivity must be in [0, 4]")
        if not 0.0 <= self.stochastic_adaptation_strength <= 1.0:
            raise ValueError("stochastic_adaptation_strength must be in [0, 1]")
        if not 0.0 <= self.minimum_stochastic_multiplier <= 1.0:
            raise ValueError("minimum_stochastic_multiplier must be in [0, 1]")
        if not 0.0 <= self.video_correction_strength <= 1.0:
            raise ValueError("video_correction_strength must be in [0, 1]")
        if not 0.0 <= self.audio_correction_strength <= 1.0:
            raise ValueError("audio_correction_strength must be in [0, 1]")
        if not 0.0 <= self.correction_bound <= 2.0:
            raise ValueError("correction_bound must be in [0, 2]")
        if not 0.0 <= self.endpoint_fidelity_fraction <= 0.50:
            raise ValueError("endpoint_fidelity_fraction must be in [0, 0.5]")
        if self.stochastic_control_mode not in {
            "legacy_global",
            "streamwise",
            "spatiotemporal_stability",
        }:
            raise ValueError("invalid stochastic_control_mode")
        if not 0.0 <= self.video_stochastic_strength_scale <= 2.0:
            raise ValueError("video_stochastic_strength_scale must be in [0, 2]")
        if not 0.0 <= self.audio_stochastic_strength_scale <= 2.0:
            raise ValueError("audio_stochastic_strength_scale must be in [0, 2]")
        if not 0.0 <= self.static_video_stochastic_adaptation_strength <= 1.0:
            raise ValueError("static_video_stochastic_adaptation_strength must be in [0, 1]")
        if not 0.0 <= self.video_stability_restore_strength <= 1.0:
            raise ValueError("video_stability_restore_strength must be in [0, 1]")
        if not 0.0 <= self.video_stability_motion_low < self.video_stability_motion_high <= 2.0:
            raise ValueError("video stability motion thresholds require 0 <= low < high <= 2")
        if not 0.0 <= self.video_stability_diffusion_low < self.video_stability_diffusion_high <= 2.0:
            raise ValueError("video stability diffusion thresholds require 0 <= low < high <= 2")
        if not 0.0 <= self.video_stability_diffusion_weight <= 1.0:
            raise ValueError("video_stability_diffusion_weight must be in [0, 1]")
        if not 0.001 <= self.video_stability_normalization_floor <= 1.0:
            raise ValueError("video_stability_normalization_floor must be in [0.001, 1]")
        if not 0.25 <= self.video_stability_gamma <= 4.0:
            raise ValueError("video_stability_gamma must be in [0.25, 4]")
        if (
            not isinstance(self.video_stability_spatial_radius, int)
            or isinstance(self.video_stability_spatial_radius, bool)
            or not 0 <= self.video_stability_spatial_radius <= 8
        ):
            raise ValueError("video_stability_spatial_radius must be an integer in [0, 8]")
        if (
            not isinstance(self.video_stability_temporal_radius, int)
            or isinstance(self.video_stability_temporal_radius, bool)
            or not 0 <= self.video_stability_temporal_radius <= 8
        ):
            raise ValueError("video_stability_temporal_radius must be an integer in [0, 8]")
        if not 0.0 <= self.video_stability_ema <= 0.99:
            raise ValueError("video_stability_ema must be in [0, 0.99]")
        if not 0.0 <= self.video_stability_start_fraction <= self.video_stability_full_fraction <= 1.0:
            raise ValueError("video stability progress requires 0 <= start <= full <= 1")
        if not 0.0 <= self.stochastic_gate_slew_limit <= 1.0:
            raise ValueError("stochastic_gate_slew_limit must be in [0, 1]")
        if self.calibration_capture and not str(self.calibration_id).strip():
            raise ValueError("calibration_id must not be empty when calibration capture is enabled")

    @property
    def is_native_equivalence_mode(self) -> bool:
        return (
            not self.adaptive_order
            and self.stochastic_adaptation_strength == 0.0
            and not self.trajectory_correction
            and not self.telemetry
            and not self.calibration_capture
        )


# Production tuning is intentionally separate from the dataclass defaults. The
# defaults above preserve saved-workflow behavior; this named preset is the
# empirically tuned rank-1024 INT8 ConvRot path exposed by the recommended node.
PRODUCTION_STABILITY_DEFAULTS: dict[str, object] = {
    "adaptive_order": True,
    "risk_sensitivity": 1.0,
    "stochastic_adaptation_strength": 0.60,
    "minimum_stochastic_multiplier": 0.50,
    "trajectory_correction": True,
    "video_correction_strength": 0.15,
    "audio_correction_strength": 0.05,
    "correction_bound": 0.50,
    "endpoint_fidelity_fraction": 0.15,
    "telemetry": False,
    "telemetry_prefix": "refdelta_trajectory",
    "calibration_capture": False,
    "calibration_id": "refdelta_calibration",
    "stochastic_control_mode": "spatiotemporal_stability",
    "video_stochastic_strength_scale": 1.0,
    "audio_stochastic_strength_scale": 1.0,
    "static_video_stochastic_adaptation_strength": 0.25,
    "video_stability_restore_strength": 1.0,
    "video_stability_motion_low": 0.15,
    "video_stability_motion_high": 0.60,
    "video_stability_diffusion_low": 0.05,
    "video_stability_diffusion_high": 0.50,
    "video_stability_diffusion_weight": 0.25,
    "video_stability_normalization_floor": 0.10,
    "video_stability_gamma": 0.75,
    "video_stability_spatial_radius": 2,
    "video_stability_temporal_radius": 2,
    "video_stability_ema": 0.70,
    "video_stability_start_fraction": 0.10,
    "video_stability_full_fraction": 0.30,
    "stochastic_gate_slew_limit": 0.0,
    "debug_stability_maps": False,
}


def production_stability_config(**overrides) -> RefDeltaSamplerConfig:
    """Build the named production stability preset with explicit overrides."""
    values = dict(PRODUCTION_STABILITY_DEFAULTS)
    values.update(overrides)
    config = RefDeltaSamplerConfig(**values)
    config.validate()
    return config


__all__ = [
    "PRODUCTION_STABILITY_DEFAULTS",
    "PRODUCTION_STABILITY_PRESET_ID",
    "RefDeltaSamplerConfig",
    "production_stability_config",
]
