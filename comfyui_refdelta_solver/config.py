from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def is_native_equivalence_mode(self) -> bool:
        return (
            not self.adaptive_order
            and self.stochastic_adaptation_strength == 0.0
            and not self.trajectory_correction
            and not self.telemetry
        )

