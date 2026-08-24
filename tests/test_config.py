from __future__ import annotations

import pytest

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig


def test_native_equivalence_mode_requires_every_adaptive_path_off():
    config = RefDeltaSamplerConfig(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )
    assert config.is_native_equivalence_mode
    assert not RefDeltaSamplerConfig().is_native_equivalence_mode


@pytest.mark.parametrize(
    "kwargs",
    (
        {"risk_sensitivity": 4.1},
        {"stochastic_adaptation_strength": -0.1},
        {"minimum_stochastic_multiplier": 1.1},
        {"correction_bound": 2.1},
        {"endpoint_fidelity_fraction": 0.51},
    ),
)
def test_invalid_config_fails(kwargs):
    with pytest.raises(ValueError):
        RefDeltaSamplerConfig(**kwargs).validate()

