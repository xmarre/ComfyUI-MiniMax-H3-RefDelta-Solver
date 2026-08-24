from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

import comfyui_refdelta_solver.sampler as sampler_module
from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde
from comfyui_refdelta_solver.spectrum_interop import (
    SPECTRUM_BRIDGE_KEY,
    SPECTRUM_INTEROP_CONTRACT,
)


def _required_comfy_module(name: str):
    if os.environ.get("COMFYUI_PATH"):
        return importlib.import_module(name)
    return pytest.importorskip(name)


model_sampling_module = _required_comfy_module("comfy.model_sampling")


class _Sampling(model_sampling_module.CONST, model_sampling_module.ModelSamplingAV):
    pass


class _Model:
    def __init__(self):
        sampling = _Sampling()
        sampling.set_parameters(shift=12.0, audio_shift=3.0)
        patcher = SimpleNamespace(
            get_model_object=lambda name: sampling if name == "model_sampling" else None
        )
        packed_model = SimpleNamespace(latent_shapes=[(1, 2), (1, 2)])
        self.inner_model = SimpleNamespace(
            model_patcher=patcher,
            inner_model=packed_model,
        )

    def __call__(self, state, sigma, **_extra_args):
        return state * 0.125 + sigma.reshape(-1, 1) * 0.03125


def _fixed_noise(state):
    return lambda _sigma, _sigma_next: torch.full_like(state, 0.125)


def test_forecast_consumes_stochastic_ratio_from_immediately_preceding_actual(monkeypatch):
    refreshed = []
    multiplier_risks = []
    original_refresh = sampler_module._forecast_observation_with_latest_stochastic_evidence
    original_multiplier = sampler_module.stochastic_multiplier

    def refresh_spy(observation, ratios, sensitivity, max_stage):
        result = original_refresh(observation, ratios, sensitivity, max_stage)
        refreshed.append(
            {
                "risk": float(result.risk.detach().cpu()),
                "pressure": float(result.stochastic_pressure.detach().cpu()),
            }
        )
        return result

    def multiplier_spy(risk, strength, minimum):
        multiplier_risks.append(float(risk.detach().cpu()))
        return original_multiplier(risk, strength, minimum)

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            # Steps 0 and 1 are actual. Step 2 is the forecast that must consume
            # the stochastic/movement ratio measured after actual step 1.
            return step_id != 2

        @staticmethod
        def publish_stochastic_increment(_step_id, _increment):
            return None

    monkeypatch.setattr(
        sampler_module,
        "_forecast_observation_with_latest_stochastic_evidence",
        refresh_spy,
    )
    monkeypatch.setattr(sampler_module, "stochastic_multiplier", multiplier_spy)

    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.82, 0.64, 0.46, 0.28, 0.0], dtype=torch.float32)
    sample_refdelta_er_sde(
        _Model(),
        initial,
        sigmas,
        extra_args={
            "seed": 7,
            "model_options": {
                "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
            },
        },
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_adaptation_strength=0.5,
            minimum_stochastic_multiplier=0.5,
            trajectory_correction=False,
        ),
    )

    assert len(refreshed) >= 2
    assert len(multiplier_risks) >= 3
    # Step 0 has no denoised movement yet. Actual step 1 does, and its native
    # stochastic ratio must be reflected in the cached forecast observation.
    assert refreshed[1]["pressure"] > 0.0
    # Nonterminal multiplier calls are step ordered, so index 2 is forecast step
    # 2. It must receive the post-step-1 refreshed risk, not step 1's stale
    # pre-measurement observation.
    assert multiplier_risks[2] == pytest.approx(refreshed[1]["risk"], abs=1e-7)
