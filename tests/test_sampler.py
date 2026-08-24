from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde


def test_native_equivalence_mode_delegates_without_reimplementation(monkeypatch):
    sentinel = torch.tensor([123.0])
    received = {}

    def native(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return sentinel

    sampling = ModuleType("comfy.k_diffusion.sampling")
    sampling.sample_er_sde = native
    k_diffusion = ModuleType("comfy.k_diffusion")
    k_diffusion.sampling = sampling
    comfy = ModuleType("comfy")
    comfy.k_diffusion = k_diffusion
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", k_diffusion)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)

    model = object()
    x = torch.tensor([1.0])
    sigmas = torch.tensor([1.0, 0.0])
    config = RefDeltaSamplerConfig(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )
    result = sample_refdelta_er_sde(
        model,
        x,
        sigmas,
        extra_args={"seed": 17},
        disable=True,
        s_noise=0.75,
        max_stage=2,
        config=config,
    )

    assert result is sentinel
    assert received["args"] == (model, x, sigmas)
    assert received["kwargs"]["extra_args"] == {"seed": 17}
    assert received["kwargs"]["s_noise"] == 0.75
    assert received["kwargs"]["max_stage"] == 2


def test_invalid_stage_fails_before_model_access():
    with pytest.raises(ValueError, match="max_stage"):
        sample_refdelta_er_sde(object(), torch.tensor([1.0]), torch.tensor([1.0, 0.0]), max_stage=4)
