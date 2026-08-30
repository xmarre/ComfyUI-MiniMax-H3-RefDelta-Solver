from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler_backends import (
    sample_refdelta_sa_solver,
    sample_refdelta_sa_solver_pece,
    sample_refdelta_seeds_2,
    sample_refdelta_seeds_3,
)


@pytest.mark.parametrize(
    ("function", "native_name", "expected"),
    (
        (sample_refdelta_seeds_2, "sample_seeds_2", {"r": 0.5, "solver_type": "phi_1"}),
        (sample_refdelta_seeds_3, "sample_seeds_3", {"r_1": 1.0 / 3.0, "r_2": 2.0 / 3.0}),
        (
            sample_refdelta_sa_solver,
            "sample_sa_solver",
            {"predictor_order": 3, "corrector_order": 4, "use_pece": False},
        ),
        (
            sample_refdelta_sa_solver_pece,
            "sample_sa_solver",
            {"predictor_order": 3, "corrector_order": 4, "use_pece": True},
        ),
    ),
)
def test_native_equivalence_delegates_each_backend_without_model_access(
    monkeypatch,
    function,
    native_name,
    expected,
):
    sentinel = torch.tensor([321.0])
    received = {}

    def native(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return sentinel

    sampling = ModuleType("comfy.k_diffusion.sampling")
    setattr(sampling, native_name, native)
    k_diffusion = ModuleType("comfy.k_diffusion")
    k_diffusion.sampling = sampling
    comfy = ModuleType("comfy")
    comfy.k_diffusion = k_diffusion
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", k_diffusion)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)

    class ModelAccessSpy:
        @property
        def inner_model(self):
            raise AssertionError("native-equivalence delegation must not inspect the model")

    config = RefDeltaSamplerConfig(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )
    x = torch.tensor([1.0])
    sigmas = torch.tensor([1.0, 0.0])
    result = function(
        ModelAccessSpy(),
        x,
        sigmas,
        extra_args={"seed": 19},
        disable=True,
        s_noise=0.75,
        config=config,
    )

    assert result is sentinel
    assert received["args"][1] is x
    assert received["args"][2] is sigmas
    assert received["kwargs"]["s_noise"] == 0.75
    for name, value in expected.items():
        assert received["kwargs"][name] == value


def test_non_er_calibration_capture_fails_before_model_access():
    class ModelAccessSpy:
        @property
        def inner_model(self):
            raise AssertionError("invalid backend capture must fail before model access")

    config = RefDeltaSamplerConfig(
        calibration_capture=True,
        calibration_id="capture",
    )
    with pytest.raises(ValueError, match="calibration_capture"):
        sample_refdelta_seeds_2(
            ModelAccessSpy(),
            torch.ones(1),
            torch.tensor([1.0, 0.0]),
            config=config,
        )

    with pytest.raises(ValueError, match="calibration_capture"):
        sample_refdelta_sa_solver_pece(
            ModelAccessSpy(),
            torch.ones(1),
            torch.tensor([1.0, 0.0]),
            config=config,
        )



def test_missing_native_backend_fails_with_explicit_requirement(monkeypatch):
    sampling = ModuleType("comfy.k_diffusion.sampling")
    k_diffusion = ModuleType("comfy.k_diffusion")
    k_diffusion.sampling = sampling
    comfy = ModuleType("comfy")
    comfy.k_diffusion = k_diffusion
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion", k_diffusion)
    monkeypatch.setitem(sys.modules, "comfy.k_diffusion.sampling", sampling)

    with pytest.raises(TypeError, match="sample_seeds_2"):
        sample_refdelta_seeds_2(
            object(),
            torch.ones(1),
            torch.tensor([1.0, 0.0]),
            config=RefDeltaSamplerConfig(
                adaptive_order=False,
                stochastic_adaptation_strength=0.0,
                trajectory_correction=False,
                telemetry=False,
            ),
        )
