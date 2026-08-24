from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde


k_sampling = pytest.importorskip("comfy.k_diffusion.sampling")
model_sampling_module = pytest.importorskip("comfy.model_sampling")


class _Sampling(model_sampling_module.CONST, model_sampling_module.ModelSamplingAV):
    pass


class _Model:
    def __init__(self):
        sampling = _Sampling()
        sampling.set_parameters(shift=12.0, audio_shift=3.0)
        patcher = SimpleNamespace(get_model_object=lambda name: sampling if name == "model_sampling" else None)
        packed_model = SimpleNamespace(latent_shapes=[(1, 2), (1, 2)])
        self.inner_model = SimpleNamespace(model_patcher=patcher, inner_model=packed_model)

    def __call__(self, state, sigma, **_extra_args):
        return state * 0.125 + sigma.reshape(-1, 1) * 0.03125


def _fixed_noise(state):
    return lambda _sigma, _sigma_next: torch.full_like(state, 0.125)


def test_instrumented_no_adaptation_matches_current_native_er_sde(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)

    native = k_sampling.sample_er_sde(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        max_stage=3,
    )
    instrumented = sample_refdelta_er_sde(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        max_stage=3,
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_adaptation_strength=0.0,
            trajectory_correction=False,
            telemetry=True,
            telemetry_prefix="native-fixture",
        ),
    )

    torch.testing.assert_close(instrumented, native, rtol=2e-6, atol=2e-7)
    assert len(list((tmp_path / "refdelta_telemetry").glob("native-fixture-*.jsonl"))) == 1
