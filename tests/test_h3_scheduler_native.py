from __future__ import annotations

import os

import pytest
import torch

from comfyui_refdelta_solver.h3_scheduler import h3_uniform_flow_sigmas


pytestmark = pytest.mark.skipif(not os.environ.get("COMFYUI_PATH"), reason="requires pinned ComfyUI source")


def _native_sampling(shift=12.0, audio_shift=3.0):
    import comfy.model_sampling

    class NativeSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    sampling = NativeSampling()
    sampling.set_parameters(shift=shift, audio_shift=audio_shift)
    return sampling


@pytest.mark.parametrize("steps", (1, 2, 7, 13, 19, 20, 37, 999))
def test_legacy_ddim_uniform_is_exact_basic_scheduler_parity(steps):
    import comfy.samplers

    sampling = _native_sampling()
    expected = comfy.samplers.calculate_sigmas(sampling, "ddim_uniform", steps).cpu()
    expected = expected[-(steps + 1) :]
    actual = h3_uniform_flow_sigmas(sampling, steps, 1.0, "legacy_ddim_uniform")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("steps,denoise", ((10, 0.5), (7, 0.35), (13, 0.65)))
def test_legacy_ddim_uniform_matches_basic_scheduler_denoise_tail(steps, denoise):
    import comfy.samplers

    sampling = _native_sampling()
    total_steps = int(steps / denoise)
    expected = comfy.samplers.calculate_sigmas(sampling, "ddim_uniform", total_steps).cpu()
    expected = expected[-(steps + 1) :]
    actual = h3_uniform_flow_sigmas(sampling, steps, denoise, "legacy_ddim_uniform")
    assert torch.equal(actual, expected)


def test_legacy_rejects_more_steps_than_unique_table_points():
    sampling = _native_sampling()
    with pytest.raises(ValueError, match="unique table-index capacity"):
        h3_uniform_flow_sigmas(sampling, 1000, 1.0, "legacy_ddim_uniform")


def test_asymmetric_symmetric_beta_is_explicitly_continuous_not_stock_rounded_beta():
    import comfy.samplers

    sampling = _native_sampling()
    continuous = h3_uniform_flow_sigmas(
        sampling, 20, 1.0, "asymmetric_beta", beta_alpha=0.6, beta_beta=0.6
    )
    stock = comfy.samplers.calculate_sigmas(sampling, "beta", 20).cpu()
    assert continuous.shape == stock.shape
    assert not torch.equal(continuous, stock)
    assert torch.max(torch.abs(continuous - stock)).item() < 0.02


def test_native_model_sampling_av_nondefault_shift_pair_is_used():
    sampling = _native_sampling(shift=7.5, audio_shift=2.25)
    changed = _native_sampling(shift=7.5, audio_shift=4.5)
    first = h3_uniform_flow_sigmas(
        sampling, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=1.0
    )
    second = h3_uniform_flow_sigmas(
        changed, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=1.0
    )
    assert not torch.equal(first, second)


def test_er_sde_first_sigma_safety_offset_boundary_is_preserved():
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr

    sampling = _native_sampling()
    uniform = h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")
    assert uniform[0].item() == 1.0
    adjusted = offset_first_sigma_for_snr(uniform, sampling)
    expected = torch.tensor(sampling.percent_to_sigma(0.0001), dtype=uniform.dtype).item()
    assert adjusted[0].item() == expected
    assert torch.equal(adjusted[1:], uniform[1:])

    legacy = h3_uniform_flow_sigmas(sampling, 20, 1.0, "legacy_ddim_uniform")
    assert legacy[0].item() < 1.0
    assert torch.equal(offset_first_sigma_for_snr(legacy, sampling), legacy)
