from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.h3_scheduler import (
    SCHEDULER_MODES,
    base_to_shifted_sigma,
    flow_profile_from_dict,
    h3_uniform_flow_sigmas,
    load_flow_profile,
    shifted_sigma_to_base,
)
from comfyui_refdelta_solver.nodes import MiniMaxH3UniformFlowScheduler


class SyntheticModelSamplingAV:
    def __init__(self, shift=12.0, audio_shift=3.0, timesteps=1000):
        self.shift = shift
        self.audio_shift = audio_shift
        self.multiplier = 1000
        base = torch.arange(1, timesteps + 1, dtype=torch.float64) / timesteps
        self.sigmas = self.sigma(base * self.multiplier).to(torch.float32)
        self.sigma_min = self.sigmas[0]
        self.sigma_max = self.sigmas[-1]

    def sigma(self, timestep):
        base = torch.as_tensor(timestep) / self.multiplier
        return base_to_shifted_sigma(base, self.shift)


def _mode_kwargs(mode, steps=20):
    kwargs = {}
    if mode in {"uniform_refinement_tail", "trailing_refined"}:
        kwargs["tail_steps"] = min(3, steps - 1) if steps > 1 else 0
    if mode == "curvature_profile":
        kwargs["profile"] = load_flow_profile("h3_uniform_neutral")
    return kwargs


@pytest.mark.parametrize("mode", SCHEDULER_MODES[1:])
@pytest.mark.parametrize("steps", (1, 2, 7, 13, 20, 37, 257))
def test_all_continuous_modes_obey_output_invariants(mode, steps):
    sampling = SyntheticModelSamplingAV()
    kwargs = _mode_kwargs(mode, steps)
    first = h3_uniform_flow_sigmas(sampling, steps, 1.0, mode, **kwargs)
    second = h3_uniform_flow_sigmas(sampling, steps, 1.0, mode, **kwargs)
    assert first.shape == (steps + 1,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first[1:] < first[:-1])
    assert first[-1].item() == 0.0


@pytest.mark.parametrize("mode", ("uniform_refinement_tail", "trailing_refined"))
@pytest.mark.parametrize("steps", (1, 2, 3, 4, 5))
def test_tail_mode_default_resolves_for_short_schedules(mode, steps):
    sigmas = h3_uniform_flow_sigmas(SyntheticModelSamplingAV(), steps, 1.0, mode)
    assert sigmas.shape == (steps + 1,)
    assert torch.all(sigmas[1:] < sigmas[:-1])


@pytest.mark.parametrize("mode", ("uniform_refinement_tail", "trailing_refined"))
def test_tail_modes_reject_explicit_invalid_count(mode):
    with pytest.raises(ValueError, match="tail_steps"):
        h3_uniform_flow_sigmas(SyntheticModelSamplingAV(), 5, 1.0, mode, tail_steps=5)


@pytest.mark.parametrize("mode", ("uniform_refinement_tail", "trailing_refined"))
@pytest.mark.parametrize("steps", (1, 2, 3, 4, 5))
def test_node_untouched_tail_defaults_are_valid_for_short_schedules(mode, steps):
    required = MiniMaxH3UniformFlowScheduler.INPUT_TYPES()["required"]
    inputs = {
        name: spec[1]["default"]
        for name, spec in required.items()
        if name != "model"
    }
    assert inputs["auto_tail_steps"] is True
    assert inputs["tail_steps"] == 5
    inputs.update(mode=mode, steps=steps)
    sampling = SyntheticModelSamplingAV()
    model = SimpleNamespace(get_model_object=lambda name: sampling)

    sigmas = MiniMaxH3UniformFlowScheduler().get_sigmas(model=model, **inputs)[0]

    assert sigmas.shape == (steps + 1,)
    assert torch.all(sigmas[1:] < sigmas[:-1])


@pytest.mark.parametrize("mode", SCHEDULER_MODES[1:])
def test_denoise_is_exact_full_schedule_tail(mode):
    sampling = SyntheticModelSamplingAV()
    kwargs = _mode_kwargs(mode)
    partial = h3_uniform_flow_sigmas(sampling, 10, 0.5, mode, **kwargs)
    full = h3_uniform_flow_sigmas(sampling, 20, 1.0, mode, **kwargs)
    assert torch.equal(partial, full[-11:])


@pytest.mark.parametrize("mode", SCHEDULER_MODES)
def test_zero_denoise_returns_empty_cpu_float32(mode):
    out = h3_uniform_flow_sigmas(SimpleNamespace(), 20, 0.0, mode)
    assert out.shape == (0,)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_reduction_identities_are_exact():
    sampling = SyntheticModelSamplingAV()
    uniform = h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "phase_offset_uniform", phase=0.0),
        uniform,
    )
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "power_uniform", power=1.0),
        uniform,
    )
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_refinement_tail", tail_steps=0),
        uniform,
    )
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "trailing_refined", tail_steps=0),
        uniform,
    )
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "av_arc_length", arc_strength=0.0),
        uniform,
    )
    assert torch.equal(
        h3_uniform_flow_sigmas(
            sampling,
            20,
            1.0,
            "piecewise_structure_refinement",
            structure_fraction=0.37,
            mid_power=1.0,
            detail_power=1.0,
        ),
        uniform,
    )
    neutral = load_flow_profile("h3_uniform_neutral")
    assert torch.equal(
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "curvature_profile", profile=neutral),
        uniform,
    )


def test_phase_has_exact_normalized_definition():
    sampling = SyntheticModelSamplingAV(shift=2.5, audio_shift=1.7)
    sigmas = h3_uniform_flow_sigmas(sampling, 4, 1.0, "phase_offset_uniform", phase=0.5)
    base = shifted_sigma_to_base(sigmas.to(torch.float64), sampling.shift)
    torch.testing.assert_close(
        base,
        torch.tensor([0.875, 0.625, 0.375, 0.125, 0.0], dtype=torch.float64),
        rtol=0,
        atol=2e-7,
    )


def test_refinement_tail_is_dense_near_zero_and_has_no_duplicate_join():
    sampling = SyntheticModelSamplingAV()
    sigmas = h3_uniform_flow_sigmas(
        sampling,
        20,
        1.0,
        "uniform_refinement_tail",
        tail_steps=5,
        tail_power=2.0,
    )
    base = shifted_sigma_to_base(sigmas.to(torch.float64), sampling.shift)
    assert torch.all(base[1:] < base[:-1])
    assert base[15].item() == pytest.approx(0.15, abs=2e-7)
    assert (base[-2] - base[-1]).item() < (base[1] - base[2]).item()
    body_join_interval = (base[14] - base[15]).item()
    first_tail_interval = (base[15] - base[16]).item()
    assert first_tail_interval <= body_join_interval * 1.1


def test_trailing_refined_uses_diffusers_last_training_index_normalization():
    sampling = SyntheticModelSamplingAV(timesteps=1000)
    sigmas = h3_uniform_flow_sigmas(
        sampling, 20, 1.0, "trailing_refined", tail_steps=5, tail_start=0.15
    )
    base = shifted_sigma_to_base(sigmas.to(torch.float64), sampling.shift)
    actual_table_top = shifted_sigma_to_base(sampling.sigmas[-1].to(torch.float64), sampling.shift)
    assert actual_table_top.item() == pytest.approx(1.0, abs=4e-7)
    assert base[0].item() == pytest.approx(0.999, abs=4e-7)


def test_piecewise_schedule_is_continuous_at_structure_boundary():
    sampling = SyntheticModelSamplingAV()
    sigmas = h3_uniform_flow_sigmas(
        sampling,
        20,
        1.0,
        "piecewise_structure_refinement",
        structure_fraction=0.4,
        mid_power=0.9,
        detail_power=1.1,
    )
    base = shifted_sigma_to_base(sigmas.to(torch.float64), sampling.shift)
    assert base[8].item() == pytest.approx(0.6, abs=2e-7)
    assert torch.all(base[1:] < base[:-1])


def test_arc_schedule_reads_both_nondefault_shifts():
    first = SyntheticModelSamplingAV(shift=7.0, audio_shift=2.0)
    changed_audio = SyntheticModelSamplingAV(shift=7.0, audio_shift=5.0)
    first_sigmas = h3_uniform_flow_sigmas(
        first, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=1.0
    )
    changed_sigmas = h3_uniform_flow_sigmas(
        changed_audio, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=1.0
    )
    assert not torch.equal(first_sigmas, changed_sigmas)

    video_only_first = h3_uniform_flow_sigmas(
        first, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=0.0
    )
    video_only_changed = h3_uniform_flow_sigmas(
        changed_audio, 20, 1.0, "av_arc_length", arc_strength=1.0, audio_weight=0.0
    )
    assert torch.equal(video_only_first, video_only_changed)


def test_nondefault_video_shift_changes_uniform_video_sigmas():
    first = SyntheticModelSamplingAV(shift=12.0, audio_shift=3.0)
    second = SyntheticModelSamplingAV(shift=5.0, audio_shift=3.0)
    assert not torch.equal(
        h3_uniform_flow_sigmas(first, 20, 1.0, "uniform_linspace"),
        h3_uniform_flow_sigmas(second, 20, 1.0, "uniform_linspace"),
    )


def test_scheduler_does_not_mutate_model_sigma_table():
    sampling = SyntheticModelSamplingAV()
    before = sampling.sigmas.clone()
    h3_uniform_flow_sigmas(sampling, 20, 1.0, "av_arc_length", arc_strength=1.0)
    assert torch.equal(sampling.sigmas, before)


def test_non_neutral_offline_profile_changes_schedule():
    neutral = load_flow_profile("h3_uniform_neutral")
    curved = replace(neutral, density=(1.0, 2.0, 1.0), progress=(0.0, 0.5, 1.0))
    sampling = SyntheticModelSamplingAV()
    uniform = h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")
    profiled = h3_uniform_flow_sigmas(sampling, 20, 1.0, "curvature_profile", profile=curved)
    assert not torch.equal(profiled, uniform)


def test_asymmetric_beta_is_continuous_base_time_not_stock_table_rounding():
    sampling = SyntheticModelSamplingAV()
    sigmas = h3_uniform_flow_sigmas(
        sampling, 20, 1.0, "asymmetric_beta", beta_alpha=0.45, beta_beta=0.75
    )
    assert sigmas.shape == (21,)
    assert torch.all(sigmas[1:] < sigmas[:-1])


def test_v1_beta_profile_cannot_be_silently_reinterpreted_as_shared_flow():
    with pytest.raises(ValueError, match="shared-flow profile version"):
        flow_profile_from_dict(
            {
                "version": 1,
                "id": "legacy",
                "domain": "video-sigma-progress-over-beta-prior",
                "points": [
                    {"progress": 0.0, "difficulty": 1.0},
                    {"progress": 1.0, "difficulty": 1.0},
                ],
            }
        )


def test_shared_flow_profile_rejects_comparison_or_unclassified_evidence():
    data = {
        "version": 2,
        "id": "bad-evidence",
        "domain": "shared-base-time-progress-density",
        "points": [
            {"progress": 0.0, "density": 1.0},
            {"progress": 1.0, "density": 1.0},
        ],
        "metadata": {
            "comparison_metrics_used_for_density": True,
            "evidence_source": "comparison_replay",
        },
    }
    with pytest.raises(ValueError, match="exclude comparison"):
        flow_profile_from_dict(data)


@pytest.mark.parametrize(
    "sampling,mode,kwargs,match",
    (
        (SimpleNamespace(), "uniform_linspace", {}, "ModelSamplingAV"),
        (SyntheticModelSamplingAV(audio_shift=0.0), "uniform_linspace", {}, "audio shift"),
        (SyntheticModelSamplingAV(), "phase_offset_uniform", {"phase": 1.0}, "phase"),
        (SyntheticModelSamplingAV(), "power_uniform", {"power": 0.0}, "power"),
        (SyntheticModelSamplingAV(), "uniform_refinement_tail", {"tail_steps": 20}, "tail_steps"),
        (SyntheticModelSamplingAV(), "trailing_refined", {"tail_steps": 5, "tail_start": 1.0}, "tail_start"),
        (SyntheticModelSamplingAV(), "asymmetric_beta", {"beta_alpha": 0.0}, "beta_alpha"),
        (SyntheticModelSamplingAV(), "av_arc_length", {"audio_weight": -1.0}, "audio_weight"),
        (
            SyntheticModelSamplingAV(),
            "piecewise_structure_refinement",
            {"structure_fraction": 0.0},
            "structure_fraction",
        ),
        (SyntheticModelSamplingAV(), "curvature_profile", {}, "requires an offline"),
    ),
)
def test_invalid_settings_fail_explicitly(sampling, mode, kwargs, match):
    with pytest.raises(ValueError, match=match):
        h3_uniform_flow_sigmas(sampling, 20, 1.0, mode, **kwargs)


def test_invalid_video_shift_fails_explicitly():
    sampling = SyntheticModelSamplingAV()
    sampling.shift = -1.0
    with pytest.raises(ValueError, match="video shift"):
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")


def test_malformed_sigma_tables_fail_explicitly():
    sampling = SyntheticModelSamplingAV()
    sampling.sigmas[10] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")

    sampling = SyntheticModelSamplingAV()
    sampling.sigmas[10] = sampling.sigmas[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        h3_uniform_flow_sigmas(sampling, 20, 1.0, "uniform_linspace")


@pytest.mark.parametrize("denoise", (-0.01, 1.01, float("nan")))
def test_invalid_denoise_fails(denoise):
    with pytest.raises(ValueError, match="denoise"):
        h3_uniform_flow_sigmas(SyntheticModelSamplingAV(), 20, denoise, "uniform_linspace")
