from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.h3_scheduler import base_to_shifted_sigma, shifted_sigma_to_base
from comfyui_refdelta_solver.sa_scheduler import (
    BASE_INTERVAL_MAX_RATIO,
    BASE_INTERVAL_MIN_RATIO,
    MAX_NODE_DISPLACEMENT_MEAN_INTERVALS,
    SA_SCHEDULER_MODES,
    h3_sa_solver_sigmas,
)
from tools.inspect_sa_schedule import (
    COMPARISON_MODES,
    adams_diagnostics,
    build_report,
    coordinate_diagnostics,
    prepare_comfyui,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH"),
    reason="requires pinned ComfyUI source",
)


def _native_sampling(shift=12.0, audio_shift=3.0, timesteps=1000, multiplier=1000):
    import comfy.model_sampling

    class NativeSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    sampling = NativeSampling()
    sampling.set_parameters(
        shift=shift,
        audio_shift=audio_shift,
        timesteps=timesteps,
        multiplier=multiplier,
    )
    return sampling


def _base(sigmas, shift=12.0):
    return shifted_sigma_to_base(sigmas.detach().cpu().double(), shift)


def test_inspector_cpu_bootstrap_does_not_require_comfy_kitchen(monkeypatch):
    comfyui_path = Path(os.environ["COMFYUI_PATH"])
    monkeypatch.setitem(sys.modules, "comfy_kitchen", None)
    prepare_comfyui(comfyui_path)


def test_adams_diagnostics_serialize_nonfinite_coefficients(monkeypatch):
    from comfy.k_diffusion import sa_solver

    sampling = _native_sampling()
    sigmas = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control")

    def nonfinite_coefficients(*args, **kwargs):
        return torch.tensor([float("inf"), float("nan")], dtype=torch.float64)

    monkeypatch.setattr(
        sa_solver,
        "compute_stochastic_adams_b_coeffs",
        nonfinite_coefficients,
    )
    diagnostics = adams_diagnostics(sampling, sigmas)

    assert diagnostics["all_coefficients_finite"] is False
    assert diagnostics["maximum_absolute_coefficient"] is None
    assert diagnostics["maximum_l1_norm"] is None
    assert diagnostics["maximum_l2_norm"] is None
    assert diagnostics["phase_maximum_absolute_coefficient"] == {
        "predictor": None,
        "corrector": None,
    }
    assert diagnostics["maximum_interval_growth"] == {
        "predictor": None,
        "corrector": None,
    }
    assert diagnostics["unexpectedly_extreme_coefficient"] is False
    assert all(record["max_abs"] is None for record in diagnostics["records"])
    assert all(record["dynamic_range"] is None for record in diagnostics["records"])
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("steps", (1, 2, 8, 10, 13, 15, 19, 20, 37, 999))
def test_simple_control_is_exact_current_comfyui_simple(steps):
    import comfy.samplers

    sampling = _native_sampling()
    expected = comfy.samplers.calculate_sigmas(sampling, "simple", steps).cpu()
    actual = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_control")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "steps,denoise",
    ((10, 0.75), (10, 0.5), (7, 0.35), (1, 0.01)),
)
def test_simple_control_matches_basic_scheduler_denoise_tail(steps, denoise):
    import comfy.samplers

    sampling = _native_sampling()
    total_steps = int(steps / denoise)
    expected = comfy.samplers.calculate_sigmas(sampling, "simple", total_steps).cpu()
    expected = expected[-(steps + 1) :]
    actual = h3_sa_solver_sigmas(sampling, steps, denoise, "simple_control")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", SA_SCHEDULER_MODES)
def test_zero_denoise_matches_basic_scheduler_empty_result(mode):
    actual = h3_sa_solver_sigmas(SimpleNamespace(), 10, 0.0, mode)
    assert actual.shape == (0,)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"


@pytest.mark.parametrize("denoise", (-0.01, 1.01, float("nan"), float("inf")))
def test_invalid_denoise_fails(denoise):
    with pytest.raises(ValueError, match="denoise"):
        h3_sa_solver_sigmas(_native_sampling(), 10, denoise, "simple_control")


def test_dedicated_scheduler_requires_native_h3_const_sampling():
    with pytest.raises(TypeError, match="ModelSamplingAV with CONST"):
        h3_sa_solver_sigmas(SimpleNamespace(), 10, 1.0, "simple_control")


@pytest.mark.parametrize("steps", (1, 2, 8, 10, 13, 15, 19, 20, 37, 257))
def test_bounded_mode_obeys_outer_schedule_invariants(steps):
    sampling = _native_sampling()
    first = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_adams_bounded")
    second = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_adams_bounded")
    assert first.shape == (steps + 1,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first[1:] < first[:-1])
    assert first[-1].item() == 0.0


@pytest.mark.parametrize("steps", (8, 10, 13, 15, 19, 20))
def test_bounded_mode_enforces_shared_base_contract(steps):
    sampling = _native_sampling()
    simple = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_control")
    candidate = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_adams_bounded")
    control_base = _base(simple, sampling.shift)
    candidate_base = _base(candidate, sampling.shift)
    control_intervals = control_base[:-1] - control_base[1:]
    candidate_intervals = candidate_base[:-1] - candidate_base[1:]
    ratios = candidate_intervals / control_intervals
    mean_interval = float(control_intervals.mean())
    max_displacement = float((candidate_base - control_base).abs().max())

    assert float(ratios.min()) >= BASE_INTERVAL_MIN_RATIO
    assert float(ratios.max()) <= BASE_INTERVAL_MAX_RATIO
    assert max_displacement / mean_interval <= MAX_NODE_DISPLACEMENT_MEAN_INTERVALS


@pytest.mark.parametrize("steps", (10, 19))
def test_bounded_mode_moves_at_most_one_interior_base_node(steps):
    sampling = _native_sampling()
    simple = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_control")
    candidate = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_adams_bounded")
    displacement = (_base(candidate) - _base(simple)).abs()
    moved = torch.nonzero(displacement > 1.0e-7).flatten()
    assert moved.numel() <= 1
    if moved.numel():
        assert 0 < int(moved[0]) < steps


def test_ten_step_bounded_schedule_matches_selected_local_transfer():
    sampling = _native_sampling()
    simple = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control")
    candidate = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_adams_bounded")
    simple_base = _base(simple)
    candidate_base = _base(candidate)

    torch.testing.assert_close(simple_base[4], torch.tensor(0.6, dtype=torch.float64), rtol=0, atol=2e-7)
    torch.testing.assert_close(
        candidate_base[4],
        torch.tensor(0.6124, dtype=torch.float64),
        rtol=0,
        atol=2e-6,
    )
    moved = torch.nonzero((candidate_base - simple_base).abs() > 1.0e-7).flatten()
    assert moved.tolist() == [4]


def test_nineteen_step_bounded_schedule_matches_selected_local_transfer():
    sampling = _native_sampling()
    simple = h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_control")
    candidate = h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_adams_bounded")
    simple_base = _base(simple)
    candidate_base = _base(candidate)

    torch.testing.assert_close(simple_base[14], torch.tensor(0.264, dtype=torch.float64), rtol=0, atol=2e-7)
    torch.testing.assert_close(
        candidate_base[14],
        torch.tensor(0.260776, dtype=torch.float64),
        rtol=0,
        atol=2e-6,
    )
    moved = torch.nonzero((candidate_base - simple_base).abs() > 1.0e-7).flatten()
    assert moved.tolist() == [14]


@pytest.mark.parametrize("steps", (10, 19))
def test_bounded_mode_reduces_worst_native_adams_l1(steps):
    sampling = _native_sampling()
    simple = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_control")
    candidate = h3_sa_solver_sigmas(sampling, steps, 1.0, "simple_adams_bounded")
    simple_diag = adams_diagnostics(sampling, simple)
    candidate_diag = adams_diagnostics(sampling, candidate)
    assert candidate_diag["all_coefficients_finite"] is True
    assert candidate_diag["maximum_l1_norm"] < simple_diag["maximum_l1_norm"]


def test_ten_step_bounded_l1_regression_value():
    sampling = _native_sampling()
    simple = adams_diagnostics(
        sampling,
        h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control"),
    )
    bounded = adams_diagnostics(
        sampling,
        h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_adams_bounded"),
    )
    assert simple["maximum_l1_norm"] == pytest.approx(0.426531, abs=2e-6)
    assert bounded["maximum_l1_norm"] == pytest.approx(0.402478, abs=2e-6)


def test_nineteen_step_bounded_l1_regression_value():
    sampling = _native_sampling()
    simple = adams_diagnostics(
        sampling,
        h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_control"),
    )
    bounded = adams_diagnostics(
        sampling,
        h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_adams_bounded"),
    )
    assert simple["maximum_l1_norm"] == pytest.approx(0.500922, abs=3e-6)
    assert bounded["maximum_l1_norm"] == pytest.approx(0.455545, abs=3e-6)
    assert bounded["maximum_absolute_coefficient"] == pytest.approx(
        simple["maximum_absolute_coefficient"],
        abs=3e-6,
    )
    assert bounded["maximum_l2_norm"] == pytest.approx(
        simple["maximum_l2_norm"],
        abs=3e-6,
    )


@pytest.mark.parametrize("denoise", (1.0, 0.75, 0.5, 0.01))
def test_bounded_denoise_is_exact_full_schedule_tail(denoise):
    sampling = _native_sampling()
    steps = 1 if denoise == 0.01 else 10
    full_steps = int(steps / denoise) if denoise < 1.0 else steps
    partial = h3_sa_solver_sigmas(
        sampling,
        steps,
        denoise,
        "simple_adams_bounded",
    )
    full = h3_sa_solver_sigmas(
        sampling,
        full_steps,
        1.0,
        "simple_adams_bounded",
    )
    assert torch.equal(partial, full[-(steps + 1) :])


def test_first_endpoint_is_owned_by_native_sa_protection_and_terminal_is_exact():
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr, sigma_to_half_log_snr

    sampling = _native_sampling()
    sigmas = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_adams_bounded")
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0
    effective = offset_first_sigma_for_snr(sigmas, sampling)
    expected = torch.tensor(sampling.percent_to_sigma(0.0001), dtype=sigmas.dtype)
    assert effective[0].item() == expected.item()
    assert math.isfinite(
        sigma_to_half_log_snr(effective[0].double(), model_sampling=sampling).item()
    )
    assert torch.isinf(
        sigma_to_half_log_snr(sigmas[-1].double(), model_sampling=sampling)
    )


def test_nondefault_shifts_and_multiplier_preserve_one_shared_av_clock():
    sampling = _native_sampling(shift=7.5, audio_shift=2.25, multiplier=777)
    sigmas = h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_adams_bounded")
    base = shifted_sigma_to_base(sigmas.double(), sampling.shift)
    audio = base_to_shifted_sigma(base, sampling.audio_shift)
    assert torch.all(base[1:] < base[:-1])
    assert torch.all(audio[1:] < audio[:-1])
    assert base[-1].item() == 0.0
    assert audio[-1].item() == 0.0


def test_candidate_does_not_mutate_model_sigma_table():
    sampling = _native_sampling()
    before = sampling.sigmas.clone()
    h3_sa_solver_sigmas(sampling, 19, 1.0, "simple_adams_bounded")
    assert torch.equal(sampling.sigmas, before)


def test_malformed_runtime_sigma_table_fails_closed():
    sampling = _native_sampling()
    sampling.sigmas[10] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control")


@pytest.mark.parametrize(
    "attribute,value",
    (("shift", 0.0), ("audio_shift", 0.0), ("audio_shift", None), ("multiplier", 0.0)),
)
def test_invalid_runtime_h3_metadata_fails(attribute, value):
    sampling = _native_sampling()
    setattr(sampling, attribute, value)
    with pytest.raises(ValueError, match=attribute.replace("_", " ")):
        h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control")


def test_unknown_mode_and_invalid_steps_fail_explicitly():
    sampling = _native_sampling()
    with pytest.raises(ValueError, match="unknown"):
        h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_lambda_uniform")
    with pytest.raises(ValueError, match="steps"):
        h3_sa_solver_sigmas(sampling, 0, 1.0, "simple_control")


@pytest.mark.parametrize("steps", (10, 19))
def test_failed_lambda_controls_remain_diagnostic_only_and_expose_distortion(steps):
    sampling = _native_sampling()
    report = build_report(
        sampling,
        [
            "simple",
            "simple_adams_bounded",
            "failed_simple_lambda_uniform",
            "failed_simple_lambda_blend",
        ],
        steps=steps,
        denoise=1.0,
        phase=0.50,
        lambda_blend=0.50,
        comfyui_revision="test-revision",
    )
    by_mode = {schedule["mode"]: schedule for schedule in report["schedules"]}
    bounded = by_mode["simple_adams_bounded"]["summary"]
    failed_uniform = by_mode["failed_simple_lambda_uniform"]["summary"]
    failed_blend = by_mode["failed_simple_lambda_blend"]["summary"]

    assert bounded["min_base_interval_ratio_vs_simple"] >= BASE_INTERVAL_MIN_RATIO
    assert bounded["max_base_interval_ratio_vs_simple"] <= BASE_INTERVAL_MAX_RATIO
    assert (
        bounded["max_node_displacement_in_simple_intervals"]
        <= MAX_NODE_DISPLACEMENT_MEAN_INTERVALS
    )
    assert failed_uniform["min_base_interval_ratio_vs_simple"] < 0.01
    assert failed_uniform["max_base_interval_ratio_vs_simple"] > 2.0
    assert failed_blend["min_base_interval_ratio_vs_simple"] < 0.10
    assert failed_blend["max_base_interval_ratio_vs_simple"] > 1.5


@pytest.mark.parametrize("steps", (8, 10, 13, 15, 19, 20))
def test_required_schedule_comparison_set_is_numerically_inspectable(steps):
    sampling = _native_sampling()
    report = build_report(
        sampling,
        list(COMPARISON_MODES),
        steps=steps,
        denoise=1.0,
        phase=0.50,
        lambda_blend=0.50,
        comfyui_revision="test-revision",
    )
    assert report["configuration"]["active_pece_logical_opportunities"] == 2 * steps - 1
    assert report["configuration"]["active_pece_callbacks"] == steps
    assert [schedule["mode"] for schedule in report["schedules"]] == list(COMPARISON_MODES)
    for schedule in report["schedules"]:
        assert len(schedule["rows"]) == steps + 1
        assert schedule["summary"]["all_returned_coordinates_finite"] is True
        assert schedule["summary"]["finite_nonterminal_lambda"] is True
        assert schedule["summary"]["terminal_lambda_is_positive_infinity"] is True
        assert schedule["adams"]["all_coefficients_finite"] is True
        assert schedule["adams"]["errors"] == []


def test_inspection_rows_expose_required_h3_sa_coordinates_and_base_contract():
    sampling = _native_sampling(shift=7.5, audio_shift=2.25, multiplier=777)
    simple = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_control")
    sigmas = h3_sa_solver_sigmas(sampling, 10, 1.0, "simple_adams_bounded")
    rows, summary = coordinate_diagnostics(
        sampling,
        sigmas,
        control_sigmas=simple,
    )
    assert set(rows[0]) == {
        "index",
        "returned_sigma",
        "effective_sa_sigma",
        "shared_base_time",
        "shifted_video_sigma",
        "implied_audio_sigma",
        "half_log_snr_lambda",
        "delta_lambda",
        "delta_base_time",
        "delta_sigma",
        "base_interval_ratio_vs_simple",
        "node_displacement_vs_simple",
    }
    assert rows[0]["returned_sigma"] == 1.0
    assert rows[0]["effective_sa_sigma"] < 1.0
    assert rows[-1]["returned_sigma"] == 0.0
    assert rows[-1]["half_log_snr_lambda"] is None
    assert summary["strict_sigma_monotonicity"] is True
    assert summary["strict_lambda_monotonicity"] is True
    assert summary["min_base_interval_ratio_vs_simple"] >= BASE_INTERVAL_MIN_RATIO
    assert summary["max_base_interval_ratio_vs_simple"] <= BASE_INTERVAL_MAX_RATIO
    assert (
        summary["max_node_displacement_in_simple_intervals"]
        <= MAX_NODE_DISPLACEMENT_MEAN_INTERVALS
    )
