from __future__ import annotations

import math
from typing import Any

import torch

from .h3_scheduler import (
    _map_base_with_model,
    _validate_base,
    _validate_sigmas,
    _validated_sampling_metadata,
    shifted_sigma_to_base,
)


SA_SCHEDULER_MODES = (
    "simple_control",
    "simple_adams_bounded",
)

BASE_INTERVAL_MIN_RATIO = 0.875
BASE_INTERVAL_MAX_RATIO = 1.125
MAX_NODE_DISPLACEMENT_MEAN_INTERVALS = 0.125

# Search just inside the public 12.5% envelope so float32 sigma round-tripping
# cannot turn an accepted trial into a schedule outside the hard contract.
_SEARCH_INTERVAL_FRACTION = 0.124
_SEARCH_SCALES = (-1.0, -0.5, 0.5, 1.0)


def _require_h3_sa_sampling(model_sampling: Any) -> None:
    import comfy.model_sampling

    if not isinstance(model_sampling, comfy.model_sampling.ModelSamplingAV) or not isinstance(
        model_sampling, comfy.model_sampling.CONST
    ):
        raise TypeError(
            "MiniMax H3 SA-Solver scheduler requires ModelSamplingAV with CONST flow sampling"
        )
    _validated_sampling_metadata(model_sampling)


def _simple_schedule(model_sampling: Any, steps: int) -> torch.Tensor:
    import comfy.samplers

    # This is the control contract. The table selection is intentionally left
    # to current ComfyUI instead of replacing its discrete behavior.
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", steps).cpu()
    if sigmas.shape != (steps + 1,):
        raise ValueError("ComfyUI simple scheduler returned the wrong number of sigma points")
    if not torch.isfinite(sigmas).all() or sigmas[-1].item() != 0.0:
        raise ValueError("ComfyUI simple scheduler returned an invalid schedule")
    return sigmas


def _adams_phase_coefficients(
    model_sampling: Any,
    effective: torch.Tensor,
    lambdas: torch.Tensor,
    *,
    predictor_order: int = 3,
    corrector_order: int = 4,
) -> list[dict[str, Any]]:
    """Return native SA phase coefficients using current ComfyUI PECE topology."""
    from comfy.k_diffusion import sa_solver

    start_sigma = model_sampling.percent_to_sigma(0.2)
    end_sigma = model_sampling.percent_to_sigma(0.8)
    tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    phase_records: list[dict[str, Any]] = []
    interval_count = effective.numel() - 1
    lower_order_to_end = effective[-1].item() == 0.0
    for index in range(interval_count):
        predictor_used = min(predictor_order, index + 1)
        corrector_used = 0 if index == 0 else min(corrector_order, index + 1)
        if lower_order_to_end:
            predictor_used = min(predictor_used, effective.numel() - 2 - index)
            corrector_used = min(corrector_used, effective.numel() - 1 - index)

        if corrector_used > 0:
            try:
                tau = tau_func(effective[index])
                curr = lambdas[index - corrector_used + 1 : index + 1]
                coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                    effective[index],
                    curr,
                    lambdas[index - 1],
                    lambdas[index],
                    tau,
                    simple_order_2=False,
                    is_corrector_step=True,
                ).detach().cpu().double()
            except (RuntimeError, ValueError) as error:
                phase_records.append(
                    {
                        "phase": "corrector",
                        "outer_index": index,
                        "order": corrector_used,
                        "tau": None,
                        "coeffs": None,
                        "error": error,
                    }
                )
            else:
                phase_records.append(
                    {
                        "phase": "corrector",
                        "outer_index": index,
                        "order": corrector_used,
                        "tau": tau,
                        "coeffs": coeffs,
                        "error": None,
                    }
                )

        if predictor_used > 0 and effective[index + 1].item() != 0.0:
            try:
                tau = tau_func(effective[index + 1])
                curr = lambdas[index - predictor_used + 1 : index + 1]
                coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                    effective[index + 1],
                    curr,
                    lambdas[index],
                    lambdas[index + 1],
                    tau,
                    simple_order_2=False,
                    is_corrector_step=False,
                ).detach().cpu().double()
            except (RuntimeError, ValueError) as error:
                phase_records.append(
                    {
                        "phase": "predictor",
                        "outer_index": index,
                        "order": predictor_used,
                        "tau": None,
                        "coeffs": None,
                        "error": error,
                    }
                )
            else:
                phase_records.append(
                    {
                        "phase": "predictor",
                        "outer_index": index,
                        "order": predictor_used,
                        "tau": tau,
                        "coeffs": coeffs,
                        "error": None,
                    }
                )

    return phase_records


def _adams_records(
    model_sampling: Any,
    sigmas: torch.Tensor,
    *,
    predictor_order: int = 3,
    corrector_order: int = 4,
) -> list[dict[str, Any]]:
    """Return native SA coefficient diagnostics used only to rank bounded trials."""
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr, sigma_to_half_log_snr

    effective = offset_first_sigma_for_snr(sigmas, model_sampling).detach().cpu().double()
    lambdas = sigma_to_half_log_snr(
        effective,
        model_sampling=model_sampling,
    ).detach().cpu().double()
    if not torch.isfinite(lambdas[:-1]).all() or not torch.isposinf(lambdas[-1]):
        raise ValueError("SA-Solver schedule produced invalid half-log-SNR coordinates")

    records: list[dict[str, Any]] = []
    for phase_record in _adams_phase_coefficients(
        model_sampling,
        effective,
        lambdas,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
    ):
        error = phase_record["error"]
        if error is not None:
            raise error
        coeffs = phase_record["coeffs"]
        if not torch.isfinite(coeffs).all():
            raise ValueError(
                f"native SA {phase_record['phase']} coefficient calculation produced NaN or Inf"
            )
        records.append(
            {
                "phase": phase_record["phase"],
                "outer_index": phase_record["outer_index"],
                "order": phase_record["order"],
                "l1_norm": float(coeffs.abs().sum()),
                "max_abs": float(coeffs.abs().max()),
                "l2_norm": float(torch.linalg.vector_norm(coeffs)),
            }
        )

    return records


def _base_contract(
    control_base: torch.Tensor,
    candidate_base: torch.Tensor,
) -> dict[str, float]:
    control_intervals = control_base[:-1] - control_base[1:]
    candidate_intervals = candidate_base[:-1] - candidate_base[1:]
    if torch.any(control_intervals <= 0.0) or torch.any(candidate_intervals <= 0.0):
        raise ValueError("SA-Solver shared-base intervals must stay strictly positive")

    ratios = candidate_intervals / control_intervals
    mean_control_interval = float(control_intervals.mean())
    if not math.isfinite(mean_control_interval) or mean_control_interval <= 0.0:
        raise ValueError("SA-Solver control schedule has an invalid mean base interval")
    displacement = (candidate_base - control_base).abs()
    return {
        "min_interval_ratio": float(ratios.min()),
        "max_interval_ratio": float(ratios.max()),
        "max_node_displacement": float(displacement.max()),
        "mean_control_interval": mean_control_interval,
        "max_node_displacement_in_mean_intervals": float(displacement.max())
        / mean_control_interval,
    }


def _validate_base_contract(
    control_base: torch.Tensor,
    candidate_base: torch.Tensor,
) -> dict[str, float]:
    contract = _base_contract(control_base, candidate_base)
    if contract["min_interval_ratio"] < BASE_INTERVAL_MIN_RATIO:
        raise ValueError("SA-Solver candidate compressed a shared-base interval below 0.875x simple")
    if contract["max_interval_ratio"] > BASE_INTERVAL_MAX_RATIO:
        raise ValueError("SA-Solver candidate expanded a shared-base interval above 1.125x simple")
    if (
        contract["max_node_displacement_in_mean_intervals"]
        > MAX_NODE_DISPLACEMENT_MEAN_INTERVALS
    ):
        raise ValueError(
            "SA-Solver candidate displaced a base node by more than 0.125 mean simple intervals"
        )
    return contract


def _bounded_adams_schedule(
    model_sampling: Any,
    simple_sigmas: torch.Tensor,
    video_shift: float,
    multiplier: float,
) -> torch.Tensor:
    """Move at most one simple base node within a hard local distortion envelope."""
    steps = simple_sigmas.numel() - 1
    if steps < 2:
        return simple_sigmas.clone()

    control_base = shifted_sigma_to_base(
        simple_sigmas.detach().cpu().double(),
        video_shift,
    )
    _validate_base(control_base, steps)

    baseline_records = _adams_records(model_sampling, simple_sigmas)
    if not baseline_records:
        return simple_sigmas.clone()
    worst = max(baseline_records, key=lambda record: record["l1_norm"])
    baseline_worst_l1 = float(worst["l1_norm"])

    support_start = int(worst["outer_index"]) - int(worst["order"]) + 1
    support_end = int(worst["outer_index"])
    control_intervals = control_base[:-1] - control_base[1:]

    best_sigmas: torch.Tensor | None = None
    best_key: tuple[float, float, int, float] | None = None

    for node_index in range(support_start, support_end + 1):
        if node_index <= 0 or node_index >= steps:
            continue
        local_interval = min(
            float(control_intervals[node_index - 1]),
            float(control_intervals[node_index]),
        )
        if not math.isfinite(local_interval) or local_interval <= 0.0:
            continue

        for scale in _SEARCH_SCALES:
            candidate_base = control_base.clone()
            candidate_base[node_index] += scale * _SEARCH_INTERVAL_FRACTION * local_interval
            try:
                _validate_base(candidate_base, steps)
            except ValueError:
                continue

            candidate_sigmas = simple_sigmas.clone()
            mapped = _map_base_with_model(
                model_sampling,
                candidate_base[node_index : node_index + 1],
                multiplier,
            )
            candidate_sigmas[node_index] = mapped[0]
            try:
                candidate_sigmas = _validate_sigmas(candidate_sigmas, steps)
            except ValueError:
                continue

            roundtrip_base = shifted_sigma_to_base(
                candidate_sigmas.detach().cpu().double(),
                video_shift,
            )
            try:
                _validate_base_contract(control_base, roundtrip_base)
            except ValueError:
                continue

            try:
                records = _adams_records(model_sampling, candidate_sigmas)
            except (RuntimeError, ValueError):
                continue
            if not records:
                continue
            worst_l1 = max(float(record["l1_norm"]) for record in records)
            displacement = abs(float(roundtrip_base[node_index] - control_base[node_index]))
            key = (worst_l1, displacement, node_index, scale)
            if best_key is None or key < best_key:
                best_key = key
                best_sigmas = candidate_sigmas

    if best_sigmas is None or best_key is None:
        return simple_sigmas.clone()
    if best_key[0] >= baseline_worst_l1 - 1.0e-12:
        return simple_sigmas.clone()
    return best_sigmas


def _failed_lambda_redistributed_schedule(
    model_sampling: Any,
    simple_sigmas: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """Retain the failed lambda family only for explicit diagnostic comparison."""
    from comfy.k_diffusion.sampling import (
        half_log_snr_to_sigma,
        offset_first_sigma_for_snr,
        sigma_to_half_log_snr,
    )

    steps = simple_sigmas.numel() - 1
    if steps < 1:
        raise ValueError("SA-Solver schedule must contain at least one outer interval")
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("diagnostic lambda blend must be finite and in [0, 1]")
    if blend == 0.0:
        return simple_sigmas.clone()

    effective = offset_first_sigma_for_snr(simple_sigmas, model_sampling)
    effective_nonterminal = effective[:-1].to(device="cpu", dtype=torch.float64)
    simple_lambdas = sigma_to_half_log_snr(
        effective_nonterminal,
        model_sampling=model_sampling,
    ).to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(simple_lambdas).all():
        raise ValueError("effective ComfyUI simple envelope produced nonfinite lambda values")
    if simple_lambdas.numel() > 1 and torch.any(simple_lambdas[1:] <= simple_lambdas[:-1]):
        raise ValueError(
            "ComfyUI simple envelope must have strictly increasing finite lambda points"
        )

    uniform_lambdas = torch.linspace(
        simple_lambdas[0],
        simple_lambdas[-1],
        steps,
        device="cpu",
        dtype=torch.float64,
    )
    candidate_lambdas = torch.lerp(simple_lambdas, uniform_lambdas, blend)
    candidate_nonterminal = half_log_snr_to_sigma(
        candidate_lambdas,
        model_sampling=model_sampling,
    ).to(device="cpu", dtype=torch.float64)

    candidate = torch.cat((candidate_nonterminal, torch.zeros(1, dtype=torch.float64)))
    candidate[0] = simple_sigmas[0]
    candidate[-2] = simple_sigmas[-2]
    candidate[-1] = 0.0
    return _validate_sigmas(candidate, steps)


def h3_sa_solver_sigmas(
    model_sampling: Any,
    steps: int,
    denoise: float,
    mode: str,
) -> torch.Tensor:
    """Build a bounded H3 schedule for SA-Solver without changing sampler topology.

    simple_control is exact current-ComfyUI parity. simple_adams_bounded starts
    from that control and permits at most one adjacent-interval transfer around
    the worst native Adams L1 record, subject to a hard 0.875x..1.125x interval
    envelope and a 0.125-mean-interval node-displacement limit. Denoise follows
    BasicScheduler: build the longer schedule and return the requested tail.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(denoise) or not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be finite and in [0, 1]")
    if denoise == 0.0:
        return torch.empty(0, dtype=torch.float32, device="cpu")
    if mode not in SA_SCHEDULER_MODES:
        raise ValueError(f"unknown MiniMax H3 SA-Solver scheduler mode {mode!r}")

    _require_h3_sa_sampling(model_sampling)
    _, video_shift, _, multiplier = _validated_sampling_metadata(model_sampling)
    full_steps = int(steps / denoise) if denoise < 1.0 else steps
    simple_sigmas = _simple_schedule(model_sampling, full_steps)

    if mode == "simple_control":
        full_sigmas = simple_sigmas
    else:
        full_sigmas = _bounded_adams_schedule(
            model_sampling,
            simple_sigmas,
            video_shift,
            multiplier,
        )

    result = full_sigmas[-(steps + 1) :].clone()
    if mode == "simple_control":
        return result
    return _validate_sigmas(result, steps)


__all__ = [
    "BASE_INTERVAL_MAX_RATIO",
    "BASE_INTERVAL_MIN_RATIO",
    "MAX_NODE_DISPLACEMENT_MEAN_INTERVALS",
    "SA_SCHEDULER_MODES",
    "h3_sa_solver_sigmas",
]
