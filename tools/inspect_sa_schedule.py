from __future__ import annotations

import argparse
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfyui_refdelta_solver.h3_scheduler import (
    base_to_shifted_sigma,
    h3_uniform_flow_sigmas,
    shifted_sigma_to_base,
)
from comfyui_refdelta_solver.sa_scheduler import (
    BASE_INTERVAL_MAX_RATIO,
    BASE_INTERVAL_MIN_RATIO,
    MAX_NODE_DISPLACEMENT_MEAN_INTERVALS,
    _adams_phase_coefficients,
    _failed_lambda_redistributed_schedule,
    _simple_schedule,
    h3_sa_solver_sigmas,
)


COMPARISON_MODES = (
    "simple",
    "simple_adams_bounded",
    "uniform_linspace",
    "phase_offset_uniform",
    "failed_simple_lambda_uniform",
    "failed_simple_lambda_blend",
    "beta",
)


def prepare_comfyui(comfyui_path: Path) -> None:
    path = str(comfyui_path.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    if not torch.cuda.is_available():
        original_argv = sys.argv[:]
        try:
            sys.argv[:] = [original_argv[0], "--cpu"]
            import comfy.options

            comfy.options.enable_args_parsing()
            import comfy.cli_args
        finally:
            sys.argv[:] = original_argv

        try:
            import comfy_kitchen
        except ModuleNotFoundError as error:
            if error.name != "comfy_kitchen":
                raise
        else:
            if not hasattr(comfy_kitchen, "int8_attention_is_available"):
                comfy_kitchen.int8_attention_is_available = lambda: False


def make_sampling(
    video_shift: float,
    audio_shift: float,
    timesteps: int,
    multiplier: float,
):
    import comfy.model_sampling

    class InspectionSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    sampling = InspectionSampling()
    sampling.set_parameters(
        shift=video_shift,
        audio_shift=audio_shift,
        timesteps=timesteps,
        multiplier=multiplier,
    )
    return sampling


def _full_steps(steps: int, denoise: float) -> int:
    return int(steps / denoise) if denoise < 1.0 else steps


def schedule_for_mode(
    model_sampling,
    mode: str,
    steps: int,
    denoise: float,
    *,
    phase: float,
    lambda_blend: float,
) -> torch.Tensor:
    if mode == "simple":
        return h3_sa_solver_sigmas(model_sampling, steps, denoise, "simple_control")
    if mode == "simple_adams_bounded":
        return h3_sa_solver_sigmas(model_sampling, steps, denoise, mode)
    if mode in {"failed_simple_lambda_uniform", "failed_simple_lambda_blend"}:
        full_steps = _full_steps(steps, denoise)
        simple_full = _simple_schedule(model_sampling, full_steps)
        blend = 1.0 if mode == "failed_simple_lambda_uniform" else lambda_blend
        failed_full = _failed_lambda_redistributed_schedule(
            model_sampling,
            simple_full,
            blend,
        )
        return failed_full[-(steps + 1) :].clone()
    if mode in {"uniform_linspace", "phase_offset_uniform"}:
        return h3_uniform_flow_sigmas(
            model_sampling,
            steps,
            denoise,
            mode,
            phase=phase,
        )
    if mode == "beta":
        import comfy.samplers

        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling,
            "beta",
            _full_steps(steps, denoise),
        ).cpu()
        return sigmas[-(steps + 1) :].clone()
    raise ValueError(f"unknown comparison mode {mode!r}")


def _number(value: torch.Tensor | float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _positive_stats(values: torch.Tensor) -> dict[str, float | None]:
    finite = values[torch.isfinite(values) & (values > 0)]
    if finite.numel() == 0:
        return {"min": None, "max": None, "mean": None, "max_min_ratio": None}
    minimum = float(finite.min())
    maximum = float(finite.max())
    return {
        "min": minimum,
        "max": maximum,
        "mean": float(finite.mean()),
        "max_min_ratio": maximum / minimum,
    }


def coordinate_diagnostics(
    model_sampling,
    sigmas: torch.Tensor,
    *,
    control_sigmas: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr, sigma_to_half_log_snr

    returned = sigmas.detach().cpu().double()
    effective = offset_first_sigma_for_snr(sigmas, model_sampling).detach().cpu().double()
    base = shifted_sigma_to_base(returned, float(model_sampling.shift))
    audio = base_to_shifted_sigma(base, float(model_sampling.audio_shift))
    lambdas = sigma_to_half_log_snr(effective, model_sampling=model_sampling).double()

    delta_sigma = returned[:-1] - returned[1:]
    delta_base = base[:-1] - base[1:]
    delta_lambda = lambdas[1:] - lambdas[:-1]

    control_base = None
    interval_ratios = None
    node_displacement = None
    mean_control_interval = None
    if control_sigmas is not None:
        control = control_sigmas.detach().cpu().double()
        if control.shape != returned.shape:
            raise ValueError("control schedule must have the same shape as inspected sigmas")
        control_base = shifted_sigma_to_base(control, float(model_sampling.shift))
        control_intervals = control_base[:-1] - control_base[1:]
        if torch.any(control_intervals <= 0.0):
            raise ValueError("control schedule has invalid shared-base intervals")
        interval_ratios = delta_base / control_intervals
        node_displacement = (base - control_base).abs()
        mean_control_interval = float(control_intervals.mean())

    rows: list[dict[str, Any]] = []
    for index in range(returned.numel()):
        rows.append(
            {
                "index": index,
                "returned_sigma": float(returned[index]),
                "effective_sa_sigma": float(effective[index]),
                "shared_base_time": float(base[index]),
                "shifted_video_sigma": float(returned[index]),
                "implied_audio_sigma": float(audio[index]),
                "half_log_snr_lambda": _number(lambdas[index]),
                "delta_lambda": _number(delta_lambda[index]) if index < delta_lambda.numel() else None,
                "delta_base_time": float(delta_base[index]) if index < delta_base.numel() else None,
                "delta_sigma": float(delta_sigma[index]) if index < delta_sigma.numel() else None,
                "base_interval_ratio_vs_simple": (
                    float(interval_ratios[index])
                    if interval_ratios is not None and index < interval_ratios.numel()
                    else None
                ),
                "node_displacement_vs_simple": (
                    float(node_displacement[index]) if node_displacement is not None else None
                ),
            }
        )

    finite_lambda_deltas = delta_lambda[torch.isfinite(delta_lambda)]
    summary = {
        "all_returned_coordinates_finite": bool(
            torch.isfinite(returned).all()
            and torch.isfinite(effective).all()
            and torch.isfinite(base).all()
            and torch.isfinite(audio).all()
        ),
        "finite_nonterminal_lambda": bool(torch.isfinite(lambdas[:-1]).all()),
        "terminal_lambda_is_positive_infinity": bool(torch.isposinf(lambdas[-1])),
        "strict_sigma_monotonicity": bool(torch.all(returned[1:] < returned[:-1])),
        "strict_lambda_monotonicity": bool(torch.all(lambdas[1:-1] > lambdas[:-2])),
        "delta_lambda": _positive_stats(finite_lambda_deltas.abs()),
        "delta_base_time": _positive_stats(delta_base.abs()),
        "delta_sigma": _positive_stats(delta_sigma.abs()),
        "first_interval_base_time": float(delta_base[0]),
        "final_interval_base_time": float(delta_base[-1]),
        "first_interval_sigma": float(delta_sigma[0]),
        "final_interval_sigma": float(delta_sigma[-1]),
        "min_base_interval_ratio_vs_simple": (
            float(interval_ratios.min()) if interval_ratios is not None else None
        ),
        "max_base_interval_ratio_vs_simple": (
            float(interval_ratios.max()) if interval_ratios is not None else None
        ),
        "max_node_displacement_vs_simple": (
            float(node_displacement.max()) if node_displacement is not None else None
        ),
        "max_node_displacement_in_simple_intervals": (
            float(node_displacement.max()) / mean_control_interval
            if node_displacement is not None and mean_control_interval
            else None
        ),
    }
    return rows, summary


def _coefficient_record(
    *,
    phase: str,
    outer_index: int,
    order: int,
    tau: float,
    coeffs: torch.Tensor,
) -> dict[str, Any]:
    values = coeffs.detach().cpu().double()
    absolute = values.abs()
    nonzero = absolute[absolute > 0]
    dynamic_range = _number(absolute.max() / nonzero.min()) if nonzero.numel() else 0.0
    return {
        "phase": phase,
        "outer_index": outer_index,
        "order": order,
        "tau": tau,
        "coefficients": [_number(value) for value in values],
        "all_finite": bool(torch.isfinite(values).all()),
        "max_abs": _number(absolute.max()),
        "l1_norm": _number(absolute.sum()),
        "l2_norm": _number(torch.linalg.vector_norm(values)),
        "dynamic_range": dynamic_range,
    }


def adams_diagnostics(
    model_sampling,
    sigmas: torch.Tensor,
    *,
    predictor_order: int = 3,
    corrector_order: int = 4,
) -> dict[str, Any]:
    from comfy.k_diffusion.sampling import offset_first_sigma_for_snr, sigma_to_half_log_snr

    effective = offset_first_sigma_for_snr(sigmas, model_sampling).detach().cpu().double()
    lambdas = sigma_to_half_log_snr(effective, model_sampling=model_sampling).double()

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for phase_record in _adams_phase_coefficients(
        model_sampling,
        effective,
        lambdas,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
    ):
        error = phase_record["error"]
        if error is not None:
            errors.append(
                {
                    "phase": phase_record["phase"],
                    "outer_index": phase_record["outer_index"],
                    "error": str(error),
                }
            )
            continue
        records.append(
            _coefficient_record(
                phase=phase_record["phase"],
                outer_index=phase_record["outer_index"],
                order=phase_record["order"],
                tau=phase_record["tau"],
                coeffs=phase_record["coeffs"],
            )
        )

    phase_maxima: dict[str, float | None] = {}
    phase_growth: dict[str, float | None] = {}
    for phase in ("predictor", "corrector"):
        maxima = [
            record["max_abs"]
            for record in records
            if record["phase"] == phase and record["max_abs"] is not None
        ]
        phase_maxima[phase] = max(maxima) if maxima else None
        growth = [right / left for left, right in pairwise(maxima) if left > 0]
        phase_growth[phase] = max(growth) if growth else None

    finite_max_records = [record for record in records if record["max_abs"] is not None]
    worst = max(finite_max_records, key=lambda record: record["max_abs"], default=None)
    finite_l1_records = [record for record in records if record["l1_norm"] is not None]
    worst_l1 = max(finite_l1_records, key=lambda record: record["l1_norm"], default=None)
    maximum_l2 = max(
        (record["l2_norm"] for record in records if record["l2_norm"] is not None),
        default=None,
    )
    maximum_dynamic_range = max(
        (
            record["dynamic_range"]
            for record in records
            if record["dynamic_range"] is not None
        ),
        default=None,
    )
    return {
        "predictor_order": predictor_order,
        "corrector_order": corrector_order,
        "records": records,
        "errors": errors,
        "all_coefficients_finite": not errors
        and all(record["all_finite"] for record in records),
        "maximum_absolute_coefficient": worst["max_abs"] if worst else None,
        "maximum_l1_norm": worst_l1["l1_norm"] if worst_l1 else None,
        "maximum_l2_norm": maximum_l2,
        "maximum_coefficient_dynamic_range": maximum_dynamic_range,
        "worst_outer_interval": worst["outer_index"] if worst else None,
        "worst_phase": worst["phase"] if worst else None,
        "worst_l1_outer_interval": worst_l1["outer_index"] if worst_l1 else None,
        "worst_l1_phase": worst_l1["phase"] if worst_l1 else None,
        "phase_maximum_absolute_coefficient": phase_maxima,
        "maximum_interval_growth": phase_growth,
        "unexpectedly_extreme_coefficient": bool(
            worst is not None
            and worst["max_abs"] is not None
            and worst["max_abs"] > 1.0e6
        ),
    }


def build_report(
    model_sampling,
    modes: list[str],
    *,
    steps: int,
    denoise: float,
    phase: float,
    lambda_blend: float,
    comfyui_revision: str | None,
) -> dict[str, Any]:
    simple_sigmas = schedule_for_mode(
        model_sampling,
        "simple",
        steps,
        denoise,
        phase=phase,
        lambda_blend=lambda_blend,
    )
    schedules = []
    for mode in modes:
        sigmas = (
            simple_sigmas.clone()
            if mode == "simple"
            else schedule_for_mode(
                model_sampling,
                mode,
                steps,
                denoise,
                phase=phase,
                lambda_blend=lambda_blend,
            )
        )
        rows, summary = coordinate_diagnostics(
            model_sampling,
            sigmas,
            control_sigmas=simple_sigmas,
        )
        schedules.append(
            {
                "mode": mode,
                "rows": rows,
                "summary": summary,
                "adams": adams_diagnostics(model_sampling, sigmas),
            }
        )

    simple_entry = next(schedule for schedule in schedules if schedule["mode"] == "simple")
    simple_l1 = simple_entry["adams"]["maximum_l1_norm"]
    for schedule in schedules:
        current_l1 = schedule["adams"]["maximum_l1_norm"]
        schedule["adams"]["maximum_l1_improvement_vs_simple"] = (
            simple_l1 - current_l1
            if simple_l1 is not None and current_l1 is not None
            else None
        )

    return {
        "authority": {
            "comfyui_revision": comfyui_revision,
            "sigma_to_lambda": "comfy.k_diffusion.sampling.sigma_to_half_log_snr",
            "lambda_to_sigma": "comfy.k_diffusion.sampling.half_log_snr_to_sigma",
            "first_sigma_protection": "comfy.k_diffusion.sampling.offset_first_sigma_for_snr",
            "adams_coefficients": "comfy.k_diffusion.sa_solver.compute_stochastic_adams_b_coeffs",
        },
        "configuration": {
            "outer_steps": steps,
            "denoise": denoise,
            "video_shift": float(model_sampling.shift),
            "audio_shift": float(model_sampling.audio_shift),
            "multiplier": float(model_sampling.multiplier),
            "phase_offset": phase,
            "failed_lambda_blend": lambda_blend,
            "bounded_base_interval_min_ratio": BASE_INTERVAL_MIN_RATIO,
            "bounded_base_interval_max_ratio": BASE_INTERVAL_MAX_RATIO,
            "bounded_max_node_displacement_mean_intervals": (
                MAX_NODE_DISPLACEMENT_MEAN_INTERVALS
            ),
            "active_pece_logical_opportunities": 2 * steps - 1,
            "active_pece_callbacks": steps,
        },
        "schedules": schedules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MiniMax-H3 SA-Solver sigma/base/lambda and Adams geometry."
    )
    default_comfyui = os.environ.get("COMFYUI_PATH")
    parser.add_argument(
        "--comfyui-path",
        type=Path,
        default=Path(default_comfyui) if default_comfyui else None,
        required=default_comfyui is None,
    )
    parser.add_argument("--comfyui-revision")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--multiplier", type=float, default=1000.0)
    parser.add_argument("--phase", type=float, default=0.50)
    parser.add_argument(
        "--failed-lambda-blend",
        type=float,
        default=0.50,
        help="Diagnostic-only blend for the media-failed lambda family.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=COMPARISON_MODES,
        default=list(COMPARISON_MODES),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prepare_comfyui(args.comfyui_path)
    sampling = make_sampling(
        args.video_shift,
        args.audio_shift,
        args.timesteps,
        args.multiplier,
    )
    report = build_report(
        sampling,
        args.modes,
        steps=args.steps,
        denoise=args.denoise,
        phase=args.phase,
        lambda_blend=args.failed_lambda_blend,
        comfyui_revision=args.comfyui_revision,
    )
    if args.output is None:
        json.dump(report, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
