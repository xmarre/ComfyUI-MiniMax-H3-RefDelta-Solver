from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from .scheduler import CalibrationProfile, calibrated_progress


# The native parity matrix is pinned in .github/workflows/tests.yml.  These are
# the reviewed ComfyUI revisions whose BasicScheduler/ddim_scheduler behavior is
# the compatibility contract for legacy_ddim_uniform.
COMFYUI_PARITY_REVISIONS = (
    "27bca654eb9a70237d93f56a6ea336ab55f8925d",
    "ff6c8a8af144fc9e9e7bc436b1b202f9316848d8",
    "b78cec879b9460d5cb25228a83a942fb78d2cd24",
)

SCHEDULER_MODES = (
    "legacy_ddim_uniform",
    "uniform_linspace",
    "phase_offset_uniform",
    "power_uniform",
    "uniform_refinement_tail",
    "trailing_refined",
    "asymmetric_beta",
    "av_arc_length",
    "piecewise_structure_refinement",
    "curvature_profile",
)

FLOW_PROFILE_VERSION = 2


@dataclass(frozen=True, slots=True)
class FlowDensityProfile:
    profile_id: str
    model_family: str
    status: str
    progress: tuple[float, ...]
    density: tuple[float, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if len(self.progress) != len(self.density) or len(self.progress) < 2:
            raise ValueError("flow profile progress and density must have the same length >= 2")
        if self.progress[0] != 0.0 or self.progress[-1] != 1.0:
            raise ValueError("flow profile progress must cover exactly [0, 1]")
        if any(not math.isfinite(value) for value in self.progress + self.density):
            raise ValueError("flow profile values must be finite")
        if any(right <= left for left, right in zip(self.progress, self.progress[1:])):
            raise ValueError("flow profile progress must be strictly increasing")
        if any(value <= 0.0 for value in self.density):
            raise ValueError("flow profile density must be positive")


def flow_profile_from_dict(data: dict[str, Any]) -> FlowDensityProfile:
    if data.get("version") != FLOW_PROFILE_VERSION:
        raise ValueError(f"unsupported shared-flow profile version {data.get('version')!r}")
    if data.get("domain") != "shared-base-time-progress-density":
        raise ValueError("shared-flow profile must declare shared-base-time-progress-density")
    points = data.get("points")
    if not isinstance(points, list):
        raise TypeError("shared-flow profile points must be a list")
    if any(not isinstance(point, dict) or set(point) != {"progress", "density"} for point in points):
        raise ValueError("shared-flow profile points may contain only progress and density")
    metadata = dict(data.get("metadata", {}))
    if metadata.get("comparison_metrics_used_for_density") is not False:
        raise ValueError("shared-flow profiles must explicitly exclude comparison metrics")
    if metadata.get("evidence_source") not in {"neutral_control", "production_actual_trajectory"}:
        raise ValueError("shared-flow profile must declare an actual-only or neutral evidence source")
    profile = FlowDensityProfile(
        profile_id=str(data["id"]),
        model_family=str(data.get("model_family", "MiniMax-H3 RefDelta")),
        status=str(data.get("status", "unknown")),
        progress=tuple(float(point["progress"]) for point in points),
        density=tuple(float(point["density"]) for point in points),
        metadata=metadata,
    )
    profile.validate()
    return profile


def load_flow_profile(name: str, path: str | Path | None = None) -> FlowDensityProfile:
    if path:
        candidate = Path(path).expanduser()
    else:
        candidate = Path(str(files("comfyui_refdelta_solver").joinpath("profiles", f"{name}.json")))
    if not candidate.is_file():
        raise FileNotFoundError(f"shared-flow profile {candidate} was not found")
    with candidate.open("r", encoding="utf-8") as handle:
        return flow_profile_from_dict(json.load(handle))


def base_to_shifted_sigma(base_time: torch.Tensor, shift: float) -> torch.Tensor:
    """Map H3's shared flow coordinate to one stream's shifted sigma."""
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("H3 flow shifts must be positive and finite")
    return shift * base_time / (1.0 + (shift - 1.0) * base_time)


def shifted_sigma_to_base(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Invert one stream's shifted sigma to H3's shared flow coordinate."""
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("H3 flow shifts must be positive and finite")
    denominator = shift + sigma * (1.0 - shift)
    if torch.any(denominator <= 0.0):
        raise ValueError("shifted sigma cannot be inverted for the supplied H3 shift")
    return sigma / denominator


def _validated_sampling_metadata(model_sampling: Any) -> tuple[torch.Tensor, float, float, float]:
    required = ("sigmas", "sigma", "shift", "audio_shift", "multiplier")
    if any(not hasattr(model_sampling, name) for name in required):
        raise ValueError("MiniMax H3 uniform-flow scheduler requires ModelSamplingAV")

    shift = float(model_sampling.shift)
    audio_shift = float(model_sampling.audio_shift)
    multiplier = float(model_sampling.multiplier)
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("H3 video shift must be positive and finite")
    if not math.isfinite(audio_shift) or audio_shift <= 0.0:
        raise ValueError("H3 audio shift must be positive and finite")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("model sampling multiplier must be positive and finite")

    table = torch.as_tensor(model_sampling.sigmas).detach().to(device="cpu", dtype=torch.float64).flatten()
    if table.numel() < 2 or not torch.isfinite(table).all():
        raise ValueError("model sampling sigma table must contain at least two finite values")
    if torch.any(table[1:] <= table[:-1]):
        raise ValueError("model sampling sigma table must be strictly increasing")
    return table, shift, audio_shift, multiplier


def _uniform_base(steps: int) -> torch.Tensor:
    return torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)


def _phase_base(steps: int, phase: float) -> torch.Tensor:
    if not math.isfinite(phase) or not 0.0 <= phase < 1.0:
        raise ValueError("phase must be finite and in [0, 1)")
    if phase == 0.0:
        return _uniform_base(steps)
    indices = torch.arange(steps, dtype=torch.float64)
    nonterminal = 1.0 - (indices + phase) / steps
    return torch.cat((nonterminal, torch.zeros(1, dtype=torch.float64)))


def _power_base(steps: int, power: float) -> torch.Tensor:
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("power must be positive and finite")
    if power == 1.0:
        return _uniform_base(steps)
    progress = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    return (1.0 - progress).pow(power)


def _joined_tail_base(
    steps: int,
    tail_steps: int,
    tail_start: float,
    tail_power: float,
    *,
    top: float = 1.0,
) -> torch.Tensor:
    if not isinstance(tail_steps, int) or tail_steps < 1 or tail_steps >= steps:
        raise ValueError("tail_steps must be an integer in [1, effective_steps - 1]")
    if not math.isfinite(tail_start) or not 0.0 < tail_start < 1.0:
        raise ValueError("tail_start must be finite and in (0, 1)")
    if not math.isfinite(tail_power) or tail_power <= 0.0:
        raise ValueError("tail_power must be positive and finite")
    if not math.isfinite(top) or not tail_start < top <= 1.0:
        raise ValueError("tail body top must be finite and above tail_start")

    body_steps = steps - tail_steps
    body = torch.linspace(top, tail_start, body_steps + 1, dtype=torch.float64)
    tail_progress = torch.arange(1, tail_steps + 1, dtype=torch.float64) / tail_steps
    tail = tail_start * (1.0 - tail_progress).pow(tail_power)
    return torch.cat((body, tail))


def _resolve_tail_steps(tail_steps: int | None, steps: int) -> int:
    if tail_steps is None:
        return min(5, steps - 1)
    if not isinstance(tail_steps, int) or tail_steps < 0 or tail_steps >= steps:
        raise ValueError("tail_steps must be an integer in [0, effective_steps - 1]")
    return tail_steps


def _uniform_refinement_tail_base(
    steps: int,
    tail_steps: int,
    tail_start: float,
    tail_power: float,
) -> torch.Tensor:
    if tail_steps == 0:
        return _uniform_base(steps)
    # The body is uniform from u=1 through the explicit join.  Reserving a
    # smaller tail span makes the first tail interval comparable to the body
    # while still allowing the final intervals to become denser.
    return _joined_tail_base(steps, tail_steps, tail_start, tail_power)


def _trailing_refined_base(
    steps: int,
    tail_steps: int,
    tail_start: float,
    tail_power: float,
    table_points: int,
) -> torch.Tensor:
    if tail_steps == 0:
        return _uniform_base(steps)
    # Diffusers-style trailing uses the final zero-based training-index
    # normalization (N - 1) / N rather than the mathematical u=1 endpoint.
    # The model table supplies N only; its actual final entry maps to u=1 for
    # current ModelSamplingDiscreteFlow.  Explicitly resolve [tail_start, 0]
    # instead of appending one large shifted-sigma jump.
    top = (table_points - 1) / table_points
    return _joined_tail_base(steps, tail_steps, tail_start, tail_power, top=top)


def _beta_base(steps: int, alpha: float, beta: float) -> torch.Tensor:
    if not math.isfinite(alpha) or alpha <= 0.0 or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta_alpha and beta_beta must be positive and finite")

    import numpy
    from scipy.stats import beta as beta_distribution

    probabilities = 1.0 - numpy.arange(steps, dtype=numpy.float64) / steps
    quantiles = numpy.asarray(beta_distribution.ppf(probabilities, alpha, beta), dtype=numpy.float64)
    if not numpy.isfinite(quantiles).all():
        raise ValueError("beta quantile calculation produced NaN or Inf")
    return torch.cat((torch.from_numpy(quantiles.copy()), torch.zeros(1, dtype=torch.float64)))


def _arc_length_base(
    steps: int,
    video_shift: float,
    audio_shift: float,
    arc_strength: float,
    audio_weight: float,
    integration_points: int = 16385,
) -> torch.Tensor:
    if not math.isfinite(arc_strength) or not 0.0 <= arc_strength <= 1.0:
        raise ValueError("arc_strength must be finite and in [0, 1]")
    if not math.isfinite(audio_weight) or audio_weight < 0.0:
        raise ValueError("audio_weight must be finite and nonnegative")
    uniform = _uniform_base(steps)
    if arc_strength == 0.0:
        return uniform

    grid = torch.linspace(0.0, 1.0, integration_points, dtype=torch.float64)
    video_speed = video_shift / (1.0 + (video_shift - 1.0) * grid).square()
    audio_speed = audio_shift / (1.0 + (audio_shift - 1.0) * grid).square()
    speed = torch.sqrt(video_speed.square() + audio_weight * audio_speed.square())
    increments = 0.5 * (speed[1:] + speed[:-1]) * (grid[1:] - grid[:-1])
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(increments, dim=0)))
    cumulative = cumulative / cumulative[-1]

    targets = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    upper = torch.searchsorted(cumulative, targets, right=True).clamp(1, cumulative.numel() - 1)
    lower = upper - 1
    fraction = (targets - cumulative[lower]) / (cumulative[upper] - cumulative[lower])
    arc = torch.lerp(grid[lower], grid[upper], fraction)
    arc[0] = 1.0
    arc[-1] = 0.0
    if arc_strength == 1.0:
        return arc
    return torch.lerp(uniform, arc, arc_strength)


def _piecewise_base(steps: int, structure_fraction: float, mid_power: float, detail_power: float) -> torch.Tensor:
    if not math.isfinite(structure_fraction) or not 0.0 < structure_fraction < 1.0:
        raise ValueError("structure_fraction must be finite and in (0, 1)")
    if not math.isfinite(mid_power) or mid_power <= 0.0:
        raise ValueError("mid_power must be positive and finite")
    if not math.isfinite(detail_power) or detail_power <= 0.0:
        raise ValueError("detail_power must be positive and finite")
    if mid_power == 1.0 and detail_power == 1.0:
        return _uniform_base(steps)

    progress = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    early = structure_fraction * (progress / structure_fraction).clamp(0.0, 1.0).pow(mid_power)
    late_fraction = ((progress - structure_fraction) / (1.0 - structure_fraction)).clamp(0.0, 1.0)
    late = structure_fraction + (1.0 - structure_fraction) * late_fraction.pow(detail_power)
    warped_progress = torch.where(progress <= structure_fraction, early, late)
    return 1.0 - warped_progress


def _profile_base(steps: int, profile: FlowDensityProfile) -> torch.Tensor:
    profile.validate()
    if all(value == profile.density[0] for value in profile.density):
        return _uniform_base(steps)
    compatibility_profile = CalibrationProfile(
        profile_id=profile.profile_id,
        model_family=profile.model_family,
        rank=None,
        status=profile.status,
        progress=profile.progress,
        difficulty=profile.density,
        metadata=profile.metadata,
    )
    return 1.0 - calibrated_progress(compatibility_profile, steps)


def _map_base_with_model(model_sampling: Any, base_time: torch.Tensor, multiplier: float) -> torch.Tensor:
    # ModelSamplingDiscreteFlow.sigma is the authoritative video-shift mapping.
    # ModelSamplingAV/H3 derives audio sigma later from this same shared clock.
    sigmas = torch.as_tensor(model_sampling.sigma(base_time * multiplier)).detach()
    return sigmas.to(device="cpu", dtype=torch.float64).flatten()


def _validate_base(base_time: torch.Tensor, steps: int) -> None:
    if base_time.shape != (steps + 1,):
        raise ValueError("base-time schedule has the wrong number of points")
    if not torch.isfinite(base_time).all() or torch.any(base_time < 0.0) or torch.any(base_time > 1.0):
        raise ValueError("base-time schedule must be finite and remain in [0, 1]")
    if base_time[-1].item() != 0.0:
        raise ValueError("base-time schedule must terminate exactly at zero")
    if torch.any(base_time[1:] >= base_time[:-1]):
        raise ValueError("base-time schedule must be strictly decreasing without duplicate points")


def _validate_sigmas(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32).flatten()
    if sigmas.shape != (steps + 1,):
        raise ValueError("scheduler produced the wrong number of sigma points")
    if not torch.isfinite(sigmas).all() or torch.any(sigmas < 0.0):
        raise ValueError("scheduler produced invalid sigma values")
    if sigmas[-1].item() != 0.0:
        raise ValueError("scheduler must terminate at sigma zero")
    if torch.any(sigmas[1:] >= sigmas[:-1]):
        raise ValueError("scheduler produced duplicate or increasing sigma points")
    return sigmas


def _legacy_ddim_uniform(model_sampling: Any, full_steps: int) -> torch.Tensor:
    table = torch.as_tensor(model_sampling.sigmas).flatten()
    if full_steps > table.numel() - 1:
        raise ValueError("legacy_ddim_uniform effective step count exceeds its unique table-index capacity")

    import comfy.samplers

    # This delegation is deliberate.  Upstream starts at table index 1, uses an
    # integer floor stride, reverses, and may yield an extra point.  BasicScheduler
    # then takes its tail.  Reimplementing a cleaner formula would lose parity.
    upstream = comfy.samplers.calculate_sigmas(model_sampling, "ddim_uniform", full_steps).cpu()
    return upstream[-(full_steps + 1) :]


def h3_uniform_flow_sigmas(
    model_sampling: Any,
    steps: int,
    denoise: float,
    mode: str,
    *,
    phase: float = 0.5,
    power: float = 1.0,
    tail_steps: int | None = None,
    tail_start: float = 0.15,
    tail_power: float = 2.0,
    beta_alpha: float = 0.6,
    beta_beta: float = 0.6,
    arc_strength: float = 0.5,
    audio_weight: float = 1.0,
    structure_fraction: float = 0.5,
    mid_power: float = 1.0,
    detail_power: float = 1.0,
    profile: FlowDensityProfile | None = None,
) -> torch.Tensor:
    """Build an immutable H3 shared-base-time schedule and return video SIGMAS.

    Denoise follows BasicScheduler semantics: construct the longer schedule, then
    take exactly the requested tail including terminal zero.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(denoise) or not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be finite and in [0, 1]")
    if denoise == 0.0:
        return torch.empty(0, dtype=torch.float32, device="cpu")
    if mode not in SCHEDULER_MODES:
        raise ValueError(f"unknown H3 uniform-flow scheduler mode {mode!r}")

    table, video_shift, audio_shift, multiplier = _validated_sampling_metadata(model_sampling)
    full_steps = int(steps / denoise) if denoise < 1.0 else steps

    if mode == "legacy_ddim_uniform":
        full_sigmas = _legacy_ddim_uniform(model_sampling, full_steps)
        full_sigmas = _validate_sigmas(full_sigmas, full_steps)
    else:
        if mode == "uniform_linspace":
            base_time = _uniform_base(full_steps)
        elif mode == "phase_offset_uniform":
            base_time = _phase_base(full_steps, phase)
        elif mode == "power_uniform":
            base_time = _power_base(full_steps, power)
        elif mode == "uniform_refinement_tail":
            base_time = _uniform_refinement_tail_base(
                full_steps,
                _resolve_tail_steps(tail_steps, full_steps),
                tail_start,
                tail_power,
            )
        elif mode == "trailing_refined":
            base_time = _trailing_refined_base(
                full_steps,
                _resolve_tail_steps(tail_steps, full_steps),
                tail_start,
                tail_power,
                table.numel(),
            )
        elif mode == "asymmetric_beta":
            base_time = _beta_base(full_steps, beta_alpha, beta_beta)
        elif mode == "av_arc_length":
            base_time = _arc_length_base(
                full_steps,
                video_shift,
                audio_shift,
                arc_strength,
                audio_weight,
            )
        elif mode == "piecewise_structure_refinement":
            base_time = _piecewise_base(full_steps, structure_fraction, mid_power, detail_power)
        else:
            if profile is None:
                raise ValueError("curvature_profile mode requires an offline calibration profile")
            base_time = _profile_base(full_steps, profile)

        _validate_base(base_time, full_steps)
        full_sigmas = _map_base_with_model(model_sampling, base_time, multiplier)
        full_sigmas[-1] = 0.0
        full_sigmas = _validate_sigmas(full_sigmas, full_steps)

    result = full_sigmas[-(steps + 1) :].clone()
    return _validate_sigmas(result, steps)


__all__ = [
    "COMFYUI_PARITY_REVISIONS",
    "FLOW_PROFILE_VERSION",
    "SCHEDULER_MODES",
    "FlowDensityProfile",
    "base_to_shifted_sigma",
    "flow_profile_from_dict",
    "h3_uniform_flow_sigmas",
    "load_flow_profile",
    "shifted_sigma_to_base",
]
