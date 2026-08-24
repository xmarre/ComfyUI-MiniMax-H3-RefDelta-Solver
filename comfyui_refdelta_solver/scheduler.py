from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch


PROFILE_VERSION = 1
FALLBACK_PROFILE_ID = "uniform-fallback"


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    profile_id: str
    model_family: str
    rank: int | None
    status: str
    progress: tuple[float, ...]
    difficulty: tuple[float, ...]
    metadata: dict[str, Any]

    def validate(self) -> None:
        if len(self.progress) != len(self.difficulty) or len(self.progress) < 2:
            raise ValueError("profile progress and difficulty must have the same length >= 2")
        if self.progress[0] != 0.0 or self.progress[-1] != 1.0:
            raise ValueError("profile progress must cover exactly [0, 1]")
        if any(not math.isfinite(value) for value in self.progress + self.difficulty):
            raise ValueError("profile values must be finite")
        if any(right <= left for left, right in zip(self.progress, self.progress[1:])):
            raise ValueError("profile progress must be strictly increasing")
        if any(value <= 0.0 for value in self.difficulty):
            raise ValueError("profile difficulty must be positive")


def uniform_fallback_profile() -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=FALLBACK_PROFILE_ID,
        model_family="MiniMax-H3 RefDelta",
        rank=None,
        status="fallback",
        progress=(0.0, 1.0),
        difficulty=(1.0, 1.0),
        metadata={"reason": "requested profile was unavailable"},
    )


def profile_from_dict(data: dict[str, Any]) -> CalibrationProfile:
    if data.get("version") != PROFILE_VERSION:
        raise ValueError(f"unsupported calibration profile version {data.get('version')!r}")
    points = data.get("points")
    if not isinstance(points, list):
        raise TypeError("calibration profile points must be a list")
    profile = CalibrationProfile(
        profile_id=str(data["id"]),
        model_family=str(data.get("model_family", "MiniMax-H3 RefDelta")),
        rank=None if data.get("rank") is None else int(data["rank"]),
        status=str(data.get("status", "unknown")),
        progress=tuple(float(point["progress"]) for point in points),
        difficulty=tuple(float(point["difficulty"]) for point in points),
        metadata=dict(data.get("metadata", {})),
    )
    profile.validate()
    return profile


def load_profile(name: str, search_directory: Path | None = None, fallback: bool = True) -> CalibrationProfile:
    candidates: list[Path] = []
    if search_directory is not None:
        candidates.append(search_directory / f"{name}.json")
    candidates.append(Path(str(files("comfyui_refdelta_solver").joinpath("profiles", f"{name}.json"))))
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                return profile_from_dict(json.load(handle))
    if fallback:
        return uniform_fallback_profile()
    raise FileNotFoundError(f"RefDelta calibration profile {name!r} was not found")


def _interpolate_density(profile: CalibrationProfile, progress: torch.Tensor) -> torch.Tensor:
    xp = torch.tensor(profile.progress, dtype=torch.float64, device=progress.device)
    fp = torch.tensor(profile.difficulty, dtype=torch.float64, device=progress.device)
    upper = torch.searchsorted(xp, progress, right=True).clamp(1, xp.numel() - 1)
    lower = upper - 1
    weight = (progress - xp[lower]) / (xp[upper] - xp[lower])
    return torch.lerp(fp[lower], fp[upper], weight)


def calibrated_progress(profile: CalibrationProfile, steps: int, integration_points: int = 4097) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    profile.validate()
    grid = torch.linspace(0.0, 1.0, integration_points, dtype=torch.float64)
    density = _interpolate_density(profile, grid)
    increments = 0.5 * (density[1:] + density[:-1]) * (grid[1:] - grid[:-1])
    cdf = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(increments, dim=0)))
    cdf = cdf / cdf[-1]
    targets = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    upper = torch.searchsorted(cdf, targets, right=True).clamp(1, cdf.numel() - 1)
    lower = upper - 1
    denominator = (cdf[upper] - cdf[lower]).clamp_min(torch.finfo(torch.float64).eps)
    weight = (targets - cdf[lower]) / denominator
    result = torch.lerp(grid[lower], grid[upper], weight)
    result[0] = 0.0
    result[-1] = 1.0
    return result


def _beta_base_sigmas(model_sampling: Any, progress: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    if alpha <= 0.0 or beta <= 0.0 or not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("beta scheduler parameters must be positive and finite")
    import numpy
    from scipy.special import betaincinv

    table = torch.as_tensor(model_sampling.sigmas).detach().to(device="cpu", dtype=torch.float64).flatten()
    if table.numel() < 2 or not torch.isfinite(table).all() or torch.any(table[1:] < table[:-1]):
        raise ValueError("model sampling sigma table must be finite and nondecreasing")
    probability = 1.0 - progress.detach().to(device="cpu", dtype=torch.float64).numpy()
    quantile = numpy.asarray(betaincinv(alpha, beta, numpy.clip(probability, 0.0, 1.0)), dtype=numpy.float64)
    positions = torch.from_numpy(quantile.copy()) * (table.numel() - 1)
    lower = positions.floor().long().clamp(0, table.numel() - 1)
    upper = positions.ceil().long().clamp(0, table.numel() - 1)
    return torch.lerp(table[lower], table[upper], positions - lower)


def calibrated_beta_sigmas(model_sampling: Any, steps: int, profile: CalibrationProfile, integration_points: int = 4097) -> torch.Tensor:
    """Weight ComfyUI's beta(0.6, 0.6) prior by profile difficulty.

    Uniform profile difficulty is the continuous-table counterpart of the
    stock beta scheduler. Nonuniform difficulty redistributes its step
    fractions without requiring the reference model at inference time.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    profile.validate()
    base = profile.metadata.get("base_scheduler", {})
    if not isinstance(base, dict) or base.get("name", "beta") != "beta":
        raise ValueError("calibration profile must declare a beta base scheduler")
    alpha = float(base.get("alpha", 0.6))
    beta = float(base.get("beta", 0.6))
    grid = torch.linspace(0.0, 1.0, integration_points, dtype=torch.float64)
    base_sigmas = _beta_base_sigmas(model_sampling, grid, alpha, beta)
    sigma_max = float(model_sampling.sigma_max)
    sigma_min = float(model_sampling.sigma_min)
    sigma_progress = ((sigma_max - base_sigmas) / (sigma_max - sigma_min)).clamp(0.0, 1.0)
    density = _interpolate_density(profile, sigma_progress)
    increments = 0.5 * (density[1:] + density[:-1]) * (grid[1:] - grid[:-1])
    cdf = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(increments, dim=0)))
    cdf = cdf / cdf[-1]
    targets = torch.arange(steps, dtype=torch.float64) / steps
    upper = torch.searchsorted(cdf, targets, right=True).clamp(1, cdf.numel() - 1)
    lower = upper - 1
    denominator = (cdf[upper] - cdf[lower]).clamp_min(torch.finfo(torch.float64).eps)
    selected_progress = torch.lerp(grid[lower], grid[upper], (targets - cdf[lower]) / denominator)
    return _beta_base_sigmas(model_sampling, selected_progress, alpha, beta)


def sigmas_from_profile(model_sampling: Any, steps: int, denoise: float, profile: CalibrationProfile) -> torch.Tensor:
    if not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be in [0, 1]")
    if denoise == 0.0:
        return torch.empty(0, dtype=torch.float32)
    if steps < 1:
        raise ValueError("steps must be positive")
    if not hasattr(model_sampling, "audio_shift"):
        raise ValueError("MiniMax H3 RefDelta scheduler requires ModelSamplingAV")
    sigma_max = float(model_sampling.sigma_max)
    sigma_min = float(model_sampling.sigma_min)
    if not math.isfinite(sigma_max) or not math.isfinite(sigma_min) or sigma_max <= sigma_min or sigma_min <= 0.0:
        raise ValueError("model sampling must expose a positive finite sigma range")
    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    total_steps = max(total_steps, steps)
    nonterminal = calibrated_beta_sigmas(model_sampling, total_steps, profile)
    sigmas = torch.cat((nonterminal, torch.zeros(1, dtype=torch.float64)))
    sigmas = sigmas[-(steps + 1) :].to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(sigmas).all():
        raise ValueError("calibrated scheduler produced NaN or Inf")
    if torch.any(sigmas[1:] > sigmas[:-1]):
        raise ValueError("calibrated scheduler produced non-monotonic sigmas")
    return sigmas


__all__ = [
    "CalibrationProfile",
    "calibrated_beta_sigmas",
    "calibrated_progress",
    "load_profile",
    "profile_from_dict",
    "sigmas_from_profile",
    "uniform_fallback_profile",
]
