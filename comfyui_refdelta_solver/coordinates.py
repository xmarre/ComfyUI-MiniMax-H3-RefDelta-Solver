from __future__ import annotations

import math

import torch


def effective_audio_sigma(sigma_video: torch.Tensor, video_shift: float, audio_shift: float) -> torch.Tensor:
    """Map H3's carried video sigma to the physical audio sigma."""
    base = sigma_video / (video_shift + sigma_video * (1.0 - video_shift))
    return audio_shift * base / (1.0 + (audio_shift - 1.0) * base)


def divided_difference(current: torch.Tensor, previous: torch.Tensor, current_coordinate: float, previous_coordinate: float) -> torch.Tensor | None:
    span = current_coordinate - previous_coordinate
    scale = max(abs(current_coordinate), abs(previous_coordinate), 1.0)
    if not math.isfinite(span) or abs(span) <= 1e-12 * scale:
        return None
    return torch.nan_to_num((current - previous) / span)


def second_divided_difference(
    current_first: torch.Tensor,
    previous_first: torch.Tensor,
    current_coordinate: float,
    two_back_coordinate: float,
) -> torch.Tensor | None:
    span = current_coordinate - two_back_coordinate
    scale = max(abs(current_coordinate), abs(two_back_coordinate), 1.0)
    if not math.isfinite(span) or abs(span) <= 1e-12 * scale:
        return None
    return torch.nan_to_num(2.0 * (current_first - previous_first) / span)


def smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if high <= low:
        raise ValueError("smoothstep requires high > low")
    unit = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def endpoint_gate(step_index: int, step_count: int, fraction: float, reference: torch.Tensor) -> torch.Tensor:
    if fraction <= 0.0 or step_count <= 1:
        return reference.new_ones(())
    remaining = max(0.0, (step_count - 1 - step_index) / max(step_count - 1, 1))
    unit = min(1.0, remaining / fraction)
    return reference.new_tensor(unit * unit * (3.0 - 2.0 * unit))
