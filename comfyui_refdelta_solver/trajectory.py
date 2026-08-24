from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .coordinates import divided_difference, second_divided_difference, smoothstep


def rms(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    return torch.linalg.vector_norm(value.reshape(-1)) / math.sqrt(value.numel())


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    denominator = left_norm * right_norm
    eps = torch.finfo(left.dtype).eps if left.dtype.is_floating_point else 1e-12
    both_zero = (left_norm <= eps) & (right_norm <= eps)
    defined = torch.dot(left_flat, right_flat) / denominator.clamp_min(eps)
    return torch.where(denominator > eps, defined, both_zero.to(dtype=defined.dtype)).clamp(-1.0, 1.0)


def bounded_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(numerator.dtype).eps
    ratio = numerator / denominator.clamp_min(eps)
    return torch.nan_to_num(ratio / (1.0 + ratio), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class StreamLayout:
    video_elements: int | None

    def split(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.video_elements is None:
            return {"combined": value}
        if value.shape[-1] <= self.video_elements:
            raise ValueError("packed H3 latent is shorter than its declared video stream")
        return {
            "video": value[..., : self.video_elements],
            "audio": value[..., self.video_elements :],
        }


@dataclass(slots=True)
class TrajectoryObservation:
    first: torch.Tensor | None
    second: torch.Tensor | None
    first_direction_cosine: torch.Tensor | None
    movement_rms: torch.Tensor
    risk: torch.Tensor
    stream_risks: dict[str, torch.Tensor]
    components: dict[str, dict[str, torch.Tensor]]


class TrajectoryHistory:
    """Per-invocation raw model history. Corrected values never enter this object."""

    def __init__(self) -> None:
        self.previous_raw: torch.Tensor | None = None
        self.previous_first: torch.Tensor | None = None
        self.previous_coordinate: float | None = None
        self.two_back_coordinate: float | None = None
        self.previous_stochastic_ratios: dict[str, torch.Tensor] | None = None

    def reset(self) -> None:
        self.previous_raw = None
        self.previous_first = None
        self.previous_coordinate = None
        self.two_back_coordinate = None
        self.previous_stochastic_ratios = None

    def observe(
        self,
        raw: torch.Tensor,
        coordinate: float,
        next_coordinate: float,
        layout: StreamLayout,
        stage2: torch.Tensor | None,
        stage3: torch.Tensor | None,
        stage1: torch.Tensor,
        sensitivity: float,
    ) -> TrajectoryObservation:
        zero = raw.new_zeros(())
        first = None
        second = None
        direction = None
        movement = zero
        if self.previous_raw is not None and self.previous_coordinate is not None:
            first = divided_difference(raw, self.previous_raw, coordinate, self.previous_coordinate)
            movement = rms(raw - self.previous_raw)
        if (
            first is not None
            and self.previous_first is not None
            and self.two_back_coordinate is not None
        ):
            second = second_divided_difference(first, self.previous_first, coordinate, self.two_back_coordinate)
            direction = cosine(first, self.previous_first)

        stream_risks: dict[str, torch.Tensor] = {}
        components: dict[str, dict[str, torch.Tensor]] = {}
        current_streams = layout.split(raw)
        previous_streams = layout.split(self.previous_raw) if self.previous_raw is not None else {}
        first_streams = layout.split(first) if first is not None else {}
        prior_first_streams = layout.split(self.previous_first) if self.previous_first is not None else {}
        second_streams = layout.split(second) if second is not None else {}
        stage1_streams = layout.split(stage1)
        stage2_streams = layout.split(stage2) if stage2 is not None else {}
        stage3_streams = layout.split(stage3) if stage3 is not None else {}

        local_span = abs(next_coordinate - coordinate)
        for name, current in current_streams.items():
            signals: list[torch.Tensor] = []
            values: dict[str, torch.Tensor] = {}
            if name in first_streams:
                first_norm = rms(first_streams[name])
                values["first_derivative_rms"] = first_norm
                if name in second_streams:
                    curvature = bounded_ratio(rms(second_streams[name]) * local_span, first_norm)
                    values["curvature"] = curvature
                    signals.append(curvature)
                if name in prior_first_streams:
                    direction_change = (1.0 - cosine(first_streams[name], prior_first_streams[name])) * 0.5
                    values["direction_change"] = direction_change
                    signals.append(direction_change)
                    prior_norm = rms(prior_first_streams[name])
                    eps = torch.finfo(first_norm.dtype).eps
                    magnitude_jump = torch.tanh(torch.abs(torch.log((first_norm + eps) / (prior_norm + eps))))
                    values["magnitude_jump"] = magnitude_jump
                    signals.append(magnitude_jump)
                if name in previous_streams and name in prior_first_streams and self.previous_coordinate is not None:
                    predicted = previous_streams[name] + (coordinate - self.previous_coordinate) * prior_first_streams[name]
                    extrapolation_error = bounded_ratio(rms(current - predicted), rms(current - previous_streams[name]))
                    values["extrapolation_error"] = extrapolation_error
                    signals.append(extrapolation_error)

            stage_reference = rms(stage1_streams[name])
            stage_ratio = zero
            if name in stage2_streams:
                stage_ratio = torch.maximum(stage_ratio, bounded_ratio(rms(stage2_streams[name]), stage_reference))
            if name in stage3_streams:
                stage_ratio = torch.maximum(stage_ratio, bounded_ratio(rms(stage3_streams[name]), stage_reference))
            values["stage_correction_ratio"] = stage_ratio
            if stage2 is not None or stage3 is not None:
                signals.append(stage_ratio)
            if self.previous_stochastic_ratios is not None and name in self.previous_stochastic_ratios:
                noise_signal = torch.nan_to_num(self.previous_stochastic_ratios[name], nan=0.0, posinf=1.0).clamp(0.0, 1.0)
                values["previous_stochastic_ratio"] = noise_signal
                signals.append(noise_signal)

            risk = torch.stack(signals).mean() if signals else zero
            risk = (risk * sensitivity).clamp(0.0, 1.0)
            stream_risks[name] = risk
            components[name] = values

        combined = torch.stack(list(stream_risks.values())).amax() if stream_risks else zero
        return TrajectoryObservation(first, second, direction, movement, combined, stream_risks, components)

    def commit(self, raw: torch.Tensor, coordinate: float, first: torch.Tensor | None) -> None:
        if self.previous_raw is not None and (
            self.previous_raw.shape != raw.shape
            or self.previous_raw.device != raw.device
            or self.previous_raw.dtype != raw.dtype
        ):
            self.reset()
        prior_coordinate = self.previous_coordinate
        self.previous_raw = raw.detach().clone(memory_format=torch.contiguous_format)
        self.previous_first = None if first is None else first.detach().clone(memory_format=torch.contiguous_format)
        self.two_back_coordinate = prior_coordinate
        self.previous_coordinate = float(coordinate)


def adaptive_order_gates(risk: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        one = risk.new_ones(())
        return one, one
    stage3 = 1.0 - smoothstep(risk, 0.20, 0.55)
    stage2 = 1.0 - smoothstep(risk, 0.50, 0.85)
    return stage2, stage3


def stochastic_multiplier(risk: torch.Tensor, strength: float, minimum: float) -> torch.Tensor:
    if strength <= 0.0:
        return risk.new_ones(())
    reduction = strength * smoothstep(risk, 0.20, 0.85)
    return (1.0 - reduction).clamp(minimum, 1.0)


def bounded_trajectory_correction(
    raw: torch.Tensor,
    previous_raw: torch.Tensor | None,
    first: torch.Tensor | None,
    second: torch.Tensor | None,
    next_span: float,
    layout: StreamLayout,
    video_strength: float,
    audio_strength: float,
    bound: float,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if previous_raw is None or first is None or bound <= 0.0:
        return raw, {name: raw.new_zeros(()) for name in layout.split(raw)}
    proposal = next_span * first
    if second is not None:
        proposal = proposal + 0.5 * (next_span ** 2) * second
    proposal = torch.nan_to_num(proposal)
    corrected = raw.clone()
    correction_norms: dict[str, torch.Tensor] = {}
    raw_streams = layout.split(raw)
    previous_streams = layout.split(previous_raw)
    proposal_streams = layout.split(proposal)
    corrected_streams = layout.split(corrected)
    for name in raw_streams:
        strength = audio_strength if name == "audio" else video_strength
        candidate = proposal_streams[name] * strength * gate
        candidate_norm = rms(candidate)
        allowed = rms(raw_streams[name] - previous_streams[name]) * bound
        eps = torch.finfo(candidate_norm.dtype).eps
        scale = torch.minimum(candidate_norm.new_ones(()), allowed / candidate_norm.clamp_min(eps))
        candidate = candidate * torch.nan_to_num(scale, nan=0.0, posinf=0.0)
        corrected_streams[name].add_(candidate)
        correction_norms[name] = rms(candidate)
    return corrected, correction_norms
