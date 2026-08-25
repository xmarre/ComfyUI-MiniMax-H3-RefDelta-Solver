from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from .config import RefDeltaSamplerConfig
from .coordinates import smoothstep
from .trajectory import StreamLayout, TrajectoryObservation, stochastic_multiplier


def _channel_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(
        value,
        dim=1,
        keepdim=True,
        dtype=torch.float32,
    ) / math.sqrt(value.shape[1])


def _relative_change_ratio(change: torch.Tensor, signal: torch.Tensor, floor: float) -> torch.Tensor:
    invalid = (~torch.isfinite(change)).any(dim=1, keepdim=True) | (
        ~torch.isfinite(signal)
    ).any(dim=1, keepdim=True)
    local_signal = _channel_rms(signal)
    global_signal = torch.linalg.vector_norm(
        signal,
        dim=(1, 2, 3, 4),
        keepdim=True,
        dtype=torch.float32,
    ) / math.sqrt(math.prod(signal.shape[1:]))
    epsilon = torch.finfo(local_signal.dtype).eps
    ratio = _channel_rms(change) / (local_signal + floor * global_signal + epsilon)
    ratio = torch.nan_to_num(ratio, nan=2.0, posinf=2.0, neginf=2.0)
    return ratio.masked_fill(invalid, 2.0).clamp(0.0, 2.0)


def temporal_motion_ratio(video: torch.Tensor, normalization_floor: float) -> torch.Tensor:
    """Return a channel-reduced relative motion map with replicated temporal edges."""
    if video.ndim != 5:
        raise ValueError("video must have shape [B, C, T, H, W]")
    frames = video.shape[2]
    if frames == 1:
        return torch.zeros(
            (video.shape[0], 1, 1, video.shape[3], video.shape[4]),
            device=video.device,
            dtype=torch.float32,
        )
    adjacent = video[:, :, 1:] - video[:, :, :-1]
    adjacent_ratio = _relative_change_ratio(
        adjacent,
        0.5 * (video[:, :, 1:] + video[:, :, :-1]),
        normalization_floor,
    )
    result = adjacent_ratio.new_empty(
        (video.shape[0], 1, frames, video.shape[3], video.shape[4])
    )
    result[:, :, 0] = adjacent_ratio[:, :, 0]
    result[:, :, -1] = adjacent_ratio[:, :, -1]
    if frames > 2:
        result[:, :, 1:-1] = 0.5 * (
            adjacent_ratio[:, :, :-1] + adjacent_ratio[:, :, 1:]
        )
    return result


def diffusion_change_ratio(
    current: torch.Tensor,
    previous: torch.Tensor,
    normalization_floor: float,
) -> torch.Tensor:
    if current.shape != previous.shape:
        raise ValueError("current and previous actual video shapes differ")
    return _relative_change_ratio(current - previous, current, normalization_floor)


def smooth_stability_map(
    value: torch.Tensor,
    spatial_radius: int,
    temporal_radius: int,
) -> torch.Tensor:
    """Separable replicate-padded mean filtering in latent T/H/W space."""
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError("stability map must have shape [B, 1, T, H, W]")
    result = value
    if temporal_radius > 0:
        result = functional.pad(
            result,
            (0, 0, 0, 0, temporal_radius, temporal_radius),
            mode="replicate",
        )
        result = functional.avg_pool3d(
            result,
            kernel_size=(2 * temporal_radius + 1, 1, 1),
            stride=1,
        )
    if spatial_radius > 0:
        result = functional.pad(
            result,
            (spatial_radius, spatial_radius, spatial_radius, spatial_radius, 0, 0),
            mode="replicate",
        )
        result = functional.avg_pool3d(
            result,
            kernel_size=(1, 2 * spatial_radius + 1, 2 * spatial_radius + 1),
            stride=1,
        )
    return result


def stability_progress_gate(
    step_index: int,
    total_steps: int,
    start_fraction: float,
    full_fraction: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    progress = step_index / max(total_steps - 1, 1)
    if progress < start_fraction:
        return reference.new_zeros(())
    if full_fraction <= start_fraction or progress >= full_fraction:
        return reference.new_ones(())
    unit = (progress - start_fraction) / (full_fraction - start_fraction)
    return reference.new_tensor(unit * unit * (3.0 - 2.0 * unit))


def tensor_distribution(value: torch.Tensor, *, active_threshold: float | None = None) -> dict[str, torch.Tensor]:
    finite = torch.nan_to_num(value.detach().float())
    flattened = finite.reshape(-1)
    quantiles = torch.quantile(
        flattened,
        flattened.new_tensor((0.05, 0.50, 0.95)),
    )
    result = {
        "min": flattened.amin(),
        "mean": flattened.mean(),
        "max": flattened.amax(),
        "std": flattened.std(unbiased=False),
        "p05": quantiles[0],
        "p50": quantiles[1],
        "p95": quantiles[2],
    }
    if active_threshold is not None:
        result["active_fraction"] = (flattened > active_threshold).float().mean()
    return result


@dataclass(slots=True)
class StabilityEvidence:
    temporal_motion_ratio: torch.Tensor
    diffusion_change_ratio: torch.Tensor | None
    ema_restore_mask: torch.Tensor
    source_actual_step: int


@dataclass(slots=True)
class StochasticControlResult:
    increment: torch.Tensor
    legacy_target_gate: torch.Tensor | None
    video_dynamic_target_gate: torch.Tensor | None
    video_static_target_gate: torch.Tensor | None
    audio_target_gate: torch.Tensor | None
    video_applied_gate: torch.Tensor | None
    audio_applied_gate: torch.Tensor | None
    restore_mask: torch.Tensor | None
    temporal_motion_ratio: torch.Tensor | None
    diffusion_change_ratio: torch.Tensor | None
    progress_gate: torch.Tensor | None
    source_actual_step: int | None
    stability_updated_this_step: bool
    slew_applied: torch.Tensor
    slew_fraction: torch.Tensor

    @property
    def compatibility_gate(self) -> torch.Tensor:
        if self.legacy_target_gate is not None:
            return self.video_applied_gate
        if self.video_applied_gate is not None:
            return self.video_applied_gate.float().mean()
        return self.increment.new_ones(())


class StochasticStabilityController:
    """Invocation-local stochastic control and actual-only stability evidence."""

    def __init__(
        self,
        config: RefDeltaSamplerConfig,
        layout: StreamLayout,
        multiplier=stochastic_multiplier,
    ) -> None:
        self.config = config
        self.layout = layout
        self._multiplier = multiplier
        self.evidence: StabilityEvidence | None = None
        self._shape_key: tuple[tuple[int, ...], torch.device, torch.dtype] | None = None
        self._previous_global_gate: torch.Tensor | None = None
        self._previous_video_gate: torch.Tensor | None = None
        self._previous_audio_gate: torch.Tensor | None = None

    @property
    def effective_video_strength(self) -> float:
        return min(1.0, max(0.0, self.config.stochastic_adaptation_strength * self.config.video_stochastic_strength_scale))

    @property
    def effective_audio_strength(self) -> float:
        return min(1.0, max(0.0, self.config.stochastic_adaptation_strength * self.config.audio_stochastic_strength_scale))

    def reset(self) -> None:
        self.evidence = None
        self._shape_key = None
        self._previous_global_gate = None
        self._previous_video_gate = None
        self._previous_audio_gate = None

    def _reset_incompatible_shape(self, video: torch.Tensor) -> None:
        key = (tuple(video.shape), video.device, video.dtype)
        if self._shape_key is not None and key != self._shape_key:
            self.reset()
        self._shape_key = key

    def update_actual(
        self,
        packed_denoised: torch.Tensor,
        previous_actual: torch.Tensor | None,
        step_index: int,
    ) -> None:
        if self.config.stochastic_control_mode != "spatiotemporal_stability":
            return
        streams = self.layout.split(packed_denoised)
        if "video" not in streams:
            raise ValueError("spatiotemporal stochastic control requires a packed H3 video/audio layout")
        video = self.layout.video_to_latent(streams["video"])
        self._reset_incompatible_shape(video)
        motion_ratio = temporal_motion_ratio(
            video,
            self.config.video_stability_normalization_floor,
        )
        temporal_staticness = 1.0 - smoothstep(
            motion_ratio,
            self.config.video_stability_motion_low,
            self.config.video_stability_motion_high,
        )

        change_ratio = None
        prior_video = None
        if previous_actual is not None:
            previous_streams = self.layout.split(previous_actual)
            if "video" in previous_streams:
                candidate = self.layout.video_to_latent(previous_streams["video"])
                if (
                    candidate.shape == video.shape
                    and candidate.device == video.device
                    and candidate.dtype == video.dtype
                ):
                    prior_video = candidate

        if prior_video is None:
            current_restore = torch.zeros_like(temporal_staticness)
        else:
            change_ratio = diffusion_change_ratio(
                video,
                prior_video,
                self.config.video_stability_normalization_floor,
            )
            diffusion_stability = 1.0 - smoothstep(
                change_ratio,
                self.config.video_stability_diffusion_low,
                self.config.video_stability_diffusion_high,
            )
            diffusion_factor = torch.lerp(
                torch.ones_like(diffusion_stability),
                diffusion_stability,
                self.config.video_stability_diffusion_weight,
            )
            current_restore = (
                temporal_staticness * diffusion_factor
            ).clamp(0.0, 1.0).pow(self.config.video_stability_gamma)
            current_restore = current_restore * self.config.video_stability_restore_strength
            current_restore = smooth_stability_map(
                current_restore,
                self.config.video_stability_spatial_radius,
                self.config.video_stability_temporal_radius,
            ).clamp(0.0, 1.0)

        if self.evidence is not None and self.evidence.ema_restore_mask.shape == current_restore.shape:
            beta = self.config.video_stability_ema
            ema_restore = beta * self.evidence.ema_restore_mask + (1.0 - beta) * current_restore
        else:
            ema_restore = current_restore
        self.evidence = StabilityEvidence(
            temporal_motion_ratio=motion_ratio.detach(),
            diffusion_change_ratio=None if change_ratio is None else change_ratio.detach(),
            ema_restore_mask=ema_restore.detach().clamp(0.0, 1.0),
            source_actual_step=step_index,
        )

    def _stream_risk(self, observation: TrajectoryObservation, name: str) -> torch.Tensor:
        try:
            return observation.stream_risks[name]
        except KeyError as error:
            raise ValueError(
                f"{self.config.stochastic_control_mode} stochastic control requires {name} stream risk"
            ) from error

    def _slew(
        self,
        target: torch.Tensor,
        previous: torch.Tensor | None,
        collect_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        limit = self.config.stochastic_gate_slew_limit
        zero = target.new_zeros(())
        if limit <= 0.0 or previous is None or previous.shape != target.shape:
            stored = target.detach().clone() if limit > 0.0 else None
            return target, stored, zero, zero
        limited = target.clamp(previous - limit, previous + limit)
        if collect_stats:
            changed = (limited != target).float().mean()
            applied = (changed > 0).to(dtype=target.dtype)
        else:
            changed = zero
            applied = zero
        return limited, limited.detach().clone(), applied, changed

    def _restore_mask(self, step_index: int, total_steps: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        progress = stability_progress_gate(
            step_index,
            total_steps,
            self.config.video_stability_start_fraction,
            self.config.video_stability_full_fraction,
            reference,
        )
        if self.evidence is None:
            video_shape = self.layout.video_shape
            if video_shape is None:
                raise ValueError("spatiotemporal stochastic control requires the H3 video latent shape")
            mask = reference.new_zeros((reference.shape[0], 1, video_shape[1], video_shape[2], video_shape[3]))
        else:
            mask = self.evidence.ema_restore_mask * progress
        return mask, progress

    def apply(
        self,
        native_increment: torch.Tensor,
        observation: TrajectoryObservation,
        endpoint: torch.Tensor,
        step_index: int,
        total_steps: int,
        *,
        collect_stats: bool = False,
    ) -> StochasticControlResult:
        mode = self.config.stochastic_control_mode
        if mode == "legacy_global":
            target = self._multiplier(
                observation.risk,
                self.config.stochastic_adaptation_strength,
                self.config.minimum_stochastic_multiplier,
            )
            adapted, self._previous_global_gate, slew_applied, slew_fraction = self._slew(
                target,
                self._previous_global_gate,
                collect_stats,
            )
            final = 1.0 + endpoint * (adapted - 1.0)
            return StochasticControlResult(
                native_increment * final,
                target,
                None,
                None,
                None,
                final,
                final,
                None,
                None,
                None,
                None,
                None,
                False,
                slew_applied,
                slew_fraction,
            )

        streams = self.layout.split(native_increment)
        if set(streams) != {"video", "audio"}:
            raise ValueError(f"{mode} stochastic control requires packed H3 video and audio streams")
        video_risk = self._stream_risk(observation, "video")
        audio_risk = self._stream_risk(observation, "audio")
        dynamic_target = self._multiplier(
            video_risk,
            self.effective_video_strength,
            self.config.minimum_stochastic_multiplier,
        )
        audio_target = self._multiplier(
            audio_risk,
            self.effective_audio_strength,
            self.config.minimum_stochastic_multiplier,
        )
        if self.config.stochastic_adaptation_strength <= 0.0:
            static_target = dynamic_target
        else:
            static_target = self._multiplier(
                video_risk,
                self.config.static_video_stochastic_adaptation_strength,
                self.config.minimum_stochastic_multiplier,
            )

        restore_mask = None
        progress = None
        if mode == "spatiotemporal_stability":
            restore_mask, progress = self._restore_mask(step_index, total_steps, native_increment)
            if (
                self.config.video_stability_restore_strength > 0.0
                and self.config.stochastic_adaptation_strength > 0.0
                and self.config.static_video_stochastic_adaptation_strength
                != self.effective_video_strength
            ):
                video_target = torch.lerp(dynamic_target, static_target, restore_mask)
            else:
                video_target = dynamic_target
        else:
            video_target = dynamic_target

        adapted_video, self._previous_video_gate, video_slew, video_slew_fraction = self._slew(
            video_target,
            self._previous_video_gate,
            collect_stats,
        )
        adapted_audio, self._previous_audio_gate, audio_slew, audio_slew_fraction = self._slew(
            audio_target,
            self._previous_audio_gate,
            collect_stats,
        )
        final_video = 1.0 + endpoint * (adapted_video - 1.0)
        final_audio = 1.0 + endpoint * (adapted_audio - 1.0)
        gated_video = streams["video"]
        final_video = final_video.to(dtype=gated_video.dtype)
        final_audio = final_audio.to(dtype=streams["audio"].dtype)
        if final_video.ndim > 0:
            video_latent = self.layout.video_to_latent(gated_video)
            gated_video = self.layout.latent_to_video(
                video_latent * final_video,
                packed_like=gated_video,
            )
        else:
            gated_video = gated_video * final_video
        increment = self.layout.combine(gated_video, streams["audio"] * final_audio)
        source_step = None if self.evidence is None else self.evidence.source_actual_step
        return StochasticControlResult(
            increment,
            None,
            dynamic_target,
            static_target,
            audio_target,
            final_video,
            final_audio,
            restore_mask,
            None if self.evidence is None else self.evidence.temporal_motion_ratio,
            None if self.evidence is None else self.evidence.diffusion_change_ratio,
            progress,
            source_step,
            source_step == step_index,
            torch.maximum(video_slew, audio_slew),
            torch.maximum(video_slew_fraction, audio_slew_fraction),
        )


__all__ = [
    "StochasticControlResult",
    "StochasticStabilityController",
    "diffusion_change_ratio",
    "smooth_stability_map",
    "stability_progress_gate",
    "temporal_motion_ratio",
    "tensor_distribution",
]
