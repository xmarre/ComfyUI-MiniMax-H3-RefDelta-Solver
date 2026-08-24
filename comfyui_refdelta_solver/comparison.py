from __future__ import annotations

from typing import Any

import torch

from .trajectory import StreamLayout, cosine, rms


def _finite_scalar(value: torch.Tensor) -> float:
    return float(
        torch.nan_to_num(value.float(), nan=0.0, posinf=1e9, neginf=-1e9)
        .detach()
        .cpu()
    )


def _validate_same_state_tensors(*values: torch.Tensor) -> None:
    first = values[0]
    if any(value.shape != first.shape for value in values[1:]):
        raise ValueError("same-state comparison tensors must have identical shapes")
    if any(value.device != first.device for value in values[1:]):
        raise ValueError("same-state comparison tensors must share a device")
    if any(value.dtype != first.dtype for value in values[1:]):
        raise ValueError("same-state comparison tensors must share a dtype")


def compare_fused_to_model(
    state: torch.Tensor,
    sigma: torch.Tensor,
    fused_x0: torch.Tensor,
    comparison_x0: torch.Tensor,
    layout: StreamLayout,
) -> dict[str, float]:
    """Compare fused and labeled-model outputs at one identical state and sigma."""
    _validate_same_state_tensors(state, fused_x0, comparison_x0)
    sigma_value = sigma.flatten()[0].to(device=state.device, dtype=state.dtype)
    sigma_value = sigma_value.clamp_min(torch.finfo(state.dtype).eps)
    fused_velocity = (state - fused_x0) / sigma_value
    comparison_velocity = (state - comparison_x0) / sigma_value
    comparison_streams = layout.split(comparison_x0)
    fused_velocity_streams = layout.split(fused_velocity)
    comparison_velocity_streams = layout.split(comparison_velocity)
    fields: dict[str, float] = {}
    for name, fused_stream in layout.split(fused_x0).items():
        comparison_stream = comparison_streams[name]
        fused_velocity_stream = fused_velocity_streams[name]
        comparison_velocity_stream = comparison_velocity_streams[name]
        eps = torch.finfo(state.dtype).eps
        x0_relative = rms(fused_stream - comparison_stream) / rms(comparison_stream).clamp_min(eps)
        velocity_relative = rms(fused_velocity_stream - comparison_velocity_stream) / rms(
            comparison_velocity_stream
        ).clamp_min(eps)
        fields[f"{name}_x0_cosine"] = float(cosine(fused_stream, comparison_stream).detach().cpu())
        fields[f"{name}_x0_relative_error"] = float(
            torch.nan_to_num(x0_relative, posinf=1e9).detach().cpu()
        )
        fields[f"{name}_velocity_cosine"] = float(
            cosine(fused_velocity_stream, comparison_velocity_stream).detach().cpu()
        )
        fields[f"{name}_velocity_relative_error"] = float(
            torch.nan_to_num(velocity_relative, posinf=1e9).detach().cpu()
        )
    return fields


def compare_ref_delta(
    fused_x0: torch.Tensor,
    fl2va_x0: torch.Tensor,
    ref2va_x0: torch.Tensor,
    layout: StreamLayout,
) -> dict[str, Any]:
    """Decompose fused-from-FL2VA against the full Ref2VA-from-FL2VA delta.

    Directional and normalized metrics are undefined when the true Ref2VA delta
    is exactly/subnormally zero. They are emitted as ``None`` with an explicit
    ``*_defined`` flag rather than manufacturing finite-looking evidence.
    """
    _validate_same_state_tensors(fused_x0, fl2va_x0, ref2va_x0)
    fields: dict[str, Any] = {}
    fl2va_streams = layout.split(fl2va_x0)
    ref2va_streams = layout.split(ref2va_x0)
    for name, fused_stream in layout.split(fused_x0).items():
        true_delta = ref2va_streams[name] - fl2va_streams[name]
        fused_delta = fused_stream - fl2va_streams[name]
        residual = fused_delta - true_delta

        # Accumulate normalized diagnostics in float32 for BF16 stability while
        # retaining the source dtype's ``tiny`` threshold as the definition of
        # absent evidence.
        true_flat = true_delta.reshape(-1).float()
        fused_flat = fused_delta.reshape(-1).float()
        true_norm = torch.linalg.vector_norm(true_flat)
        fused_norm = torch.linalg.vector_norm(fused_flat)
        tiny = float(torch.finfo(fused_x0.dtype).tiny)
        defined = bool(torch.isfinite(true_norm) and true_norm > tiny)

        prefix = name
        fields[f"{prefix}_defined"] = defined
        fields[f"{prefix}_true_rms"] = _finite_scalar(rms(true_delta))
        fields[f"{prefix}_fused_rms"] = _finite_scalar(rms(fused_delta))
        fields[f"{prefix}_residual_rms"] = _finite_scalar(rms(residual))
        if not defined:
            fields[f"{prefix}_cosine"] = None
            fields[f"{prefix}_magnitude_ratio"] = None
            fields[f"{prefix}_relative_residual"] = None
            fields[f"{prefix}_projection_fraction"] = None
            fields[f"{prefix}_orthogonal_residual"] = None
            fields[f"{prefix}_orthogonal_rms"] = _finite_scalar(rms(fused_delta))
            continue

        true_norm_squared = torch.dot(true_flat, true_flat)
        projection_fraction = torch.dot(fused_flat, true_flat) / true_norm_squared
        orthogonal = fused_flat - projection_fraction * true_flat
        fields[f"{prefix}_cosine"] = float(
            torch.nan_to_num(
                torch.dot(fused_flat, true_flat) / (fused_norm * true_norm).clamp_min(tiny),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(-1.0, 1.0).cpu()
        )
        fields[f"{prefix}_magnitude_ratio"] = _finite_scalar(fused_norm / true_norm)
        fields[f"{prefix}_relative_residual"] = _finite_scalar(
            torch.linalg.vector_norm(fused_flat - true_flat) / true_norm
        )
        fields[f"{prefix}_projection_fraction"] = _finite_scalar(projection_fraction)
        fields[f"{prefix}_orthogonal_residual"] = _finite_scalar(
            torch.linalg.vector_norm(orthogonal) / true_norm
        )
        fields[f"{prefix}_orthogonal_rms"] = _finite_scalar(
            torch.linalg.vector_norm(orthogonal) / max(orthogonal.numel(), 1) ** 0.5
        )
    return fields


__all__ = ["compare_fused_to_model", "compare_ref_delta"]
