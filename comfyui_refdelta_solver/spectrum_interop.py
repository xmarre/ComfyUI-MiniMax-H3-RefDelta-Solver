from __future__ import annotations

from typing import Any

import torch


SPECTRUM_BRIDGE_KEY = "spectrum_h3_refdelta_bridge"
SPECTRUM_INTEROP_CONTRACT = (
    "comfyui-refdelta-spectrum",
    1,
    "actual-anchor-history",
    "exact-gated-stochastic-increment",
)

SPECTRUM_BACKEND_INTEROP_CONTRACT = (
    "comfyui-refdelta-spectrum-backend",
    1,
    "actual-anchor-history",
    "native-solver-noise-geometry",
)


class SpectrumInteropError(RuntimeError):
    """Spectrum exposed a RefDelta bridge that does not satisfy API v1."""


def spectrum_bridge(extra_args: dict[str, Any]) -> Any | None:
    """Return Spectrum's structural bridge without introducing a package dependency."""
    transformer_options = (extra_args.get("model_options") or {}).get(
        "transformer_options"
    ) or {}
    bridge = transformer_options.get(SPECTRUM_BRIDGE_KEY)
    if bridge is None:
        return None
    if (
        getattr(bridge, "api_version", None) != 1
        or getattr(bridge, "interop_contract", None) != SPECTRUM_INTEROP_CONTRACT
        or not callable(getattr(bridge, "model_result_is_actual", None))
        or not callable(getattr(bridge, "publish_stochastic_increment", None))
    ):
        raise SpectrumInteropError("invalid Spectrum RefDelta bridge API")
    return bridge


def spectrum_backend_bridge(extra_args: dict[str, Any]) -> Any | None:
    """Return Spectrum's generic RefDelta backend bridge.

    ER-SDE keeps its v1 exact-increment contract. Multi-stage SEEDS and
    multistep SA-Solver use a narrower classification contract because their
    native stochastic geometry must remain owned by the base solver.
    """
    transformer_options = (extra_args.get("model_options") or {}).get(
        "transformer_options"
    ) or {}
    bridge = transformer_options.get(SPECTRUM_BRIDGE_KEY)
    if bridge is None:
        return None
    if (
        getattr(bridge, "api_version", None) != 1
        or getattr(bridge, "interop_contract", None)
        != SPECTRUM_BACKEND_INTEROP_CONTRACT
        or not callable(getattr(bridge, "model_result_is_actual", None))
    ):
        raise SpectrumInteropError("invalid Spectrum RefDelta backend bridge API")
    return bridge


def model_result_is_actual(bridge: Any | None, step_id: int) -> bool:
    if bridge is None:
        return True
    result = bridge.model_result_is_actual(step_id)
    if type(result) is not bool:
        raise SpectrumInteropError(
            "Spectrum RefDelta bridge returned a non-boolean model-result classification"
        )
    return result


def publish_stochastic_increment(
    bridge: Any | None,
    step_id: int,
    increment: torch.Tensor,
) -> None:
    if bridge is None:
        return
    bridge.publish_stochastic_increment(step_id, increment)


__all__ = [
    "SPECTRUM_BACKEND_INTEROP_CONTRACT",
    "SPECTRUM_BRIDGE_KEY",
    "SPECTRUM_INTEROP_CONTRACT",
    "SpectrumInteropError",
    "model_result_is_actual",
    "publish_stochastic_increment",
    "spectrum_backend_bridge",
    "spectrum_bridge",
]
