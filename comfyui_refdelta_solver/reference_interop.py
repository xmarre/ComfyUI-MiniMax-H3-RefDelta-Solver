from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .diagnostics import RefDeltaReferenceGuiderMixin, ReferenceResult


REFERENCE_DIAGNOSTIC_MODEL_OPTION = "refdelta_reference_diagnostic"
REFERENCE_DIAGNOSTIC_CONTRACT = (
    "comfyui-refdelta-reference-diagnostic",
    1,
    "model-option-guider-mixin",
)


class RuntimeReferenceGuiderMixin(RefDeltaReferenceGuiderMixin):
    """Reference guider variant that supports positive-only runtime guiders.

    Continuum builds its own BasicGuider from the exact per-chunk positive
    CONDITIONING.  This mixin preserves that runtime contract instead of
    requiring an unrelated external negative-conditioning socket.
    """

    def initialize_reference(self, reference_model, positive, negative=None) -> None:
        if negative is not None:
            super().initialize_reference(reference_model, positive, negative)
            return

        import comfy.sampler_helpers

        self.reference_model_patcher = reference_model
        self.reference_model_options = reference_model.model_options
        self.reference_original_conds = {
            "positive": comfy.sampler_helpers.convert_cond(positive),
        }
        self._refdelta_reference_result: ReferenceResult | None = None
        self._refdelta_reference_call_index = 0


@dataclass(frozen=True, slots=True)
class ReferenceDiagnosticSpec:
    """Opaque model-option contract consumed by compatible runtime guiders.

    ``__deepcopy__`` deliberately preserves the reference ModelPatcher identity.
    ComfyUI clones ``model_options`` frequently; recursively copying a complete
    second H3 ModelPatcher there would be both incorrect and extremely costly.
    Continuum replaces only ``reference_model`` with its own call-local,
    chunk-configured clone before sampling.
    """

    reference_model: Any
    contract: tuple[str, int, str] = REFERENCE_DIAGNOSTIC_CONTRACT
    guider_mixin: type = RuntimeReferenceGuiderMixin

    def __deepcopy__(self, memo):
        del memo
        return self

    def with_reference_model(self, reference_model: Any) -> ReferenceDiagnosticSpec:
        return replace(self, reference_model=reference_model)


def attach_reference_diagnostic(model: Any, reference_model: Any):
    """Clone ``model`` and attach an opt-in same-state reference specification."""
    if model is reference_model:
        raise ValueError(
            "RefDelta reference diagnostic requires a distinct reference MODEL"
        )
    if not hasattr(model, "clone"):
        raise TypeError("RefDelta reference diagnostic requires a cloneable MODEL")
    if not hasattr(reference_model, "clone"):
        raise TypeError("RefDelta reference diagnostic requires a cloneable reference MODEL")
    reference_options = getattr(reference_model, "model_options", None) or {}
    if REFERENCE_DIAGNOSTIC_MODEL_OPTION in reference_options:
        raise ValueError("reference MODEL is already carrying a RefDelta reference diagnostic")

    patched = model.clone()
    model_options = dict(getattr(patched, "model_options", None) or {})
    model_options[REFERENCE_DIAGNOSTIC_MODEL_OPTION] = ReferenceDiagnosticSpec(reference_model)
    patched.model_options = model_options
    return patched


__all__ = [
    "REFERENCE_DIAGNOSTIC_CONTRACT",
    "REFERENCE_DIAGNOSTIC_MODEL_OPTION",
    "ReferenceDiagnosticSpec",
    "RuntimeReferenceGuiderMixin",
    "attach_reference_diagnostic",
]
