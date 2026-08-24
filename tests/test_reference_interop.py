from __future__ import annotations

import copy
import sys
from types import ModuleType

import pytest

from comfyui_refdelta_solver.reference_interop import (
    REFERENCE_DIAGNOSTIC_CONTRACT,
    REFERENCE_DIAGNOSTIC_MODEL_OPTION,
    ReferenceDiagnosticSpec,
    RuntimeReferenceGuiderMixin,
    attach_reference_diagnostic,
)


class FakeModel:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


def test_reference_spec_preserves_model_identity_across_deepcopy_and_replacement():
    reference = FakeModel()
    spec = ReferenceDiagnosticSpec(reference)

    assert copy.deepcopy(spec) is spec
    assert copy.deepcopy(spec).reference_model is reference
    assert spec.contract == REFERENCE_DIAGNOSTIC_CONTRACT
    assert spec.guider_mixin is RuntimeReferenceGuiderMixin

    replacement_model = FakeModel()
    replacement = spec.with_reference_model(replacement_model)
    assert replacement is not spec
    assert replacement.reference_model is replacement_model
    assert replacement.contract == spec.contract
    assert replacement.guider_mixin is spec.guider_mixin


def test_attach_reference_diagnostic_is_copy_on_write():
    fused = FakeModel({"existing": "kept"})
    reference = FakeModel({"reference": True})

    decorated = attach_reference_diagnostic(fused, reference)

    assert decorated is not fused
    assert fused.model_options == {"existing": "kept"}
    assert decorated.model_options["existing"] == "kept"
    spec = decorated.model_options[REFERENCE_DIAGNOSTIC_MODEL_OPTION]
    assert isinstance(spec, ReferenceDiagnosticSpec)
    assert spec.reference_model is reference
    assert REFERENCE_DIAGNOSTIC_MODEL_OPTION not in reference.model_options


def test_attach_reference_diagnostic_rejects_recursive_reference_contract():
    fused = FakeModel()
    reference = FakeModel(
        {REFERENCE_DIAGNOSTIC_MODEL_OPTION: ReferenceDiagnosticSpec(FakeModel())}
    )
    with pytest.raises(ValueError, match="already carrying"):
        attach_reference_diagnostic(fused, reference)


def test_runtime_reference_mixin_accepts_positive_only_conditioning(monkeypatch):
    sampler_helpers = ModuleType("comfy.sampler_helpers")
    converted = []

    def convert_cond(value):
        converted.append(value)
        return [f"converted:{value}"]

    sampler_helpers.convert_cond = convert_cond
    comfy = sys.modules.get("comfy") or ModuleType("comfy")
    comfy.sampler_helpers = sampler_helpers
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", sampler_helpers)

    class Diagnostic(RuntimeReferenceGuiderMixin):
        pass

    reference = FakeModel({"foo": "bar"})
    guider = Diagnostic()
    guider.initialize_reference(reference, "positive", None)

    assert converted == ["positive"]
    assert guider.reference_model_patcher is reference
    assert guider.reference_model_options == {"foo": "bar"}
    assert guider.reference_original_conds == {"positive": ["converted:positive"]}
    assert guider._refdelta_reference_result is None
    assert guider._refdelta_reference_call_index == 0
