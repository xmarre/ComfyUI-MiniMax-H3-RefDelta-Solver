from __future__ import annotations

from dataclasses import dataclass

import torch

from .comparison import compare_fused_to_model
from .trajectory import StreamLayout


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    call_index: int
    sigma: torch.Tensor
    denoised: torch.Tensor


def compare_same_state(
    state: torch.Tensor,
    sigma: torch.Tensor,
    fused_x0: torch.Tensor,
    reference_x0: torch.Tensor,
    layout: StreamLayout,
) -> dict[str, float]:
    """Legacy name for saved-workflow diagnostics.

    New replay telemetry labels the compared model explicitly and calls
    :func:`compare_fused_to_model` directly.
    """
    return compare_fused_to_model(state, sigma, fused_x0, reference_x0, layout)


def spectrum_step_is_forecast(model_options: dict | None) -> bool:
    """Return whether Spectrum explicitly classified the current guider call as forecast.

    Spectrum's PREDICT_NOISE wrapper still executes the guider on forecast steps so
    downstream wrappers can preserve normal call ordering. The fused transformer may
    be skipped inside that call. Running the genuine Ref2VA diagnostic there would
    compare a Spectrum forecast against a real reference-model evaluation and pollute
    the same-state calibration dataset.
    """
    transformer_options = (model_options or {}).get("transformer_options")
    if not isinstance(transformer_options, dict):
        return False
    actual = transformer_options.get("spectrum_h3_actual")
    return type(actual) is bool and not actual


class RefDeltaReferenceGuiderMixin:
    """Mixin installed on ComfyUI's CFGGuider by the diagnostic node.

    Both models receive independently processed forms of the same original
    CONDITIONING objects. The reference model is evaluated after the fused model
    on the exact state tensor and sigma passed into the guider.
    """

    def initialize_reference(self, reference_model, positive, negative) -> None:
        import comfy.sampler_helpers

        self.reference_model_patcher = reference_model
        self.reference_model_options = reference_model.model_options
        self.reference_original_conds = {
            "positive": comfy.sampler_helpers.convert_cond(positive),
            "negative": comfy.sampler_helpers.convert_cond(negative),
        }
        self._refdelta_reference_result: ReferenceResult | None = None
        self._refdelta_reference_call_index = 0

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        import comfy.samplers

        model_options = {} if model_options is None else model_options
        self._refdelta_reference_result = None
        fused = super().predict_noise(x, timestep, model_options=model_options, seed=seed)
        if spectrum_step_is_forecast(model_options):
            return fused
        reference = comfy.samplers.sampling_function(
            self.reference_inner_model,
            x,
            timestep,
            self.reference_conds.get("negative"),
            self.reference_conds.get("positive"),
            self.cfg,
            model_options=self.reference_runtime_model_options,
            seed=seed,
        )
        self._refdelta_reference_result = ReferenceResult(
            call_index=self._refdelta_reference_call_index,
            sigma=timestep.detach().clone(),
            denoised=reference.detach(),
        )
        self._refdelta_reference_call_index += 1
        return fused

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        import comfy.hooks
        import comfy.model_patcher
        import comfy.sampler_helpers
        import comfy.samplers

        if self.reference_model_patcher.load_device != device:
            raise ValueError("RefDelta reference diagnostic requires both H3 models on the same load device")

        reference_conds = {key: [item.copy() for item in values] for key, values in self.reference_original_conds.items()}
        comfy.samplers.preprocess_conds_hooks(reference_conds)
        reference_options = comfy.model_patcher.create_model_options_clone(self.reference_model_options)
        reference_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        original_hook_mode = self.reference_model_patcher.hook_mode
        reference_loaded = []
        prepared = False
        try:
            if comfy.samplers.get_total_hook_groups_in_conds(reference_conds) <= 1:
                self.reference_model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.reference_model_patcher, reference_conds, reference_options)
            comfy.samplers.filter_registered_hooks_on_conds(reference_conds, reference_options)
            self.reference_inner_model, reference_conds, reference_loaded = comfy.sampler_helpers.prepare_sampling(
                self.reference_model_patcher,
                noise.shape,
                reference_conds,
                reference_options,
            )
            prepared = True
            self.reference_conds = reference_conds
            self.reference_inner_model.latent_shapes = latent_shapes
            reference_latent = latent_image
            if reference_latent is not None and torch.count_nonzero(reference_latent) > 0:
                reference_latent = self.reference_inner_model.process_latent_in(reference_latent)
            self.reference_conds = comfy.samplers.process_conds(
                self.reference_inner_model,
                noise,
                reference_conds,
                device,
                reference_latent,
                denoise_mask,
                seed,
                latent_shapes=latent_shapes,
            )
            self.reference_runtime_model_options = reference_options
            comfy.samplers.cast_to_load_options(reference_options, device=device, dtype=self.reference_model_patcher.model_dtype())
            self.reference_model_patcher.pre_run()
            self._refdelta_reference_call_index = 0
            return super().inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            self._refdelta_reference_result = None
            if prepared:
                self.reference_model_patcher.cleanup()
                comfy.sampler_helpers.cleanup_models(self.reference_conds, reference_loaded)
            comfy.samplers.cast_to_load_options(reference_options, device=self.reference_model_patcher.offload_device)
            self.reference_model_patcher.hook_mode = original_hook_mode
            self.reference_model_patcher.restore_hook_patches()
            for attribute in ("reference_inner_model", "reference_conds", "reference_runtime_model_options"):
                if hasattr(self, attribute):
                    delattr(self, attribute)


def consume_reference_result(model, call_index: int | None, sigma: torch.Tensor) -> torch.Tensor | None:
    guider = getattr(model, "inner_model", None)
    result = getattr(guider, "_refdelta_reference_result", None)
    if result is None:
        return None
    guider._refdelta_reference_result = None
    if call_index is not None and result.call_index != call_index:
        raise RuntimeError(f"stale RefDelta reference result belongs to call {result.call_index}, current call is {call_index}")
    if result.sigma.shape != sigma.shape or not torch.allclose(result.sigma, sigma, rtol=1e-6, atol=1e-7):
        raise RuntimeError("RefDelta reference result belongs to a different sigma")
    return result.denoised
