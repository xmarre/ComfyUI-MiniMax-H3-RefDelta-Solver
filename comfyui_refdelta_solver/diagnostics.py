from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory import StreamLayout, cosine, rms


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
    if fused_x0.shape != reference_x0.shape or fused_x0.shape != state.shape:
        raise ValueError("same-state diagnostic tensors must have identical shapes")
    if fused_x0.device != reference_x0.device or fused_x0.device != state.device:
        raise ValueError("same-state diagnostic tensors must share a device")
    if fused_x0.dtype != reference_x0.dtype or fused_x0.dtype != state.dtype:
        raise ValueError("same-state diagnostic tensors must share a dtype")

    sigma_value = sigma.flatten()[0].to(device=state.device, dtype=state.dtype).clamp_min(torch.finfo(state.dtype).eps)
    fused_velocity = (state - fused_x0) / sigma_value
    reference_velocity = (state - reference_x0) / sigma_value
    fields: dict[str, float] = {}
    for name, fused_stream in layout.split(fused_x0).items():
        reference_stream = layout.split(reference_x0)[name]
        fused_velocity_stream = layout.split(fused_velocity)[name]
        reference_velocity_stream = layout.split(reference_velocity)[name]
        x0_error = rms(fused_stream - reference_stream) / rms(reference_stream).clamp_min(torch.finfo(state.dtype).eps)
        velocity_error = rms(fused_velocity_stream - reference_velocity_stream) / rms(reference_velocity_stream).clamp_min(torch.finfo(state.dtype).eps)
        fields[f"{name}_x0_cosine"] = float(cosine(fused_stream, reference_stream).detach().cpu())
        fields[f"{name}_x0_relative_error"] = float(torch.nan_to_num(x0_error, posinf=1e9).detach().cpu())
        fields[f"{name}_velocity_cosine"] = float(cosine(fused_velocity_stream, reference_velocity_stream).detach().cpu())
        fields[f"{name}_velocity_relative_error"] = float(torch.nan_to_num(velocity_error, posinf=1e9).detach().cpu())
    return fields


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
