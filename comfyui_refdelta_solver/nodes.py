from __future__ import annotations

from .config import RefDeltaSamplerConfig
from .diagnostics import RefDeltaReferenceGuiderMixin
from .sampler import sample_refdelta_er_sde
from .scheduler import load_profile, sigmas_from_profile


class MiniMaxH3RefDeltaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adaptive_order": ("BOOLEAN", {"default": True}),
                "risk_sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "advanced": True}),
                "stochastic_adaptation_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "minimum_stochastic_multiplier": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "trajectory_correction": ("BOOLEAN", {"default": False, "advanced": True}),
                "video_correction_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "audio_correction_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "correction_bound": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True}),
                "endpoint_fidelity_fraction": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.50, "step": 0.01, "advanced": True}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "advanced": True}),
                "max_stage": ("INT", {"default": 3, "min": 1, "max": 3, "step": 1, "advanced": True}),
                "debug_telemetry": ("BOOLEAN", {"default": False, "advanced": True}),
                "telemetry_prefix": ("STRING", {"default": "refdelta_trajectory", "advanced": True}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "ER-SDE-derived sampler with nonuniform raw-anchor risk controls for MiniMax-H3 RefDelta checkpoints."

    def build(
        self,
        adaptive_order,
        risk_sensitivity,
        stochastic_adaptation_strength,
        minimum_stochastic_multiplier,
        trajectory_correction,
        video_correction_strength,
        audio_correction_strength,
        correction_bound,
        endpoint_fidelity_fraction,
        s_noise,
        max_stage,
        debug_telemetry,
        telemetry_prefix,
    ):
        import comfy.samplers

        config = RefDeltaSamplerConfig(
            adaptive_order=adaptive_order,
            risk_sensitivity=risk_sensitivity,
            stochastic_adaptation_strength=stochastic_adaptation_strength,
            minimum_stochastic_multiplier=minimum_stochastic_multiplier,
            trajectory_correction=trajectory_correction,
            video_correction_strength=video_correction_strength,
            audio_correction_strength=audio_correction_strength,
            correction_bound=correction_bound,
            endpoint_fidelity_fraction=endpoint_fidelity_fraction,
            telemetry=debug_telemetry,
            telemetry_prefix=telemetry_prefix,
        )
        config.validate()
        return (comfy.samplers.KSAMPLER(
            sample_refdelta_er_sde,
            extra_options={"config": config, "s_noise": s_noise, "max_stage": max_stage},
        ),)


class MiniMaxH3RefDeltaScheduler:
    PROFILES = ["r1024_provisional"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "profile": (cls.PROFILES, {"default": "r1024_provisional"}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = "Profile-density scheduler for MiniMax-H3 RefDelta. The bundled rank-1024 profile is provisional until calibrated from matched same-state runs."

    def get_sigmas(self, model, steps, denoise, profile):
        model_sampling = model.get_model_object("model_sampling")
        calibration = load_profile(profile, fallback=True)
        return (sigmas_from_profile(model_sampling, steps, denoise, calibration),)


class MiniMaxH3RefDeltaReferenceGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/guiders"
    DESCRIPTION = "Expensive diagnostic guider that evaluates fused and genuine Ref2VA models on the same state, sigma, prompt, and references."

    def build(self, model, reference_model, positive, negative, cfg):
        import comfy.samplers

        class RefDeltaReferenceGuider(RefDeltaReferenceGuiderMixin, comfy.samplers.CFGGuider):
            pass

        guider = RefDeltaReferenceGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.initialize_reference(reference_model, positive, negative)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RefDeltaSampler": MiniMaxH3RefDeltaSampler,
    "MiniMaxH3RefDeltaScheduler": MiniMaxH3RefDeltaScheduler,
    "MiniMaxH3RefDeltaReferenceGuider": MiniMaxH3RefDeltaReferenceGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefDeltaSampler": "MiniMax H3 RefDelta Sampler",
    "MiniMaxH3RefDeltaScheduler": "MiniMax H3 RefDelta Scheduler",
    "MiniMaxH3RefDeltaReferenceGuider": "MiniMax H3 RefDelta Reference Diagnostic",
}

