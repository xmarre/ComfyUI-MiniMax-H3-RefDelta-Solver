from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

from .calibration_replay import (
    sample_refdelta_comparison_replay,
    sample_refdelta_reference_replay,
)
from .config import (
    PRODUCTION_STABILITY_DEFAULTS,
    PRODUCTION_STABILITY_PRESET_ID,
    RefDeltaSamplerConfig,
    production_stability_config,
)
from .diagnostics import RefDeltaReferenceGuiderMixin
from .h3_scheduler import SCHEDULER_MODES, h3_uniform_flow_sigmas, load_flow_profile
from .sampler import sample_refdelta_er_sde
from .sampler_backends import (
    REFDELTA_BASE_SAMPLERS,
    sample_refdelta_sa_solver,
    sample_refdelta_sa_solver_pece,
    sample_refdelta_seeds_2,
    sample_refdelta_seeds_3,
)
from .sa_scheduler import SA_SCHEDULER_MODES, h3_sa_solver_sigmas
from .scheduler import load_profile, sigmas_from_profile


def _ksampler(
    config: RefDeltaSamplerConfig,
    s_noise: float,
    max_stage: int,
    base_sampler: str = "er_sde",
):
    import comfy.samplers

    config.validate()
    if base_sampler == "er_sde":
        function = sample_refdelta_er_sde
        options = {"config": config, "s_noise": s_noise, "max_stage": max_stage}
    elif base_sampler == "seeds_2":
        function = sample_refdelta_seeds_2
        options = {
            "config": config,
            "eta": 1.0,
            "s_noise": s_noise,
            "r": 0.5,
            "solver_type": "phi_1",
        }
    elif base_sampler == "seeds_3":
        function = sample_refdelta_seeds_3
        options = {
            "config": config,
            "eta": 1.0,
            "s_noise": s_noise,
            "r_1": 1.0 / 3.0,
            "r_2": 2.0 / 3.0,
        }
    elif base_sampler in {"sa_solver", "sa_solver_pece"}:
        function = (
            sample_refdelta_sa_solver_pece
            if base_sampler == "sa_solver_pece"
            else sample_refdelta_sa_solver
        )
        options = {
            "config": config,
            "s_noise": s_noise,
            "predictor_order": 3,
            "corrector_order": 4,
            "simple_order_2": False,
        }
    else:
        raise ValueError(f"unsupported RefDelta base_sampler {base_sampler!r}")
    return (comfy.samplers.KSAMPLER(function, extra_options=options),)


def _with_default(spec, default):
    value_type, options = spec
    return (value_type, {**options, "default": default})


class MiniMaxH3RefDeltaSampler:
    """Compatibility/manual sampler exposing every diagnostic and controller knob."""

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
                "calibration_capture": ("BOOLEAN", {"default": False, "advanced": True}),
                "calibration_id": ("STRING", {"default": "refdelta_calibration", "advanced": True}),
                "stochastic_control_mode": (
                    ["legacy_global", "streamwise", "spatiotemporal_stability"],
                    {"default": "legacy_global", "advanced": True},
                ),
                "video_stochastic_strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True}),
                "audio_stochastic_strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True}),
                "static_video_stochastic_adaptation_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "video_stability_restore_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "video_stability_motion_low": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 2.0, "step": 0.005, "advanced": True}),
                "video_stability_motion_high": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 2.0, "step": 0.005, "advanced": True}),
                "video_stability_diffusion_low": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 2.0, "step": 0.005, "advanced": True}),
                "video_stability_diffusion_high": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.005, "advanced": True}),
                "video_stability_diffusion_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "video_stability_normalization_floor": ("FLOAT", {"default": 0.10, "min": 0.001, "max": 1.0, "step": 0.01, "advanced": True}),
                "video_stability_gamma": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "advanced": True}),
                "video_stability_spatial_radius": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "advanced": True}),
                "video_stability_temporal_radius": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "advanced": True}),
                "video_stability_ema": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 0.99, "step": 0.05, "advanced": True}),
                "video_stability_start_fraction": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "video_stability_full_fraction": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "stochastic_gate_slew_limit": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "debug_stability_maps": ("BOOLEAN", {"default": False, "advanced": True}),
                "base_sampler": (list(REFDELTA_BASE_SAMPLERS), {"default": "er_sde"}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Advanced/manual RefDelta sampler family. Select ER-SDE, SEEDS-2, SEEDS-3, "
        "SA-Solver PEC, or SA-Solver PECE while keeping shared trajectory and "
        "stochastic-stability controls."
    )

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
        calibration_capture,
        calibration_id,
        stochastic_control_mode,
        video_stochastic_strength_scale,
        audio_stochastic_strength_scale,
        static_video_stochastic_adaptation_strength,
        video_stability_restore_strength,
        video_stability_motion_low,
        video_stability_motion_high,
        video_stability_diffusion_low,
        video_stability_diffusion_high,
        video_stability_diffusion_weight,
        video_stability_normalization_floor,
        video_stability_gamma,
        video_stability_spatial_radius,
        video_stability_temporal_radius,
        video_stability_ema,
        video_stability_start_fraction,
        video_stability_full_fraction,
        stochastic_gate_slew_limit,
        debug_stability_maps,
        base_sampler="er_sde",
    ):
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
            calibration_capture=calibration_capture,
            calibration_id=calibration_id,
            stochastic_control_mode=stochastic_control_mode,
            video_stochastic_strength_scale=video_stochastic_strength_scale,
            audio_stochastic_strength_scale=audio_stochastic_strength_scale,
            static_video_stochastic_adaptation_strength=static_video_stochastic_adaptation_strength,
            video_stability_restore_strength=video_stability_restore_strength,
            video_stability_motion_low=video_stability_motion_low,
            video_stability_motion_high=video_stability_motion_high,
            video_stability_diffusion_low=video_stability_diffusion_low,
            video_stability_diffusion_high=video_stability_diffusion_high,
            video_stability_diffusion_weight=video_stability_diffusion_weight,
            video_stability_normalization_floor=video_stability_normalization_floor,
            video_stability_gamma=video_stability_gamma,
            video_stability_spatial_radius=video_stability_spatial_radius,
            video_stability_temporal_radius=video_stability_temporal_radius,
            video_stability_ema=video_stability_ema,
            video_stability_start_fraction=video_stability_start_fraction,
            video_stability_full_fraction=video_stability_full_fraction,
            stochastic_gate_slew_limit=stochastic_gate_slew_limit,
            debug_stability_maps=debug_stability_maps,
        )
        return _ksampler(config, s_noise, max_stage, base_sampler)


class MiniMaxH3RefDeltaProductionSampler(MiniMaxH3RefDeltaSampler):
    """Recommended rank-1024 INT8 ConvRot stability preset with tunable controls."""

    PRESET_ID = PRODUCTION_STABILITY_PRESET_ID

    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(super().INPUT_TYPES())
        required = inputs["required"]
        required.pop("calibration_capture")
        required.pop("calibration_id")
        required.pop("stochastic_control_mode")
        widget_defaults = {
            "adaptive_order": PRODUCTION_STABILITY_DEFAULTS["adaptive_order"],
            "risk_sensitivity": PRODUCTION_STABILITY_DEFAULTS["risk_sensitivity"],
            "stochastic_adaptation_strength": PRODUCTION_STABILITY_DEFAULTS["stochastic_adaptation_strength"],
            "minimum_stochastic_multiplier": PRODUCTION_STABILITY_DEFAULTS["minimum_stochastic_multiplier"],
            "trajectory_correction": PRODUCTION_STABILITY_DEFAULTS["trajectory_correction"],
            "video_correction_strength": PRODUCTION_STABILITY_DEFAULTS["video_correction_strength"],
            "audio_correction_strength": PRODUCTION_STABILITY_DEFAULTS["audio_correction_strength"],
            "correction_bound": PRODUCTION_STABILITY_DEFAULTS["correction_bound"],
            "endpoint_fidelity_fraction": PRODUCTION_STABILITY_DEFAULTS["endpoint_fidelity_fraction"],
            "video_stochastic_strength_scale": PRODUCTION_STABILITY_DEFAULTS["video_stochastic_strength_scale"],
            "audio_stochastic_strength_scale": PRODUCTION_STABILITY_DEFAULTS["audio_stochastic_strength_scale"],
            "static_video_stochastic_adaptation_strength": PRODUCTION_STABILITY_DEFAULTS["static_video_stochastic_adaptation_strength"],
            "video_stability_restore_strength": PRODUCTION_STABILITY_DEFAULTS["video_stability_restore_strength"],
            "video_stability_motion_low": PRODUCTION_STABILITY_DEFAULTS["video_stability_motion_low"],
            "video_stability_motion_high": PRODUCTION_STABILITY_DEFAULTS["video_stability_motion_high"],
            "video_stability_diffusion_low": PRODUCTION_STABILITY_DEFAULTS["video_stability_diffusion_low"],
            "video_stability_diffusion_high": PRODUCTION_STABILITY_DEFAULTS["video_stability_diffusion_high"],
            "video_stability_diffusion_weight": PRODUCTION_STABILITY_DEFAULTS["video_stability_diffusion_weight"],
            "video_stability_normalization_floor": PRODUCTION_STABILITY_DEFAULTS["video_stability_normalization_floor"],
            "video_stability_gamma": PRODUCTION_STABILITY_DEFAULTS["video_stability_gamma"],
            "video_stability_spatial_radius": PRODUCTION_STABILITY_DEFAULTS["video_stability_spatial_radius"],
            "video_stability_temporal_radius": PRODUCTION_STABILITY_DEFAULTS["video_stability_temporal_radius"],
            "video_stability_ema": PRODUCTION_STABILITY_DEFAULTS["video_stability_ema"],
            "video_stability_start_fraction": PRODUCTION_STABILITY_DEFAULTS["video_stability_start_fraction"],
            "video_stability_full_fraction": PRODUCTION_STABILITY_DEFAULTS["video_stability_full_fraction"],
            "stochastic_gate_slew_limit": PRODUCTION_STABILITY_DEFAULTS["stochastic_gate_slew_limit"],
            "debug_stability_maps": PRODUCTION_STABILITY_DEFAULTS["debug_stability_maps"],
        }
        for name, default in widget_defaults.items():
            required[name] = _with_default(required[name], default)
        return inputs

    DESCRIPTION = (
        "Recommended MiniMax-H3 RefDelta sampler-family preset for the rank-1024 INT8 ConvRot "
        "checkpoint. Choose the enhanced base sampler from one dropdown while retaining the "
        "shared spatiotemporal stability controls."
    )

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
        video_stochastic_strength_scale,
        audio_stochastic_strength_scale,
        static_video_stochastic_adaptation_strength,
        video_stability_restore_strength,
        video_stability_motion_low,
        video_stability_motion_high,
        video_stability_diffusion_low,
        video_stability_diffusion_high,
        video_stability_diffusion_weight,
        video_stability_normalization_floor,
        video_stability_gamma,
        video_stability_spatial_radius,
        video_stability_temporal_radius,
        video_stability_ema,
        video_stability_start_fraction,
        video_stability_full_fraction,
        stochastic_gate_slew_limit,
        debug_stability_maps,
        base_sampler="er_sde",
    ):
        config = production_stability_config(
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
            video_stochastic_strength_scale=video_stochastic_strength_scale,
            audio_stochastic_strength_scale=audio_stochastic_strength_scale,
            static_video_stochastic_adaptation_strength=static_video_stochastic_adaptation_strength,
            video_stability_restore_strength=video_stability_restore_strength,
            video_stability_motion_low=video_stability_motion_low,
            video_stability_motion_high=video_stability_motion_high,
            video_stability_diffusion_low=video_stability_diffusion_low,
            video_stability_diffusion_high=video_stability_diffusion_high,
            video_stability_diffusion_weight=video_stability_diffusion_weight,
            video_stability_normalization_floor=video_stability_normalization_floor,
            video_stability_gamma=video_stability_gamma,
            video_stability_spatial_radius=video_stability_spatial_radius,
            video_stability_temporal_radius=video_stability_temporal_radius,
            video_stability_ema=video_stability_ema,
            video_stability_start_fraction=video_stability_start_fraction,
            video_stability_full_fraction=video_stability_full_fraction,
            stochastic_gate_slew_limit=stochastic_gate_slew_limit,
            debug_stability_maps=debug_stability_maps,
        )
        return _ksampler(config, s_noise, max_stage, base_sampler)


class MiniMaxH3RefDeltaComparisonReplaySampler:
    """Evaluate one labeled MODEL against states captured by the fused sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_id": ("STRING", {"default": "refdelta_calibration"}),
                "comparison_label": ("STRING", {"default": "fl2va"}),
                "telemetry_prefix": ("STRING", {"default": "refdelta_comparison_replay", "advanced": True}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Disk-backed diagnostic comparison sampler. Replays exact captured fused states through "
        "the currently loaded labeled MODEL and returns the captured fused final latent."
    )

    def build(self, calibration_id, comparison_label, telemetry_prefix):
        import comfy.samplers

        return (
            comfy.samplers.KSAMPLER(
                sample_refdelta_comparison_replay,
                extra_options={
                    "calibration_id": calibration_id,
                    "comparison_label": comparison_label,
                    "telemetry_prefix": telemetry_prefix,
                },
            ),
        )


class MiniMaxH3RefDeltaReferenceReplaySampler:
    """Deprecated saved-workflow alias for a Ref2VA-labeled comparison pass."""

    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_id": ("STRING", {"default": "refdelta_calibration"}),
                "telemetry_prefix": ("STRING", {"default": "refdelta_reference_replay", "advanced": True}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "Legacy Ref2VA replay alias. Use MiniMax H3 RefDelta Comparison Replay."

    def build(self, calibration_id, telemetry_prefix):
        import comfy.samplers

        return (
            comfy.samplers.KSAMPLER(
                sample_refdelta_reference_replay,
                extra_options={
                    "calibration_id": calibration_id,
                    "telemetry_prefix": telemetry_prefix,
                },
            ),
        )


class MiniMaxH3RefDeltaScheduler:
    """Saved-workflow/research scheduler retained after production returned to stock beta."""

    DEPRECATED = True
    PROFILES: ClassVar[list[str]] = ["r1024_provisional"]

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
    DESCRIPTION = (
        "Legacy/research profile-density scheduler retained for saved workflows. Production "
        "recommendation is ComfyUI BasicScheduler with beta; the bundled profile is neutral."
    )

    def get_sigmas(self, model, steps, denoise, profile):
        model_sampling = model.get_model_object("model_sampling")
        calibration = load_profile(profile, fallback=True)
        return (sigmas_from_profile(model_sampling, steps, denoise, calibration),)


class MiniMaxH3UniformFlowScheduler:
    """Experimental shared-base-time scheduler laboratory for MiniMax H3."""

    PROFILES: ClassVar[list[str]] = ["h3_uniform_neutral"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 19, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (list(SCHEDULER_MODES), {"default": "phase_offset_uniform"}),
                "phase": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 0.99, "step": 0.01, "advanced": True}),
                "power": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "advanced": True}),
                "tail_steps": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "advanced": True}),
                "tail_start": ("FLOAT", {"default": 0.15, "min": 0.001, "max": 0.999, "step": 0.005, "advanced": True}),
                "tail_power": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 4.0, "step": 0.05, "advanced": True}),
                "beta_alpha": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 5.0, "step": 0.05, "advanced": True}),
                "beta_beta": ("FLOAT", {"default": 0.60, "min": 0.05, "max": 5.0, "step": 0.05, "advanced": True}),
                "arc_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True}),
                "audio_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "advanced": True}),
                "structure_fraction": ("FLOAT", {"default": 0.50, "min": 0.05, "max": 0.95, "step": 0.05, "advanced": True}),
                "mid_power": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "advanced": True}),
                "detail_power": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "advanced": True}),
                "profile": (cls.PROFILES, {"default": "h3_uniform_neutral", "advanced": True}),
                "profile_path": ("STRING", {"default": "", "advanced": True}),
                "auto_tail_steps": ("BOOLEAN", {"default": True, "advanced": True}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = (
        "Experimental MiniMax-H3 shared-flow scheduler family for ER-SDE A/B research. "
        "The media-tested default is phase_offset_uniform at phase 0.50 / 19 steps; "
        "legacy_ddim_uniform remains the exact ComfyUI parity control."
    )

    def get_sigmas(
        self,
        model,
        steps,
        denoise,
        mode,
        phase,
        power,
        tail_steps,
        tail_start,
        tail_power,
        beta_alpha,
        beta_beta,
        arc_strength,
        audio_weight,
        structure_fraction,
        mid_power,
        detail_power,
        profile,
        profile_path,
        auto_tail_steps,
    ):
        model_sampling = model.get_model_object("model_sampling")
        calibration = (
            load_flow_profile(profile, profile_path or None)
            if mode == "curvature_profile"
            else None
        )
        sigmas = h3_uniform_flow_sigmas(
            model_sampling,
            steps,
            denoise,
            mode,
            phase=phase,
            power=power,
            tail_steps=None if auto_tail_steps else tail_steps,
            tail_start=tail_start,
            tail_power=tail_power,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            arc_strength=arc_strength,
            audio_weight=audio_weight,
            structure_fraction=structure_fraction,
            mid_power=mid_power,
            detail_power=detail_power,
            profile=calibration,
        )
        return (sigmas,)


class MiniMaxH3SASolverScheduler:
    """Dedicated bounded base-time schedule research node for MiniMax-H3 SA-Solver."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of outer SA-Solver intervals; PECE phase count is unchanged.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Matches ComfyUI BasicScheduler tail-slicing semantics.",
                    },
                ),
                "mode": (
                    list(SA_SCHEDULER_MODES),
                    {
                        "default": "simple_control",
                        "tooltip": (
                            "simple_control is exact ComfyUI simple. simple_adams_bounded permits "
                            "one solver-native local base-time transfer around the worst Adams L1 "
                            "record while enforcing a hard 0.875x..1.125x interval envelope."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"
    DESCRIPTION = (
        "Experimental MiniMax-H3 scheduler designed for native SA-Solver PEC/PECE. "
        "It changes outer point placement only: sampler equations, stochastic noise, "
        "predictor/corrector topology, callbacks, and NFE count remain native."
    )

    def get_sigmas(self, model, steps, denoise, mode):
        model_sampling = model.get_model_object("model_sampling")
        return (h3_sa_solver_sigmas(model_sampling, steps, denoise, mode),)


class MiniMaxH3RefDeltaReferenceGuider:
    """Deprecated simultaneous dual-model diagnostic retained for saved workflows."""

    DEPRECATED = True

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
    DESCRIPTION = (
        "Legacy simultaneous dual-model diagnostic. It may require both full H3 models resident; "
        "prefer diagnostic capture + Comparison Replay."
    )

    def build(self, model, reference_model, positive, negative, cfg):
        import comfy.samplers

        if model is reference_model:
            raise ValueError("RefDelta reference diagnostic requires a distinct reference MODEL")

        class RefDeltaReferenceGuider(RefDeltaReferenceGuiderMixin, comfy.samplers.CFGGuider):
            pass

        guider = RefDeltaReferenceGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.initialize_reference(reference_model, positive, negative)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RefDeltaSampler": MiniMaxH3RefDeltaSampler,
    "MiniMaxH3RefDeltaProductionSampler": MiniMaxH3RefDeltaProductionSampler,
    "MiniMaxH3RefDeltaComparisonReplaySampler": MiniMaxH3RefDeltaComparisonReplaySampler,
    "MiniMaxH3RefDeltaReferenceReplaySampler": MiniMaxH3RefDeltaReferenceReplaySampler,
    "MiniMaxH3RefDeltaScheduler": MiniMaxH3RefDeltaScheduler,
    "MiniMaxH3UniformFlowScheduler": MiniMaxH3UniformFlowScheduler,
    "MiniMaxH3SASolverScheduler": MiniMaxH3SASolverScheduler,
    "MiniMaxH3RefDeltaReferenceGuider": MiniMaxH3RefDeltaReferenceGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefDeltaSampler": "[Advanced/Diagnostic] MiniMax H3 RefDelta Sampler",
    "MiniMaxH3RefDeltaProductionSampler": "MiniMax H3 RefDelta Stability Sampler",
    "MiniMaxH3RefDeltaComparisonReplaySampler": "MiniMax H3 RefDelta Comparison Replay",
    "MiniMaxH3RefDeltaReferenceReplaySampler": "[Legacy] MiniMax H3 RefDelta Reference Replay",
    "MiniMaxH3RefDeltaScheduler": "[Legacy/Research] MiniMax H3 RefDelta Scheduler",
    "MiniMaxH3UniformFlowScheduler": "MiniMax H3 Uniform Flow Scheduler [Experimental]",
    "MiniMaxH3SASolverScheduler": "MiniMax H3 SA-Solver Scheduler [Experimental]",
    "MiniMaxH3RefDeltaReferenceGuider": "[Legacy] MiniMax H3 RefDelta Reference Guider",
}
