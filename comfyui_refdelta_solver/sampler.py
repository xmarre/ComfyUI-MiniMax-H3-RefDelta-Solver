from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .config import RefDeltaSamplerConfig
from .coordinates import (
    divided_difference,
    effective_audio_sigma,
    endpoint_gate,
    second_divided_difference,
)
from .diagnostics import compare_same_state, consume_reference_result
from .spectrum_interop import (
    SPECTRUM_INTEROP_CONTRACT,
    SpectrumInteropError,
    model_result_is_actual,
    publish_stochastic_increment,
    spectrum_bridge,
)
from .telemetry import TelemetryWriter
from .trajectory import (
    StreamLayout,
    TrajectoryHistory,
    adaptive_order_gates,
    bounded_trajectory_correction,
    rms,
    stochastic_multiplier,
)


def native_noise_scaler(value: torch.Tensor) -> torch.Tensor:
    return value * ((value ** 0.3).exp() + 10.0)


def _model_sampling(model):
    return model.inner_model.model_patcher.get_model_object("model_sampling")


def _stream_layout(model, x: torch.Tensor) -> StreamLayout:
    guider = getattr(model, "inner_model", None)
    base_model = getattr(guider, "inner_model", None)
    latent_shapes = getattr(base_model, "latent_shapes", None)
    if latent_shapes is None or len(latent_shapes) < 2:
        return StreamLayout(None)
    video_elements = math.prod(latent_shapes[0][1:])
    if x.shape[-1] <= video_elements:
        raise ValueError("MiniMax H3 RefDelta sampler received an invalid packed AV latent")
    return StreamLayout(video_elements)


def _validate_h3_sampling(model_sampling: Any) -> tuple[float, float]:
    video_shift = getattr(model_sampling, "shift", None)
    audio_shift = getattr(model_sampling, "audio_shift", None)
    if video_shift is None or audio_shift is None:
        raise ValueError("MiniMax H3 RefDelta sampler requires ComfyUI ModelSamplingAV with video and audio shifts")
    video_shift = float(video_shift)
    audio_shift = float(audio_shift)
    if not math.isfinite(video_shift) or not math.isfinite(audio_shift) or video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError("MiniMax H3 sampling shifts must be positive finite values")
    return video_shift, audio_shift


def _telemetry_writer(config: RefDeltaSamplerConfig, extra_args: dict[str, Any]) -> TelemetryWriter | None:
    if not config.telemetry:
        return None
    import folder_paths

    output = Path(folder_paths.get_output_directory()) / "refdelta_telemetry"
    return TelemetryWriter(output, config.telemetry_prefix, extra_args.get("seed"))


def _record_stream_norms(record: dict[str, Any], prefix: str, value: torch.Tensor, layout: StreamLayout) -> None:
    for name, stream in layout.split(value).items():
        record.setdefault(name, {})[prefix] = rms(stream)


@torch.no_grad()
def sample_refdelta_er_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_noise=1.0,
    noise_sampler=None,
    noise_scaler=None,
    max_stage=3,
    config: RefDeltaSamplerConfig | None = None,
):
    """ER-SDE-3 with raw-anchor risk gates for rank-limited H3 RefDelta fields."""
    extra_args = {} if extra_args is None else extra_args
    config = RefDeltaSamplerConfig() if config is None else config
    config.validate()
    if not isinstance(max_stage, int) or isinstance(max_stage, bool) or not 1 <= max_stage <= 3:
        raise ValueError("max_stage must be an integer in [1, 3]")

    from comfy.k_diffusion import sampling as k_sampling

    if config.is_native_equivalence_mode:
        return k_sampling.sample_er_sde(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            noise_scaler=noise_scaler,
            max_stage=max_stage,
        )

    from tqdm.auto import trange

    model_sampling = _model_sampling(model)
    video_shift, audio_shift = _validate_h3_sampling(model_sampling)
    seed = extra_args.get("seed")
    noise_sampler = k_sampling.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    noise_scaler = native_noise_scaler if noise_scaler is None else noise_scaler
    effective_s_noise = float(s_noise) * float(getattr(model_sampling, "noise_scale", 1.0))
    s_in = x.new_ones([x.shape[0]])
    layout = _stream_layout(model, x)

    num_integration_points = 200.0
    point_indices = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)
    sigmas = k_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = k_sampling.sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()
    coordinates = [float(value) for value in er_lambdas.detach().float().cpu()]

    # Solver history follows every value returned to ER-SDE, including a Spectrum
    # forecast. Evidence history accepts only genuine model evaluations, so a
    # forecast can never become a RefDelta risk or correction anchor.
    solver_history = TrajectoryHistory()
    evidence_history = TrajectoryHistory()
    last_actual_observation = None
    bridge = spectrum_bridge(extra_args)
    writer = _telemetry_writer(config, extra_args)
    total_steps = len(sigmas) - 1
    try:
        for i in trange(total_steps, disable=disable):
            x_current = x
            raw_denoised = model(x_current, sigmas[i] * s_in, **extra_args)
            result_is_actual = model_result_is_actual(bridge, i)
            reference_denoised = consume_reference_result(
                model,
                i if bridge is None else None,
                sigmas[i] * s_in,
            )
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": raw_denoised})

            record: dict[str, Any] = {
                "step": i,
                "steps": total_steps,
                "sigma": sigmas[i],
                "sigma_next": sigmas[i + 1],
                "er_lambda": er_lambdas[i],
                "er_lambda_next": er_lambdas[i + 1],
                "audio_sigma": effective_audio_sigma(sigmas[i], video_shift, audio_shift),
                "audio_sigma_next": effective_audio_sigma(sigmas[i + 1], video_shift, audio_shift),
                "sigma_min": model_sampling.sigma_min,
                "sigma_max": model_sampling.sigma_max,
            }
            if writer is not None:
                _record_stream_norms(record, "latent_rms", x, layout)
                _record_stream_norms(record, "denoised_rms", raw_denoised, layout)

            if sigmas[i + 1] == 0:
                x = raw_denoised
                if writer is not None:
                    if result_is_actual:
                        terminal_observation = evidence_history.observe(
                            raw_denoised,
                            coordinates[i],
                            coordinates[i + 1],
                            layout,
                            None,
                            None,
                            raw_denoised,
                            config.risk_sensitivity,
                        )
                    elif last_actual_observation is not None:
                        terminal_observation = last_actual_observation
                    else:
                        raise SpectrumInteropError(
                            "Spectrum forecast before RefDelta established an actual anchor"
                        )
                    record["effective_order"] = 1
                    record["risk"] = terminal_observation.risk
                    record["stream_risk"] = terminal_observation.stream_risks
                    record["risk_components"] = terminal_observation.components
                    record["denoised_difference_rms"] = terminal_observation.movement_rms
                    record["first_derivative_direction_cosine"] = terminal_observation.first_direction_cosine
                    record["stochastic_multiplier"] = raw_denoised.new_zeros(())
                    record["terminal"] = True
                    if terminal_observation.first is not None:
                        _record_stream_norms(record, "first_derivative_rms", terminal_observation.first, layout)
                    if terminal_observation.second is not None:
                        _record_stream_norms(record, "second_derivative_rms", terminal_observation.second, layout)
                    if reference_denoised is not None:
                        record["reference"] = compare_same_state(x_current, sigmas[i], raw_denoised, reference_denoised, layout)
                    writer.write(record)
                continue

            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            lambda_s, lambda_t = coordinates[i], coordinates[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)
            stage1_coefficient = alpha_t * (1.0 - r)

            first = None
            second = None
            evidence_first = None
            evidence_second = None
            stage2 = None
            stage3 = None
            if solver_history.previous_raw is not None and solver_history.previous_coordinate is not None:
                span = lambda_s - solver_history.previous_coordinate
                scale = max(abs(lambda_s), abs(solver_history.previous_coordinate), 1.0)
                if math.isfinite(span) and abs(span) > 1e-12 * scale:
                    first = torch.nan_to_num((raw_denoised - solver_history.previous_raw) / span)
            if first is not None and solver_history.previous_first is not None and solver_history.two_back_coordinate is not None:
                span = lambda_s - solver_history.two_back_coordinate
                scale = max(abs(lambda_s), abs(solver_history.two_back_coordinate), 1.0)
                if math.isfinite(span) and abs(span) > 1e-12 * scale:
                    second = torch.nan_to_num(2.0 * (first - solver_history.previous_first) / span)

            if (
                result_is_actual
                and evidence_history.previous_raw is not None
                and evidence_history.previous_coordinate is not None
            ):
                evidence_first = divided_difference(
                    raw_denoised,
                    evidence_history.previous_raw,
                    lambda_s,
                    evidence_history.previous_coordinate,
                )
            if (
                evidence_first is not None
                and evidence_history.previous_first is not None
                and evidence_history.two_back_coordinate is not None
            ):
                evidence_second = second_divided_difference(
                    evidence_first,
                    evidence_history.previous_first,
                    lambda_s,
                    evidence_history.two_back_coordinate,
                )

            dt = er_lambda_t - er_lambda_s
            stage2_coefficient = None
            stage3_coefficient = None
            if (first is not None or evidence_first is not None) and max_stage >= 2:
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indices * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)
                integral = torch.sum(1.0 / scaled_pos) * lambda_step_size
                stage2_coefficient = alpha_t * (dt + integral * noise_scaler(er_lambda_t))
                if first is not None:
                    stage2 = stage2_coefficient * first
                if (second is not None or evidence_second is not None) and max_stage >= 3:
                    integral_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    stage3_coefficient = alpha_t * ((dt ** 2) / 2.0 + integral_u * noise_scaler(er_lambda_t))
                    if second is not None:
                        stage3 = stage3_coefficient * second

            stage1_raw = stage1_coefficient * raw_denoised
            if result_is_actual:
                evidence_stage2 = (
                    None
                    if stage2_coefficient is None or evidence_first is None
                    else stage2_coefficient * evidence_first
                )
                evidence_stage3 = (
                    None
                    if stage3_coefficient is None or evidence_second is None
                    else stage3_coefficient * evidence_second
                )
                observation = evidence_history.observe(
                    raw_denoised,
                    lambda_s,
                    lambda_t,
                    layout,
                    evidence_stage2,
                    evidence_stage3,
                    stage1_raw,
                    config.risk_sensitivity,
                )
                last_actual_observation = observation
            elif last_actual_observation is None:
                raise SpectrumInteropError(
                    "Spectrum forecast before RefDelta established an actual anchor"
                )
            else:
                observation = last_actual_observation
            stage2_gate, stage3_gate = adaptive_order_gates(observation.risk, config.adaptive_order)
            endpoint = endpoint_gate(i, total_steps, config.endpoint_fidelity_fraction, raw_denoised)
            corrected_denoised = raw_denoised
            correction_norms: dict[str, torch.Tensor] = {}
            if config.trajectory_correction:
                corrected_denoised, correction_norms = bounded_trajectory_correction(
                    raw_denoised,
                    evidence_history.previous_raw if result_is_actual else None,
                    observation.first if result_is_actual else None,
                    observation.second if result_is_actual else None,
                    lambda_t - lambda_s,
                    layout,
                    config.video_correction_strength,
                    config.audio_correction_strength,
                    config.correction_bound,
                    endpoint,
                )

            stage1 = stage1_coefficient * corrected_denoised
            x = r_alpha * r * x + stage1
            if stage2 is not None:
                x = x + stage2 * stage2_gate
            if stage3 is not None:
                x = x + stage3 * stage3_gate

            adapted_stochastic_gate = stochastic_multiplier(
                observation.risk,
                config.stochastic_adaptation_strength,
                config.minimum_stochastic_multiplier,
            )
            # Fade only this sampler's adaptation near the endpoint. A value of
            # one is native ER-SDE stochasticity, so endpoint fidelity must
            # return to one rather than suppressing the stock noise increment.
            stochastic_gate = 1.0 + endpoint * (adapted_stochastic_gate - 1.0)
            stochastic = None
            if effective_s_noise > 0.0:
                stochastic_scale = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                stochastic = alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * effective_s_noise * stochastic_scale * stochastic_gate
                x = x + stochastic
                publish_stochastic_increment(bridge, i, stochastic)

            previous_raw = evidence_history.previous_raw
            if result_is_actual and stochastic is not None and previous_raw is not None:
                stochastic_streams = layout.split(stochastic)
                movement_streams = layout.split(raw_denoised - previous_raw)
                evidence_history.previous_stochastic_ratios = {
                    name: torch.nan_to_num(
                        rms(stream) / rms(movement_streams[name]).clamp_min(torch.finfo(stream.dtype).eps),
                        nan=0.0,
                        posinf=1.0,
                    ).clamp(0.0, 1.0).detach()
                    for name, stream in stochastic_streams.items()
                }
            else:
                evidence_history.previous_stochastic_ratios = None

            solver_history.commit(raw_denoised, lambda_s, first)
            if result_is_actual:
                evidence_history.commit(raw_denoised, lambda_s, observation.first)

            if writer is not None:
                record.update({
                    "terminal": False,
                    "risk": observation.risk,
                    "stream_risk": observation.stream_risks,
                    "risk_components": observation.components,
                    "stage2_gate": stage2_gate,
                    "stage3_gate": stage3_gate,
                    "effective_order": 1.0 + stage2_gate + stage3_gate,
                    "stochastic_multiplier": stochastic_gate,
                    "correction_norm": correction_norms,
                    "denoised_difference_rms": observation.movement_rms,
                    "first_derivative_direction_cosine": observation.first_direction_cosine,
                })
                if first is not None:
                    _record_stream_norms(record, "first_derivative_rms", first, layout)
                if second is not None:
                    _record_stream_norms(record, "second_derivative_rms", second, layout)
                _record_stream_norms(record, "stage1_contribution_rms", stage1, layout)
                if stage2 is not None:
                    _record_stream_norms(record, "stage2_contribution_rms", stage2 * stage2_gate, layout)
                if stage3 is not None:
                    _record_stream_norms(record, "stage3_contribution_rms", stage3 * stage3_gate, layout)
                if stochastic is not None:
                    _record_stream_norms(record, "stochastic_rms", stochastic, layout)
                    for name, stream in layout.split(stochastic).items():
                        latent_stream = layout.split(x)[name]
                        movement_stream = layout.split(raw_denoised - previous_raw)[name] if previous_raw is not None else raw_denoised.new_zeros(stream.shape)
                        eps = torch.finfo(stream.dtype).eps
                        record.setdefault(name, {})["stochastic_to_latent"] = rms(stream) / rms(latent_stream).clamp_min(eps)
                        record.setdefault(name, {})["stochastic_to_denoised_movement"] = rms(stream) / rms(movement_stream).clamp_min(eps)
                if reference_denoised is not None:
                    record["reference"] = compare_same_state(x_current, sigmas[i], raw_denoised, reference_denoised, layout)
                writer.write(record)
    finally:
        solver_history.reset()
        evidence_history.reset()
        if writer is not None:
            writer.close()
    return x


sample_refdelta_er_sde.__spectrum_interop_contract__ = SPECTRUM_INTEROP_CONTRACT


__all__ = ["native_noise_scaler", "sample_refdelta_er_sde"]
