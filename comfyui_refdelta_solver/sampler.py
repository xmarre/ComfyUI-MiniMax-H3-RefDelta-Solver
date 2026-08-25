from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch

from .calibration_replay import CalibrationCaptureWriter
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
from .stochastic_control import (
    StochasticControlResult,
    StochasticStabilityController,
    tensor_distribution,
)
from .telemetry import TelemetryWriter
from .trajectory import (
    StreamLayout,
    TrajectoryHistory,
    adaptive_order_gates,
    bounded_trajectory_correction,
    rms,
    stochastic_multiplier,
    stochastic_pressure_from_ratio,
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
    video_shape = tuple(int(size) for size in latent_shapes[0][1:])
    video_elements = math.prod(video_shape)
    if x.shape[-1] <= video_elements:
        raise ValueError("MiniMax H3 RefDelta sampler received an invalid packed AV latent")
    return StreamLayout(
        video_elements,
        video_shape if len(video_shape) == 4 else None,
    )


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


def _calibration_capture_writer(
    config: RefDeltaSamplerConfig,
    extra_args: dict[str, Any],
    x: torch.Tensor,
    sigmas: torch.Tensor,
    layout: StreamLayout,
) -> CalibrationCaptureWriter | None:
    if not config.calibration_capture:
        return None
    import folder_paths

    return CalibrationCaptureWriter(
        Path(folder_paths.get_output_directory()),
        config.calibration_id,
        extra_args.get("seed"),
        len(sigmas) - 1,
        tuple(x.shape),
        sigmas,
        layout,
    )


def _record_stream_norms(record: dict[str, Any], prefix: str, value: torch.Tensor, layout: StreamLayout) -> None:
    for name, stream in layout.split(value).items():
        record.setdefault(name, {})[prefix] = rms(stream)


def _stochastic_movement_ratio(
    stochastic: torch.Tensor,
    movement: torch.Tensor,
) -> torch.Tensor | None:
    """Return stochastic/movement RMS only when movement defines a useful denominator."""
    denominator = rms(movement)
    tiny = torch.finfo(movement.dtype).tiny
    if not bool(torch.isfinite(denominator)) or bool(denominator <= tiny):
        return None
    ratio = rms(stochastic) / denominator
    return torch.nan_to_num(ratio, nan=0.0, posinf=1e9, neginf=0.0).clamp_min(0.0)


_TRAJECTORY_RISK_COMPONENTS = (
    "curvature",
    "direction_change",
    "magnitude_jump",
    "extrapolation_error",
)


def _forecast_observation_with_latest_stochastic_evidence(
    observation,
    ratios: dict[str, torch.Tensor] | None,
    sensitivity: float,
    max_stage: int,
):
    """Refresh only stochastic evidence for a cached actual observation.

    The current actual step must keep the observation that actually controlled
    that step.  A subsequent Spectrum forecast, however, should see the native
    stochastic/movement ratio measured after that actual step.  Rebuild only
    the combined risk/stochastic fields while leaving every trajectory term and
    derivative anchored to the actual model evaluation.
    """
    refreshed = copy.copy(observation)
    zero = observation.risk.new_zeros(())
    stream_risks: dict[str, torch.Tensor] = {}
    stream_stochastic_pressures: dict[str, torch.Tensor] = {}
    components: dict[str, dict[str, torch.Tensor]] = {}

    for name, source_values in observation.components.items():
        values = dict(source_values)
        values.pop("previous_native_stochastic_ratio", None)
        values.pop("previous_stochastic_pressure", None)

        trajectory_signals = [
            values[key]
            for key in _TRAJECTORY_RISK_COMPONENTS
            if key in values
        ]
        # observe() includes the stage term whenever an actual first derivative
        # allowed ER stage 2/3 evidence for this step.
        if (
            max_stage >= 2
            and observation.first is not None
            and "stage_correction_ratio" in values
        ):
            trajectory_signals.append(values["stage_correction_ratio"])

        combined_signals = list(trajectory_signals)
        if ratios is not None and name in ratios:
            native_ratio = torch.nan_to_num(
                ratios[name],
                nan=0.0,
                posinf=1e9,
                neginf=0.0,
            ).clamp_min(0.0)
            pressure = stochastic_pressure_from_ratio(native_ratio)
            values["previous_native_stochastic_ratio"] = native_ratio
            values["previous_stochastic_pressure"] = pressure
            stream_stochastic_pressures[name] = pressure
            combined_signals.append(pressure)

        risk = torch.stack(combined_signals).mean() if combined_signals else zero
        stream_risks[name] = (risk * sensitivity).clamp(0.0, 1.0)
        components[name] = values

    refreshed.risk = (
        torch.stack(list(stream_risks.values())).amax()
        if stream_risks
        else zero
    )
    refreshed.stream_risks = stream_risks
    refreshed.stochastic_pressure = (
        torch.stack(list(stream_stochastic_pressures.values())).amax()
        if stream_stochastic_pressures
        else zero
    )
    refreshed.stream_stochastic_pressures = stream_stochastic_pressures
    refreshed.components = components
    return refreshed


def _write_record(
    writer: TelemetryWriter | None,
    capture: CalibrationCaptureWriter | None,
    step: int,
    record: dict[str, Any],
) -> None:
    if writer is not None:
        writer.write(record)
    if capture is not None:
        capture.write_record(step, record)


def _stochastic_control_record(
    config: RefDeltaSamplerConfig,
    controller: StochasticStabilityController,
    result: StochasticControlResult | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stochastic_control_mode": config.stochastic_control_mode,
        "effective_video_stochastic_strength": controller.effective_video_strength,
        "effective_audio_stochastic_strength": controller.effective_audio_strength,
        "static_video_stochastic_adaptation_strength": (
            config.static_video_stochastic_adaptation_strength
        ),
    }
    if result is None:
        return record
    record.update(
        {
            "legacy_target_stochastic_multiplier": result.legacy_target_gate,
            "video_dynamic_target_multiplier": result.video_dynamic_target_gate,
            "video_static_target_multiplier": result.video_static_target_gate,
            "audio_target_multiplier": result.audio_target_gate,
            "audio_applied_gate": result.audio_applied_gate,
            "video_stability_progress_gate": result.progress_gate,
            "video_stability_source_actual_step": result.source_actual_step,
            "video_stability_updated_this_step": result.stability_updated_this_step,
            "stochastic_gate_slew_applied": result.slew_applied,
            "stochastic_gate_slew_fraction": result.slew_fraction,
        }
    )
    if result.video_applied_gate is not None:
        for name, value in tensor_distribution(result.video_applied_gate).items():
            record[f"video_applied_gate_{name}"] = value
    if result.restore_mask is not None:
        for name, value in tensor_distribution(
            result.restore_mask,
            active_threshold=0.5,
        ).items():
            record[f"video_restore_{name}"] = value
    if result.temporal_motion_ratio is not None:
        stats = tensor_distribution(result.temporal_motion_ratio)
        for name in ("mean", "p50", "p95"):
            record[f"video_temporal_motion_ratio_{name}"] = stats[name]
    if result.diffusion_change_ratio is not None:
        stats = tensor_distribution(result.diffusion_change_ratio)
        for name in ("mean", "p50", "p95"):
            record[f"video_diffusion_change_ratio_{name}"] = stats[name]
    return record


def _write_stability_maps(
    writer: TelemetryWriter | None,
    config: RefDeltaSamplerConfig,
    result: StochasticControlResult | None,
    step: int,
    sigma: torch.Tensor,
    seed: int | None,
) -> None:
    if (
        writer is None
        or not config.debug_stability_maps
        or result is None
        or not result.stability_updated_this_step
        or result.restore_mask is None
        or result.temporal_motion_ratio is None
        or result.video_applied_gate is None
    ):
        return
    applied_gate = result.video_applied_gate
    if applied_gate.ndim == 0:
        applied_gate = applied_gate.expand_as(result.restore_mask)
    writer.write_stability_maps(
        step,
        sigma,
        result.temporal_motion_ratio,
        result.diffusion_change_ratio,
        result.restore_mask,
        applied_gate,
        {
            "seed": seed,
            "stochastic_control_mode": config.stochastic_control_mode,
            "config": {
                "stochastic_adaptation_strength": config.stochastic_adaptation_strength,
                "minimum_stochastic_multiplier": config.minimum_stochastic_multiplier,
                "video_stochastic_strength_scale": config.video_stochastic_strength_scale,
                "static_video_stochastic_adaptation_strength": (
                    config.static_video_stochastic_adaptation_strength
                ),
                "video_stability_restore_strength": config.video_stability_restore_strength,
                "video_stability_motion_low": config.video_stability_motion_low,
                "video_stability_motion_high": config.video_stability_motion_high,
                "video_stability_diffusion_low": config.video_stability_diffusion_low,
                "video_stability_diffusion_high": config.video_stability_diffusion_high,
                "video_stability_diffusion_weight": config.video_stability_diffusion_weight,
                "video_stability_normalization_floor": (
                    config.video_stability_normalization_floor
                ),
                "video_stability_gamma": config.video_stability_gamma,
                "video_stability_spatial_radius": config.video_stability_spatial_radius,
                "video_stability_temporal_radius": config.video_stability_temporal_radius,
                "video_stability_ema": config.video_stability_ema,
                "video_stability_start_fraction": config.video_stability_start_fraction,
                "video_stability_full_fraction": config.video_stability_full_fraction,
                "stochastic_gate_slew_limit": config.stochastic_gate_slew_limit,
            },
        },
    )


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
    stochastic_controller = StochasticStabilityController(
        config,
        layout,
        multiplier=stochastic_multiplier,
    )

    num_integration_points = 200.0
    point_indices = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)
    sigmas = k_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = k_sampling.sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()
    coordinates = [float(value) for value in er_lambdas.detach().float().cpu()]

    # Solver history follows every value returned to ER-SDE, including a Spectrum
    # forecast. Evidence history accepts only genuine model evaluations. Forecast
    # values may consume the latest actual-only control evidence, but they never
    # become risk, stochastic-pressure, or trajectory-correction anchors.
    solver_history = TrajectoryHistory()
    evidence_history = TrajectoryHistory()
    last_actual_observation = None
    bridge = spectrum_bridge(extra_args)
    writer = _telemetry_writer(config, extra_args)
    capture = _calibration_capture_writer(config, extra_args, x, sigmas, layout)
    recording = writer is not None or capture is not None
    total_steps = len(sigmas) - 1
    completed = False
    try:
        for i in trange(total_steps, disable=disable):
            x_current = x
            raw_denoised = model(x_current, sigmas[i] * s_in, **extra_args)
            result_is_actual = model_result_is_actual(bridge, i)
            if capture is not None:
                capture.write_step(
                    i,
                    x_current,
                    raw_denoised,
                    actual=result_is_actual,
                )
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
                "actual_model_evaluation": result_is_actual,
                "sigma": sigmas[i],
                "sigma_next": sigmas[i + 1],
                "er_lambda": er_lambdas[i],
                "er_lambda_next": er_lambdas[i + 1],
                "audio_sigma": effective_audio_sigma(sigmas[i], video_shift, audio_shift),
                "audio_sigma_next": effective_audio_sigma(sigmas[i + 1], video_shift, audio_shift),
                "sigma_min": model_sampling.sigma_min,
                "sigma_max": model_sampling.sigma_max,
            }
            record.update(
                _stochastic_control_record(config, stochastic_controller, None)
            )
            if recording:
                _record_stream_norms(record, "latent_rms", x, layout)
                _record_stream_norms(record, "denoised_rms", raw_denoised, layout)

            if sigmas[i + 1] == 0:
                x = raw_denoised
                if result_is_actual:
                    stochastic_controller.update_actual(
                        raw_denoised,
                        evidence_history.previous_raw,
                        i,
                    )
                if recording:
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
                    record["trajectory_risk"] = terminal_observation.trajectory_risk
                    record["stream_trajectory_risk"] = terminal_observation.stream_trajectory_risks
                    record["stochastic_pressure"] = terminal_observation.stochastic_pressure
                    record["stream_stochastic_pressure"] = terminal_observation.stream_stochastic_pressures
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
                    _write_record(writer, capture, i, record)
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
            if result_is_actual:
                stochastic_controller.update_actual(
                    raw_denoised,
                    evidence_history.previous_raw,
                    i,
                )
            # ER order responds only to the model trajectory. Native stochastic
            # pressure is tracked separately and must not suppress deterministic
            # stage-2/3 corrections merely because ER-SDE itself injects noise.
            stage2_gate, stage3_gate = adaptive_order_gates(
                observation.trajectory_risk,
                config.adaptive_order,
            )
            endpoint = endpoint_gate(i, total_steps, config.endpoint_fidelity_fraction, raw_denoised)
            corrected_denoised = raw_denoised
            correction_norms: dict[str, torch.Tensor] = {}
            if config.trajectory_correction:
                # Application may target a Spectrum forecast, but the steering
                # vector and bound anchor remain derived from actual-model history.
                corrected_denoised, correction_norms = bounded_trajectory_correction(
                    raw_denoised,
                    evidence_history.previous_raw,
                    observation.first,
                    observation.second,
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

            native_stochastic = None
            stochastic = None
            stochastic_control = None
            if effective_s_noise > 0.0:
                stochastic_scale = (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
                native_stochastic = (
                    alpha_t
                    * noise_sampler(sigmas[i], sigmas[i + 1])
                    * effective_s_noise
                    * stochastic_scale
                )
                stochastic_control = stochastic_controller.apply(
                    native_stochastic,
                    observation,
                    endpoint,
                    i,
                    total_steps,
                    collect_stats=recording,
                )
                stochastic = stochastic_control.increment
                x = x + stochastic
                # Spectrum owns the exact post-adaptation increment that was
                # actually applied to x, not the hypothetical native increment.
                publish_stochastic_increment(bridge, i, stochastic)

            previous_raw = evidence_history.previous_raw
            if result_is_actual:
                if native_stochastic is not None and previous_raw is not None:
                    native_streams = layout.split(native_stochastic)
                    movement_streams = layout.split(raw_denoised - previous_raw)
                    ratios: dict[str, torch.Tensor] = {}
                    for name, stream in native_streams.items():
                        ratio = _stochastic_movement_ratio(stream, movement_streams[name])
                        if ratio is not None:
                            ratios[name] = ratio.detach()
                    evidence_history.previous_stochastic_ratios = ratios or None
                else:
                    evidence_history.previous_stochastic_ratios = None
                if bridge is not None:
                    # The ratio measured above is evidence from this genuine
                    # model evaluation and is causally available to the very next
                    # forecast. Keep the current step's local observation intact
                    # for telemetry, but refresh the cached forecast view now.
                    last_actual_observation = _forecast_observation_with_latest_stochastic_evidence(
                        observation,
                        evidence_history.previous_stochastic_ratios,
                        config.risk_sensitivity,
                        max_stage,
                    )
            # A forecast contributes no new stochastic evidence. Preserve the
            # most recent actual native stochastic/movement ratios until another
            # actual model evaluation replaces or clears them.

            solver_history.commit(raw_denoised, lambda_s, first)
            if result_is_actual:
                evidence_history.commit(raw_denoised, lambda_s, observation.first)

            if recording:
                record.update({
                    "terminal": False,
                    "risk": observation.risk,
                    "stream_risk": observation.stream_risks,
                    "trajectory_risk": observation.trajectory_risk,
                    "stream_trajectory_risk": observation.stream_trajectory_risks,
                    "stochastic_pressure": observation.stochastic_pressure,
                    "stream_stochastic_pressure": observation.stream_stochastic_pressures,
                    "risk_components": observation.components,
                    "stage2_gate": stage2_gate,
                    "stage3_gate": stage3_gate,
                    "effective_order": 1.0 + stage2_gate + stage3_gate,
                    "stochastic_multiplier": (
                        raw_denoised.new_ones(())
                        if stochastic_control is None
                        else stochastic_control.compatibility_gate
                    ),
                    "correction_norm": correction_norms,
                    "denoised_difference_rms": observation.movement_rms,
                    "first_derivative_direction_cosine": observation.first_direction_cosine,
                })
                record.update(
                    _stochastic_control_record(
                        config,
                        stochastic_controller,
                        stochastic_control,
                    )
                )
                if first is not None:
                    _record_stream_norms(record, "first_derivative_rms", first, layout)
                if second is not None:
                    _record_stream_norms(record, "second_derivative_rms", second, layout)
                _record_stream_norms(record, "stage1_contribution_rms", stage1, layout)
                if stage2 is not None:
                    _record_stream_norms(record, "stage2_contribution_rms", stage2 * stage2_gate, layout)
                if stage3 is not None:
                    _record_stream_norms(record, "stage3_contribution_rms", stage3 * stage3_gate, layout)
                if native_stochastic is not None and stochastic is not None:
                    _record_stream_norms(record, "native_stochastic_rms", native_stochastic, layout)
                    _record_stream_norms(record, "stochastic_rms", stochastic, layout)
                    native_streams = layout.split(native_stochastic)
                    applied_streams = layout.split(stochastic)
                    latent_streams = layout.split(x)
                    movement_streams = (
                        layout.split(raw_denoised - previous_raw)
                        if previous_raw is not None
                        else None
                    )
                    for name, native_stream in native_streams.items():
                        applied_stream = applied_streams[name]
                        latent_stream = latent_streams[name]
                        eps = torch.finfo(native_stream.dtype).eps
                        values = record.setdefault(name, {})
                        values["native_stochastic_to_latent"] = (
                            rms(native_stream) / rms(latent_stream).clamp_min(eps)
                        )
                        values["stochastic_to_latent"] = (
                            rms(applied_stream) / rms(latent_stream).clamp_min(eps)
                        )
                        native_ratio = None
                        applied_ratio = None
                        if movement_streams is not None:
                            native_ratio = _stochastic_movement_ratio(
                                native_stream,
                                movement_streams[name],
                            )
                            applied_ratio = _stochastic_movement_ratio(
                                applied_stream,
                                movement_streams[name],
                            )
                        values["native_stochastic_to_denoised_movement"] = native_ratio
                        values["stochastic_to_denoised_movement"] = applied_ratio
                if reference_denoised is not None:
                    record["reference"] = compare_same_state(x_current, sigmas[i], raw_denoised, reference_denoised, layout)
                _write_record(writer, capture, i, record)
                _write_stability_maps(
                    writer,
                    config,
                    stochastic_control,
                    i,
                    sigmas[i],
                    seed,
                )
        completed = True
    finally:
        solver_history.reset()
        evidence_history.reset()
        stochastic_controller.reset()
        if writer is not None:
            writer.close()
        if capture is not None:
            if completed:
                capture.close(x)
            else:
                capture.abort()
    return x


sample_refdelta_er_sde.__spectrum_interop_contract__ = SPECTRUM_INTEROP_CONTRACT


__all__ = ["native_noise_scaler", "sample_refdelta_er_sde"]
