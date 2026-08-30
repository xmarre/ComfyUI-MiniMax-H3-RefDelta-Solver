from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .config import RefDeltaSamplerConfig
from .coordinates import endpoint_gate
from .sampler import (
    _record_stream_norms,
    _stochastic_control_record,
    _stream_layout,
    _telemetry_writer,
    _validate_h3_sampling,
    _write_record,
)
from .spectrum_interop import (
    SPECTRUM_BACKEND_INTEROP_CONTRACT,
    SpectrumInteropError,
    model_result_is_actual,
    spectrum_backend_bridge,
)
from .stochastic_control import (
    StochasticControlResult,
    StochasticStabilityController,
    apply_stochastic_control_gates,
)
from .trajectory import (
    TrajectoryHistory,
    bounded_trajectory_correction,
    stochastic_multiplier,
)


REFDELTA_BASE_SAMPLERS = (
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
)


def _native_backend(name: str):
    from comfy.k_diffusion import sampling as native_sampling

    function = getattr(native_sampling, name, None)
    if not callable(function):
        raise TypeError(
            f"RefDelta backend {name!r} requires a ComfyUI revision that provides "
            f"comfy.k_diffusion.sampling.{name}"
        )
    return function


@dataclass(slots=True)
class _NoiseContext:
    observation: Any
    endpoint: torch.Tensor
    outer_step: int


class _RefDeltaBackendState:
    """Outer-interval RefDelta control shared by non-ER base samplers.

    SEEDS internal stages stay native. One stochastic gate is resolved from the
    outer denoiser and then reused for every correlated noise draw belonging to
    that outer interval.

    SA-Solver PEC has one endpoint model evaluation per interval. Active PECE has
    an explicit P0, then P_i/C_i topology. For PECE, only P0 and exact corrected
    C_i calls own persistent RefDelta evidence and stochastic-control state.
    Predicted P_i calls for i > 0 remain ephemeral current-corrector inputs, even
    when Spectrum executes them as actual H3 evaluations.
    """

    def __init__(
        self,
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict[str, Any],
        config: RefDeltaSamplerConfig,
        backend_name: str,
        stage_count: int,
        active_pece: bool | None = None,
    ) -> None:
        from comfy.k_diffusion import sampling as native_sampling

        self.model = model
        self.config = config
        self.backend_name = backend_name
        self.stage_count = stage_count
        self.active_pece = (
            backend_name == "sa_solver_pece"
            if active_pece is None
            else bool(active_pece)
        )
        self.layout = _stream_layout(model, x)
        model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
        _validate_h3_sampling(model_sampling)
        prepared = native_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
        self.sigmas = prepared
        self.outer_steps = max(0, len(prepared) - 1)
        # Keep RefDelta's released ER-lambda evidence coordinate independent
        # of the selected numerical backend so curvature/risk thresholds and
        # trajectory-correction strengths retain the same scale.
        coordinate_tensor = native_sampling.sigma_to_half_log_snr(
            prepared,
            model_sampling=model_sampling,
        ).neg().exp()
        # Transfer the complete schedule once. Pulling one scalar to CPU inside
        # every outer model call would introduce a device synchronization at
        # every solver interval.
        self.coordinates = [
            float(value)
            for value in coordinate_tensor.detach().float().cpu()
        ]
        self.bridge = spectrum_backend_bridge(extra_args)
        self.history = TrajectoryHistory()
        self.controller = StochasticStabilityController(
            config,
            self.layout,
            multiplier=stochastic_multiplier,
        )
        self.last_actual_observation = None
        self.call_index = 0
        self.outer_index = 0
        self.noise_context: _NoiseContext | None = None
        self.frozen_control: StochasticControlResult | None = None
        self.writer = _telemetry_writer(config, extra_args)
        self.pending_record: dict[str, Any] | None = None

    def _pece_call_descriptor(self, call_id: int) -> tuple[int, str, bool]:
        """Map native PECE model-call order to (outer_step, phase, endpoint_owner)."""
        if call_id == 0:
            return 0, "predicted", True
        outer_step = (call_id + 1) // 2
        phase = "predicted" if call_id % 2 else "corrected"
        return outer_step, phase, phase == "corrected"

    def _coordinate(self, index: int) -> float:
        numeric = self.coordinates[index]
        if not math.isfinite(numeric):
            raise ValueError(
                f"{self.backend_name} produced a non-finite RefDelta trajectory coordinate"
            )
        return numeric

    def _flush_record(self) -> None:
        if self.pending_record is None:
            return
        _write_record(
            self.writer,
            None,
            int(self.pending_record["step"]),
            self.pending_record,
        )
        self.pending_record = None

    def _record_outer(
        self,
        raw: torch.Tensor,
        corrected: torch.Tensor,
        observation: Any,
        actual: bool,
        outer_step: int,
        call_id: int,
        coordinate: float,
        next_coordinate: float,
        correction_norms: dict[str, torch.Tensor],
        terminal: bool,
        *,
        solver_phase: str = "outer",
        persistent_endpoint: bool = True,
    ) -> None:
        if self.writer is None:
            return
        self._flush_record()
        record: dict[str, Any] = {
            "step": outer_step,
            "steps": self.outer_steps,
            "model_call_id": call_id,
            "base_sampler": self.backend_name,
            "solver_phase": solver_phase,
            "persistent_endpoint_evidence": persistent_endpoint,
            "actual_model_evaluation": actual,
            "sigma": self.sigmas[outer_step],
            "sigma_next": self.sigmas[outer_step + 1],
            "coordinate": coordinate,
            "coordinate_next": next_coordinate,
            "terminal": terminal,
            "risk": observation.risk,
            "stream_risk": observation.stream_risks,
            "trajectory_risk": observation.trajectory_risk,
            "stream_trajectory_risk": observation.stream_trajectory_risks,
            "stochastic_pressure": observation.stochastic_pressure,
            "stream_stochastic_pressure": observation.stream_stochastic_pressures,
            "risk_components": observation.components,
            "denoised_difference_rms": observation.movement_rms,
            "first_derivative_direction_cosine": observation.first_direction_cosine,
            "correction_norm": correction_norms,
            "stochastic_multiplier": raw.new_ones(()),
        }
        _record_stream_norms(record, "denoised_rms", raw, self.layout)
        if corrected is not raw:
            _record_stream_norms(record, "corrected_denoised_rms", corrected, self.layout)
        if observation.first is not None:
            _record_stream_norms(record, "first_derivative_rms", observation.first, self.layout)
        if observation.second is not None:
            _record_stream_norms(record, "second_derivative_rms", observation.second, self.layout)
        record.update(_stochastic_control_record(self.config, self.controller, None))
        self.pending_record = record
        if terminal:
            self._flush_record()

    def model_call(self, state: torch.Tensor, sigma: torch.Tensor, **extra_args):
        raw = self.model(state, sigma, **extra_args)
        call_id = self.call_index
        self.call_index += 1
        actual = model_result_is_actual(self.bridge, call_id)

        solver_phase = "outer"
        persistent_endpoint = True

        if self.active_pece:
            outer_step, solver_phase, persistent_endpoint = (
                self._pece_call_descriptor(call_id)
            )
            if outer_step >= self.outer_steps:
                raise RuntimeError(
                    "sa_solver_pece emitted more model calls than its PECE topology allows"
                )

            if not persistent_endpoint:
                # Native PECE uses P_i only for the current corrector; C_i then
                # replaces that same-coordinate entry before the next predictor.
                # Do not let P_i become a second RefDelta history observation.
                if self.last_actual_observation is None:
                    raise SpectrumInteropError(
                        "SA-Solver PECE predicted phase arrived before RefDelta "
                        "established an exact persistent endpoint"
                    )
                self.noise_context = None
                self.frozen_control = None
                correction_norms = {
                    name: raw.new_zeros(()) for name in self.layout.split(raw)
                }
                terminal = bool(self.sigmas[outer_step + 1] == 0)
                coordinate = self._coordinate(outer_step)
                next_coordinate = (
                    coordinate if terminal else self._coordinate(outer_step + 1)
                )
                self._record_outer(
                    raw,
                    raw,
                    self.last_actual_observation,
                    actual,
                    outer_step,
                    call_id,
                    coordinate,
                    next_coordinate,
                    correction_norms,
                    terminal,
                    solver_phase=solver_phase,
                    persistent_endpoint=False,
                )
                return raw

            # P0 and every C_i are native PECE persistent endpoints. Under
            # Spectrum they must be actual model evaluations; accepting a
            # forecast here would contaminate RefDelta evidence and the native
            # Adams endpoint simultaneously.
            if not actual:
                raise SpectrumInteropError(
                    "Spectrum forecasted a persistent SA-Solver PECE endpoint; "
                    f"outer_step={outer_step} phase={solver_phase}"
                )
        else:
            stage_index = (call_id % self.stage_count)
            if stage_index != 0:
                # Internal SEEDS evaluations stay mathematically native. We still
                # consume Spectrum's classification so bridge ordering is checked.
                return raw

            outer_step = self.outer_index
            self.outer_index += 1
            if outer_step >= self.outer_steps:
                raise RuntimeError(
                    f"{self.backend_name} emitted more outer model calls than its sigma schedule"
                )
            if self.backend_name == "sa_solver":
                solver_phase = "predicted"

        terminal = bool(self.sigmas[outer_step + 1] == 0)
        coordinate = self._coordinate(outer_step)
        next_coordinate = (
            coordinate if terminal else self._coordinate(outer_step + 1)
        )

        if actual:
            observation = self.history.observe(
                raw,
                coordinate,
                next_coordinate,
                self.layout,
                None,
                None,
                raw,
                self.config.risk_sensitivity,
            )
            self.last_actual_observation = observation
            self.controller.update_actual(
                raw,
                self.history.previous_raw,
                outer_step,
            )
        elif self.last_actual_observation is None:
            raise SpectrumInteropError(
                "Spectrum forecast before RefDelta established an actual outer anchor"
            )
        else:
            observation = self.last_actual_observation

        endpoint = endpoint_gate(
            outer_step,
            self.outer_steps,
            self.config.endpoint_fidelity_fraction,
            raw,
        )
        corrected = raw
        correction_norms: dict[str, torch.Tensor] = {
            name: raw.new_zeros(()) for name in self.layout.split(raw)
        }
        if not terminal and self.config.trajectory_correction:
            corrected, correction_norms = bounded_trajectory_correction(
                raw,
                self.history.previous_raw,
                observation.first,
                observation.second,
                next_coordinate - coordinate,
                self.layout,
                self.config.video_correction_strength,
                self.config.audio_correction_strength,
                self.config.correction_bound,
                endpoint,
            )

        if actual:
            self.history.commit(raw, coordinate, observation.first)

        self.noise_context = None if terminal else _NoiseContext(
            observation=observation,
            endpoint=endpoint,
            outer_step=outer_step,
        )
        self.frozen_control = None
        self._record_outer(
            raw,
            corrected,
            observation,
            actual,
            outer_step,
            call_id,
            coordinate,
            next_coordinate,
            correction_norms,
            terminal,
            solver_phase=solver_phase,
            persistent_endpoint=persistent_endpoint,
        )
        return corrected

    def adapt_noise(self, native_noise: torch.Tensor) -> torch.Tensor:
        context = self.noise_context
        if context is None or self.config.stochastic_adaptation_strength <= 0.0:
            return native_noise

        if self.frozen_control is None:
            result = self.controller.apply(
                native_noise,
                context.observation,
                context.endpoint,
                context.outer_step,
                self.outer_steps,
                collect_stats=self.writer is not None,
            )
            self.frozen_control = result
            adapted = result.increment
            if self.pending_record is not None:
                self.pending_record.update(
                    _stochastic_control_record(self.config, self.controller, result)
                )
                self.pending_record["stochastic_multiplier"] = result.compatibility_gate
                _record_stream_norms(
                    self.pending_record,
                    "native_noise_draw_rms",
                    native_noise,
                    self.layout,
                )
                _record_stream_norms(
                    self.pending_record,
                    "adapted_noise_draw_rms",
                    adapted,
                    self.layout,
                )
            return adapted

        return apply_stochastic_control_gates(
            native_noise,
            self.layout,
            self.frozen_control,
        )

    def finish(self) -> None:
        self._flush_record()
        self.history.reset()
        self.controller.reset()
        if self.writer is not None:
            self.writer.close()


class _ModelAdapter:
    def __init__(self, state: _RefDeltaBackendState) -> None:
        self._state = state

    def __getattr__(self, name: str):
        return getattr(self._state.model, name)

    def __call__(self, x, sigma, **extra_args):
        return self._state.model_call(x, sigma, **extra_args)


class _NoiseAdapter:
    def __init__(
        self,
        state: _RefDeltaBackendState,
        base: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._state = state
        self._base = base

    def __call__(self, sigma, sigma_next):
        return self._state.adapt_noise(self._base(sigma, sigma_next))


def _prepare_backend(
    model,
    x,
    sigmas,
    extra_args,
    config,
    backend_name,
    stage_count,
    noise_sampler,
    active_pece=None,
):
    from comfy.k_diffusion import sampling as native_sampling

    extra_args = {} if extra_args is None else extra_args
    config = RefDeltaSamplerConfig() if config is None else config
    config.validate()
    if config.calibration_capture:
        raise ValueError(
            "RefDelta calibration_capture is currently defined only for the ER-SDE backend"
        )
    state = _RefDeltaBackendState(
        model,
        x,
        sigmas,
        extra_args,
        config,
        backend_name,
        stage_count,
        active_pece=active_pece,
    )
    seed = extra_args.get("seed")
    base_noise = (
        native_sampling.default_noise_sampler(x, seed=seed)
        if noise_sampler is None
        else noise_sampler
    )
    return extra_args, config, state, _ModelAdapter(state), _NoiseAdapter(state, base_noise)


def prepare_refdelta_backend_adapters(
    model,
    x,
    sigmas,
    extra_args,
    config,
    backend_name,
    stage_count,
    noise_sampler=None,
    active_pece=None,
):
    """Build the reviewed RefDelta model/noise adapters for an external solver owner.

    Spectrum uses this narrow hook for its SA isolated-history adapter so the
    validated Adams-history policy remains Spectrum-owned while RefDelta retains
    ownership of trajectory correction and stochastic gating.
    """
    _, _, state, adapted_model, adapted_noise = _prepare_backend(
        model,
        x,
        sigmas,
        extra_args,
        config,
        backend_name,
        stage_count,
        noise_sampler,
        active_pece=active_pece,
    )
    return state, adapted_model, adapted_noise


@torch.no_grad()
def sample_refdelta_seeds_2(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=0.5,
    solver_type="phi_1",
    config: RefDeltaSamplerConfig | None = None,
):
    """RefDelta outer-trajectory control over native ComfyUI SEEDS-2."""
    config = RefDeltaSamplerConfig() if config is None else config
    config.validate()
    if config.calibration_capture:
        raise ValueError(
            "RefDelta calibration_capture is currently defined only for the ER-SDE backend"
        )
    native_function = _native_backend("sample_seeds_2")
    if config.is_native_equivalence_mode:
        return native_function(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            r=r,
            solver_type=solver_type,
        )

    extra_args, config, state, adapted_model, adapted_noise = _prepare_backend(
        model,
        x,
        sigmas,
        extra_args,
        config,
        "seeds_2",
        2,
        noise_sampler,
    )
    try:
        return native_function(
            adapted_model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=adapted_noise,
            r=r,
            solver_type=solver_type,
        )
    finally:
        state.finish()


@torch.no_grad()
def sample_refdelta_seeds_3(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r_1=1.0 / 3.0,
    r_2=2.0 / 3.0,
    config: RefDeltaSamplerConfig | None = None,
):
    """RefDelta outer-trajectory control over native ComfyUI SEEDS-3."""
    config = RefDeltaSamplerConfig() if config is None else config
    config.validate()
    if config.calibration_capture:
        raise ValueError(
            "RefDelta calibration_capture is currently defined only for the ER-SDE backend"
        )
    native_function = _native_backend("sample_seeds_3")
    if config.is_native_equivalence_mode:
        return native_function(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            r_1=r_1,
            r_2=r_2,
        )

    extra_args, config, state, adapted_model, adapted_noise = _prepare_backend(
        model,
        x,
        sigmas,
        extra_args,
        config,
        "seeds_3",
        3,
        noise_sampler,
    )
    try:
        return native_function(
            adapted_model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            eta=eta,
            s_noise=s_noise,
            noise_sampler=adapted_noise,
            r_1=r_1,
            r_2=r_2,
        )
    finally:
        state.finish()


def _sample_refdelta_sa_solver_common(
    model,
    x,
    sigmas,
    *,
    extra_args,
    callback,
    disable,
    tau_func,
    s_noise,
    noise_sampler,
    predictor_order,
    corrector_order,
    simple_order_2,
    config,
    use_pece,
):
    config = RefDeltaSamplerConfig() if config is None else config
    config.validate()
    if config.calibration_capture:
        raise ValueError(
            "RefDelta calibration_capture is currently defined only for the ER-SDE backend"
        )

    native_function = _native_backend("sample_sa_solver")
    if config.is_native_equivalence_mode:
        return native_function(
            model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            tau_func=tau_func,
            s_noise=s_noise,
            noise_sampler=noise_sampler,
            predictor_order=predictor_order,
            corrector_order=corrector_order,
            use_pece=use_pece,
            simple_order_2=simple_order_2,
        )

    backend_name = "sa_solver_pece" if use_pece else "sa_solver"
    extra_args, config, state, adapted_model, adapted_noise = _prepare_backend(
        model,
        x,
        sigmas,
        extra_args,
        config,
        backend_name,
        1,
        noise_sampler,
        active_pece=bool(use_pece and corrector_order > 0),
    )
    try:
        return native_function(
            adapted_model,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            tau_func=tau_func,
            s_noise=s_noise,
            noise_sampler=adapted_noise,
            predictor_order=predictor_order,
            corrector_order=corrector_order,
            use_pece=use_pece,
            simple_order_2=simple_order_2,
        )
    finally:
        state.finish()


@torch.no_grad()
def sample_refdelta_sa_solver(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    tau_func=None,
    s_noise=1.0,
    noise_sampler=None,
    predictor_order=3,
    corrector_order=4,
    simple_order_2=False,
    config: RefDeltaSamplerConfig | None = None,
):
    """RefDelta trajectory/stochastic control over native SA-Solver PEC."""
    return _sample_refdelta_sa_solver_common(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        tau_func=tau_func,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        simple_order_2=simple_order_2,
        config=config,
        use_pece=False,
    )


@torch.no_grad()
def sample_refdelta_sa_solver_pece(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    tau_func=None,
    s_noise=1.0,
    noise_sampler=None,
    predictor_order=3,
    corrector_order=4,
    simple_order_2=False,
    config: RefDeltaSamplerConfig | None = None,
):
    """RefDelta endpoint control over native SA-Solver PECE.

    P_i for i > 0 is an ephemeral current-corrector evaluation. P0 and corrected
    C_i evaluations are the only persistent RefDelta trajectory/stochastic
    evidence owners, matching native PECE endpoint replacement.
    """
    return _sample_refdelta_sa_solver_common(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        tau_func=tau_func,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        simple_order_2=simple_order_2,
        config=config,
        use_pece=True,
    )


for _function in (
    sample_refdelta_seeds_2,
    sample_refdelta_seeds_3,
    sample_refdelta_sa_solver,
    sample_refdelta_sa_solver_pece,
):
    _function.__spectrum_interop_contract__ = SPECTRUM_BACKEND_INTEROP_CONTRACT


__all__ = [
    "REFDELTA_BASE_SAMPLERS",
    "prepare_refdelta_backend_adapters",
    "sample_refdelta_sa_solver",
    "sample_refdelta_sa_solver_pece",
    "sample_refdelta_seeds_2",
    "sample_refdelta_seeds_3",
]
