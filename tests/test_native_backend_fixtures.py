from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler_backends import (
    _RefDeltaBackendState,
    sample_refdelta_sa_solver,
    sample_refdelta_sa_solver_pece,
    sample_refdelta_seeds_2,
    sample_refdelta_seeds_3,
)
from comfyui_refdelta_solver.spectrum_interop import (
    SPECTRUM_BACKEND_INTEROP_CONTRACT,
    SPECTRUM_BRIDGE_KEY,
    SpectrumInteropError,
)


def _required_comfy_module(name: str):
    if os.environ.get("COMFYUI_PATH"):
        return importlib.import_module(name)
    return pytest.importorskip(name)


k_sampling = _required_comfy_module("comfy.k_diffusion.sampling")
model_sampling_module = _required_comfy_module("comfy.model_sampling")


class _Sampling(model_sampling_module.CONST, model_sampling_module.ModelSamplingAV):
    pass


class _Model:
    def __init__(self):
        sampling = _Sampling()
        sampling.set_parameters(shift=12.0, audio_shift=3.0)
        patcher = SimpleNamespace(
            get_model_object=lambda name: sampling if name == "model_sampling" else None
        )
        packed_model = SimpleNamespace(latent_shapes=[(1, 1, 1, 1, 2), (1, 2)])
        self.inner_model = SimpleNamespace(
            model_patcher=patcher,
            inner_model=packed_model,
        )

    def __call__(self, state, sigma, **_extra_args):
        return state * 0.125 + sigma.reshape(-1, 1) * 0.03125


def _fixed_noise(state):
    return lambda _sigma, _sigma_next: torch.full_like(state, 0.125)


@pytest.mark.parametrize(
    ("native_name", "refdelta", "options"),
    (
        (
            "sample_seeds_2",
            sample_refdelta_seeds_2,
            {"eta": 1.0, "s_noise": 0.8, "r": 0.5, "solver_type": "phi_1"},
        ),
        (
            "sample_seeds_3",
            sample_refdelta_seeds_3,
            {"eta": 1.0, "s_noise": 0.8, "r_1": 1.0 / 3.0, "r_2": 2.0 / 3.0},
        ),
        (
            "sample_sa_solver",
            sample_refdelta_sa_solver,
            {
                "s_noise": 0.8,
                "predictor_order": 3,
                "corrector_order": 4,
                "use_pece": False,
                "simple_order_2": False,
            },
        ),
        (
            "sample_sa_solver",
            sample_refdelta_sa_solver_pece,
            {
                "s_noise": 0.8,
                "predictor_order": 3,
                "corrector_order": 4,
                "use_pece": True,
                "simple_order_2": False,
            },
        ),
    ),
)
def test_instrumented_zero_adaptation_matches_native_backend(
    native_name,
    refdelta,
    options,
):
    native = getattr(k_sampling, native_name, None)
    if native is None:
        pytest.skip(f"reviewed ComfyUI revision does not provide {native_name}")

    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)
    noise_sampler = _fixed_noise(initial)
    native_options = dict(options)
    if native_name == "sample_sa_solver":
        native_options["noise_sampler"] = noise_sampler
    else:
        native_options["noise_sampler"] = noise_sampler

    expected = native(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        **native_options,
    )

    refdelta_options = dict(options)
    refdelta_options.pop("use_pece", None)
    actual = refdelta(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        noise_sampler=_fixed_noise(initial),
        config=RefDeltaSamplerConfig(
            adaptive_order=True,
            stochastic_adaptation_strength=0.0,
            trajectory_correction=False,
            telemetry=False,
        ),
        **refdelta_options,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)



def test_multi_backend_evidence_coordinate_matches_released_refdelta_er_lambda():
    model = _Model()
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)
    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    prepared = k_sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    expected = k_sampling.sigma_to_half_log_snr(
        prepared,
        model_sampling=model_sampling,
    ).neg().exp()

    state = _RefDeltaBackendState(
        model,
        initial,
        sigmas,
        {"seed": 7},
        RefDeltaSamplerConfig(),
        "seeds_2",
        2,
    )
    try:
        assert state.coordinates == expected.detach().float().cpu().tolist()
    finally:
        state.finish()



def test_spectrum_bridge_keeps_seeds_evidence_outer_actual_only():
    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_BACKEND_INTEROP_CONTRACT

        def __init__(self):
            self.calls = []

        def model_result_is_actual(self, step_id):
            self.calls.append(step_id)
            return {0: True, 1: False, 2: True, 3: True, 4: False}[step_id]

    bridge = Bridge()
    extra_args = {
        "model_options": {
            "transformer_options": {
                SPECTRUM_BRIDGE_KEY: bridge,
            }
        }
    }
    model = _Model()
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    state = _RefDeltaBackendState(
        model,
        initial,
        torch.tensor([1.0, 0.6, 0.2, 0.0], dtype=torch.float32),
        extra_args,
        RefDeltaSamplerConfig(
            stochastic_control_mode="spatiotemporal_stability",
            trajectory_correction=False,
            telemetry=False,
        ),
        "seeds_2",
        2,
    )
    try:
        state.model_call(initial, torch.tensor([1.0]))
        state.model_call(initial + 1.0, torch.tensor([0.8]))
        second_outer = state.model_call(initial + 2.0, torch.tensor([0.6]))
        state.model_call(initial + 3.0, torch.tensor([0.4]))
        state.model_call(initial + 4.0, torch.tensor([0.2]))

        assert bridge.calls == [0, 1, 2, 3, 4]
        torch.testing.assert_close(state.history.previous_raw, second_outer, rtol=0, atol=0)
        assert state.history.previous_coordinate == state.coordinates[1]
        assert state.controller.evidence is not None
        assert state.controller.evidence.source_actual_step == 1
    finally:
        state.finish()



def test_spectrum_bridge_pece_uses_corrected_calls_as_persistent_evidence():
    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_BACKEND_INTEROP_CONTRACT

        def __init__(self):
            self.calls = []

        def model_result_is_actual(self, step_id):
            self.calls.append(step_id)
            return {0: True, 1: False, 2: True, 3: False, 4: True}[step_id]

    bridge = Bridge()
    extra_args = {
        "model_options": {
            "transformer_options": {
                SPECTRUM_BRIDGE_KEY: bridge,
            }
        }
    }
    model = _Model()
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    state = _RefDeltaBackendState(
        model,
        initial,
        torch.tensor([1.0, 0.6, 0.2, 0.0], dtype=torch.float32),
        extra_args,
        RefDeltaSamplerConfig(
            stochastic_control_mode="spatiotemporal_stability",
            trajectory_correction=False,
            telemetry=False,
        ),
        "sa_solver_pece",
        1,
    )
    try:
        p0 = state.model_call(initial, torch.tensor([1.0]))
        torch.testing.assert_close(state.history.previous_raw, p0, rtol=0, atol=0)
        assert state.history.previous_coordinate == state.coordinates[0]

        state.model_call(initial + 1.0, torch.tensor([0.6]))  # P1 forecast
        torch.testing.assert_close(state.history.previous_raw, p0, rtol=0, atol=0)

        c1 = state.model_call(initial + 2.0, torch.tensor([0.6]))
        torch.testing.assert_close(state.history.previous_raw, c1, rtol=0, atol=0)
        assert state.history.previous_coordinate == state.coordinates[1]

        state.model_call(initial + 3.0, torch.tensor([0.2]))  # P2 forecast
        torch.testing.assert_close(state.history.previous_raw, c1, rtol=0, atol=0)

        c2 = state.model_call(initial + 4.0, torch.tensor([0.2]))
        torch.testing.assert_close(state.history.previous_raw, c2, rtol=0, atol=0)
        assert state.history.previous_coordinate == state.coordinates[2]
        assert state.controller.evidence is not None
        assert state.controller.evidence.source_actual_step == 2
        assert bridge.calls == [0, 1, 2, 3, 4]
    finally:
        state.finish()


def test_spectrum_bridge_pece_rejects_forecasted_corrected_endpoint():
    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_BACKEND_INTEROP_CONTRACT

        def model_result_is_actual(self, step_id):
            return {0: True, 1: False, 2: False}[step_id]

    extra_args = {
        "model_options": {
            "transformer_options": {
                SPECTRUM_BRIDGE_KEY: Bridge(),
            }
        }
    }
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    state = _RefDeltaBackendState(
        _Model(),
        initial,
        torch.tensor([1.0, 0.6, 0.0], dtype=torch.float32),
        extra_args,
        RefDeltaSamplerConfig(trajectory_correction=False, telemetry=False),
        "sa_solver_pece",
        1,
    )
    try:
        state.model_call(initial, torch.tensor([1.0]))
        state.model_call(initial + 1.0, torch.tensor([0.6]))
        with pytest.raises(
            SpectrumInteropError,
            match="persistent SA-Solver PECE endpoint",
        ):
            state.model_call(initial + 2.0, torch.tensor([0.6]))
    finally:
        state.finish()


@pytest.mark.parametrize(
    ("refdelta", "use_pece"),
    (
        (sample_refdelta_sa_solver, False),
        (sample_refdelta_sa_solver_pece, True),
    ),
)
def test_refdelta_sa_preserves_native_noise_sampler_call_topology(
    refdelta,
    use_pece,
):
    native = getattr(k_sampling, "sample_sa_solver", None)
    if native is None:
        pytest.skip("reviewed ComfyUI revision does not provide sample_sa_solver")

    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)

    class Recorder:
        def __init__(self):
            self.calls = []

        def __call__(self, sigma, sigma_next):
            self.calls.append((float(sigma), float(sigma_next)))
            return torch.full_like(initial, 0.125)

    native_noise = Recorder()
    refdelta_noise = Recorder()
    native(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 11},
        disable=True,
        noise_sampler=native_noise,
        s_noise=0.8,
        predictor_order=3,
        corrector_order=4,
        use_pece=use_pece,
        simple_order_2=False,
    )
    refdelta(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 11},
        disable=True,
        noise_sampler=refdelta_noise,
        s_noise=0.8,
        predictor_order=3,
        corrector_order=4,
        simple_order_2=False,
        config=RefDeltaSamplerConfig(
            adaptive_order=True,
            stochastic_adaptation_strength=0.75,
            trajectory_correction=False,
            telemetry=False,
        ),
    )

    assert refdelta_noise.calls == native_noise.calls


@pytest.mark.parametrize(
    ("native_name", "refdelta", "options"),
    (
        (
            "sample_seeds_2",
            sample_refdelta_seeds_2,
            {"eta": 1.0, "s_noise": 0.8, "r": 0.5, "solver_type": "phi_1"},
        ),
        (
            "sample_seeds_3",
            sample_refdelta_seeds_3,
            {"eta": 1.0, "s_noise": 0.8, "r_1": 1.0 / 3.0, "r_2": 2.0 / 3.0},
        ),
    ),
)
def test_refdelta_seeds_preserves_native_noise_sampler_call_topology(
    native_name,
    refdelta,
    options,
):
    native = getattr(k_sampling, native_name, None)
    if native is None:
        pytest.skip(f"reviewed ComfyUI revision does not provide {native_name}")

    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)

    class Recorder:
        def __init__(self):
            self.calls = []

        def __call__(self, sigma, sigma_next):
            self.calls.append((float(sigma), float(sigma_next)))
            return torch.full_like(initial, 0.125)

    native_noise = Recorder()
    refdelta_noise = Recorder()
    native(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 11},
        disable=True,
        noise_sampler=native_noise,
        **options,
    )
    refdelta(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 11},
        disable=True,
        noise_sampler=refdelta_noise,
        config=RefDeltaSamplerConfig(
            adaptive_order=True,
            stochastic_adaptation_strength=0.75,
            trajectory_correction=False,
            telemetry=False,
        ),
        **options,
    )

    assert refdelta_noise.calls == native_noise.calls
