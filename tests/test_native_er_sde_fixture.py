from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

import comfyui_refdelta_solver.sampler as sampler_module
from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde
from comfyui_refdelta_solver.spectrum_interop import (
    SPECTRUM_BRIDGE_KEY,
    SPECTRUM_INTEROP_CONTRACT,
)
from comfyui_refdelta_solver.trajectory import TrajectoryHistory


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
        patcher = SimpleNamespace(get_model_object=lambda name: sampling if name == "model_sampling" else None)
        packed_model = SimpleNamespace(latent_shapes=[(1, 1, 1, 1, 2), (1, 2)])
        self.inner_model = SimpleNamespace(model_patcher=patcher, inner_model=packed_model)

    def __call__(self, state, sigma, **_extra_args):
        return state * 0.125 + sigma.reshape(-1, 1) * 0.03125


def _fixed_noise(state):
    return lambda _sigma, _sigma_next: torch.full_like(state, 0.125)


def _clone_ratios(ratios):
    if ratios is None:
        return None
    return {name: value.detach().clone() for name, value in ratios.items()}


def test_instrumented_no_adaptation_matches_current_native_er_sde(tmp_path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)

    native = k_sampling.sample_er_sde(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        max_stage=3,
    )
    instrumented = sample_refdelta_er_sde(
        _Model(),
        initial.clone(),
        sigmas,
        extra_args={"seed": 7},
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        max_stage=3,
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_adaptation_strength=0.0,
            trajectory_correction=False,
            telemetry=True,
            telemetry_prefix="native-fixture",
        ),
    )

    torch.testing.assert_close(instrumented, native, rtol=2e-6, atol=2e-7)
    assert len(list((tmp_path / "refdelta_telemetry").glob("native-fixture-*.jsonl"))) == 1


def test_spectrum_forecasts_never_commit_to_refdelta_evidence_history(monkeypatch):
    histories = []

    class SpyHistory(TrajectoryHistory):
        def __init__(self):
            super().__init__()
            self.committed_coordinates = []
            histories.append(self)

        def commit(self, raw, coordinate, first):
            self.committed_coordinates.append(coordinate)
            return super().commit(raw, coordinate, first)

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            return step_id != 2

        @staticmethod
        def publish_stochastic_increment(_step_id, _increment):
            raise AssertionError("deterministic fixture must not publish noise")

    monkeypatch.setattr(sampler_module, "TrajectoryHistory", SpyHistory)
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.72, 0.47, 0.23, 0.0], dtype=torch.float32)
    sample_refdelta_er_sde(
        _Model(),
        initial,
        sigmas,
        extra_args={
            "seed": 7,
            "model_options": {
                "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
            },
        },
        disable=True,
        s_noise=0.0,
        config=RefDeltaSamplerConfig(),
    )

    assert len(histories) == 2
    solver_history, evidence_history = histories
    assert len(solver_history.committed_coordinates) == 3
    assert len(evidence_history.committed_coordinates) == 2
    assert solver_history.committed_coordinates[2] not in evidence_history.committed_coordinates


def test_spectrum_forecast_preserves_last_actual_stochastic_evidence(monkeypatch):
    histories = []
    published_steps = []

    class SpyHistory(TrajectoryHistory):
        def __init__(self):
            super().__init__()
            self.stochastic_ratios_seen = []
            self.stochastic_ratios_after_commit = []
            histories.append(self)

        def observe(self, *args, **kwargs):
            self.stochastic_ratios_seen.append(_clone_ratios(self.previous_stochastic_ratios))
            return super().observe(*args, **kwargs)

        def commit(self, raw, coordinate, first):
            result = super().commit(raw, coordinate, first)
            self.stochastic_ratios_after_commit.append(
                _clone_ratios(self.previous_stochastic_ratios)
            )
            return result

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            return step_id != 2

        @staticmethod
        def publish_stochastic_increment(step_id, _increment):
            published_steps.append(step_id)

    monkeypatch.setattr(sampler_module, "TrajectoryHistory", SpyHistory)
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.82, 0.64, 0.46, 0.28, 0.0], dtype=torch.float32)
    sample_refdelta_er_sde(
        _Model(),
        initial,
        sigmas,
        extra_args={
            "seed": 7,
            "model_options": {
                "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
            },
        },
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_adaptation_strength=0.5,
            minimum_stochastic_multiplier=0.5,
            trajectory_correction=False,
        ),
    )

    assert len(histories) == 2
    _, evidence_history = histories
    # Step 1 establishes native stochastic/movement evidence. Step 2 is a
    # Spectrum forecast, so step 3 must consume exactly the evidence committed
    # by step 1 rather than any forecast-derived replacement.
    assert len(evidence_history.stochastic_ratios_seen) >= 3
    assert len(evidence_history.stochastic_ratios_after_commit) >= 2
    expected_last_actual = evidence_history.stochastic_ratios_after_commit[1]
    after_forecast = evidence_history.stochastic_ratios_seen[2]
    assert expected_last_actual is not None
    assert after_forecast is not None
    assert set(after_forecast) == {"video", "audio"}
    assert set(after_forecast) == set(expected_last_actual)
    for name in after_forecast:
        assert torch.equal(after_forecast[name], expected_last_actual[name])
        assert torch.isfinite(after_forecast[name]) and after_forecast[name] > 0
    assert 2 in published_steps


def test_spectrum_forecast_applies_correction_from_actual_only_evidence(monkeypatch):
    histories = []
    correction_calls = []
    original_correction = sampler_module.bounded_trajectory_correction

    class SpyHistory(TrajectoryHistory):
        def __init__(self):
            super().__init__()
            self.commits = []
            self.observed_firsts = []
            histories.append(self)

        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            self.observed_firsts.append(
                None
                if observation.first is None
                else observation.first.detach().clone()
            )
            return observation

        def commit(self, raw, coordinate, first):
            self.commits.append(
                {
                    "raw": raw.detach().clone(),
                    "first": None if first is None else first.detach().clone(),
                }
            )
            return super().commit(raw, coordinate, first)

    def correction_spy(
        raw,
        previous_raw,
        first,
        second,
        next_span,
        layout,
        video_strength,
        audio_strength,
        bound,
        gate,
    ):
        correction_calls.append(
            {
                "previous_raw": None if previous_raw is None else previous_raw.detach().clone(),
                "first": None if first is None else first.detach().clone(),
                "second": None if second is None else second.detach().clone(),
            }
        )
        return original_correction(
            raw,
            previous_raw,
            first,
            second,
            next_span,
            layout,
            video_strength,
            audio_strength,
            bound,
            gate,
        )

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            return step_id != 2

        @staticmethod
        def publish_stochastic_increment(_step_id, _increment):
            raise AssertionError("deterministic fixture must not publish noise")

    monkeypatch.setattr(sampler_module, "TrajectoryHistory", SpyHistory)
    monkeypatch.setattr(
        sampler_module,
        "bounded_trajectory_correction",
        correction_spy,
    )
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.82, 0.64, 0.46, 0.28, 0.0], dtype=torch.float32)
    sample_refdelta_er_sde(
        _Model(),
        initial,
        sigmas,
        extra_args={
            "seed": 7,
            "model_options": {
                "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
            },
        },
        disable=True,
        s_noise=0.0,
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_adaptation_strength=0.0,
            trajectory_correction=True,
        ),
    )

    assert len(histories) == 2
    _, evidence_history = histories
    # Nonterminal calls are steps 0, 1, forecast 2, and actual 3. The forecast
    # must receive the exact raw/first pair committed by actual step 1. The
    # following actual step must still use step 1 as previous_raw, proving the
    # forecast never replaced the actual evidence anchor.
    assert len(correction_calls) == 4
    assert len(evidence_history.commits) == 3
    assert len(evidence_history.observed_firsts) >= 3
    last_actual_before_forecast = evidence_history.commits[1]

    forecast_call = correction_calls[2]
    assert forecast_call["previous_raw"] is not None
    assert forecast_call["first"] is not None
    torch.testing.assert_close(
        forecast_call["previous_raw"],
        last_actual_before_forecast["raw"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        forecast_call["first"],
        last_actual_before_forecast["first"],
        rtol=0,
        atol=0,
    )

    post_forecast_actual_call = correction_calls[3]
    assert post_forecast_actual_call["previous_raw"] is not None
    assert post_forecast_actual_call["first"] is not None
    torch.testing.assert_close(
        post_forecast_actual_call["previous_raw"],
        last_actual_before_forecast["raw"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        post_forecast_actual_call["first"],
        evidence_history.observed_firsts[2],
        rtol=0,
        atol=0,
    )


def test_spatial_controller_updates_only_actuals_and_publishes_exact_applied_increment(monkeypatch):
    original_controller = sampler_module.StochasticStabilityController
    controller_updates = []
    applied_increments = []
    published = []

    class SpyController(original_controller):
        def update_actual(self, packed_denoised, previous_actual, step_index):
            controller_updates.append(step_index)
            return super().update_actual(packed_denoised, previous_actual, step_index)

        def apply(self, *args, **kwargs):
            result = super().apply(*args, **kwargs)
            applied_increments.append(result.increment)
            return result

    class Bridge:
        api_version = 1
        interop_contract = SPECTRUM_INTEROP_CONTRACT

        @staticmethod
        def model_result_is_actual(step_id):
            return step_id != 2

        @staticmethod
        def publish_stochastic_increment(step_id, increment):
            published.append((step_id, increment))

    monkeypatch.setattr(sampler_module, "StochasticStabilityController", SpyController)
    initial = torch.tensor([[0.2, -0.1, 0.4, -0.3]], dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.82, 0.64, 0.46, 0.28, 0.0], dtype=torch.float32)
    sample_refdelta_er_sde(
        _Model(),
        initial,
        sigmas,
        extra_args={
            "seed": 7,
            "model_options": {
                "transformer_options": {SPECTRUM_BRIDGE_KEY: Bridge()}
            },
        },
        disable=True,
        s_noise=0.8,
        noise_sampler=_fixed_noise(initial),
        config=RefDeltaSamplerConfig(
            adaptive_order=False,
            stochastic_control_mode="spatiotemporal_stability",
            stochastic_adaptation_strength=0.75,
            minimum_stochastic_multiplier=0.1,
            video_stability_spatial_radius=0,
            video_stability_temporal_radius=0,
            video_stability_ema=0.0,
            video_stability_start_fraction=0.0,
            video_stability_full_fraction=0.0,
        ),
    )

    assert 2 not in controller_updates
    assert {0, 1, 3, 4} <= set(controller_updates)
    assert [step for step, _ in published] == [0, 1, 2, 3]
    assert len(applied_increments) == len(published)
    for applied, (_, sent) in zip(applied_increments, published, strict=True):
        assert sent is applied
