from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfyui_refdelta_solver.scheduler import (
    FALLBACK_PROFILE_ID,
    calibrated_progress,
    load_profile,
    profile_from_dict,
    sigmas_from_profile,
)


def model_sampling():
    sigmas = torch.linspace(0.01, 1.0, 1000)
    return SimpleNamespace(audio_shift=3.0, sigmas=sigmas, sigma_max=sigmas[-1], sigma_min=sigmas[0])


@pytest.mark.parametrize("steps", (1, 2, 7, 20, 37, 101))
def test_scheduler_count_monotonic_terminal_and_deterministic(steps):
    profile = load_profile("r1024_provisional", fallback=False)
    first = sigmas_from_profile(model_sampling(), steps, 1.0, profile)
    second = sigmas_from_profile(model_sampling(), steps, 1.0, profile)
    assert first.shape == (steps + 1,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first[1:] <= first[:-1])
    assert first[-1].item() == 0.0
    assert first[0].item() <= 1.0


def test_scheduler_denoise_uses_tail_of_full_schedule():
    profile = load_profile("r1024_provisional", fallback=False)
    partial = sigmas_from_profile(model_sampling(), 10, 0.5, profile)
    full = sigmas_from_profile(model_sampling(), 20, 1.0, profile)
    torch.testing.assert_close(partial, full[-11:], rtol=0, atol=0)


def test_zero_denoise_returns_empty_sigmas():
    profile = load_profile("r1024_provisional", fallback=False)
    assert sigmas_from_profile(model_sampling(), 20, 0.0, profile).numel() == 0


def test_profile_loading_and_fallback(tmp_path: Path):
    assert load_profile("missing", search_directory=tmp_path).profile_id == FALLBACK_PROFILE_ID
    with pytest.raises(FileNotFoundError):
        load_profile("missing", search_directory=tmp_path, fallback=False)

    data = {
        "version": 1,
        "id": "test",
        "model_family": "MiniMax-H3 RefDelta",
        "rank": 512,
        "status": "test",
        "points": [
            {"progress": 0.0, "difficulty": 1.0},
            {"progress": 0.4, "difficulty": 3.0},
            {"progress": 1.0, "difficulty": 1.0},
        ],
    }
    (tmp_path / "test.json").write_text(json.dumps(data), encoding="utf-8")
    profile = load_profile("test", search_directory=tmp_path, fallback=False)
    assert profile.rank == 512
    progress = calibrated_progress(profile, 20)
    assert progress.shape == (21,)
    assert torch.all(progress[1:] >= progress[:-1])
    assert progress[0].item() == 0.0
    assert progress[-1].item() == 1.0


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda data: data.update(version=2), "version"),
        (lambda data: data.update(points=[]), "same length"),
        (lambda data: data["points"].insert(1, {"progress": 0.0, "difficulty": 1.0}), "strictly increasing"),
        (lambda data: data["points"][0].update(difficulty=0.0), "positive"),
    ),
)
def test_invalid_profiles_fail(mutation, match):
    data = {
        "version": 1,
        "id": "bad",
        "points": [
            {"progress": 0.0, "difficulty": 1.0},
            {"progress": 1.0, "difficulty": 1.0},
        ],
    }
    mutation(data)
    with pytest.raises(ValueError, match=match):
        profile_from_dict(data)


def test_scheduler_rejects_non_h3_sampling():
    profile = load_profile("r1024_provisional", fallback=False)
    with pytest.raises(ValueError, match="ModelSamplingAV"):
        sigmas_from_profile(SimpleNamespace(sigma_max=torch.tensor(1.0)), 20, 1.0, profile)
