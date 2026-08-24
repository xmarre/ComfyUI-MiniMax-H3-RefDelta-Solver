from __future__ import annotations

import pytest

from tools.build_profile import build_profile


WEIGHTS = {
    "trajectory_risk": 1.0,
    "curvature": 0.25,
    "extrapolation_error": 0.25,
    "stochastic_pressure": 0.0,
    "instability_slope": 0.25,
}


def _records():
    return [
        {
            "sigma": sigma,
            "sigma_min": 0.0,
            "sigma_max": 1.0,
            "trajectory_risk": risk,
            "risk_components_video_curvature": risk / 2.0,
            "risk_components_video_extrapolation_error": risk / 3.0,
            "comparison_ref2va_video_x0_relative_error": comparison_error,
            "ref_delta_video_cosine": 1.0 - risk,
        }
        for sigma, risk, comparison_error in (
            (1.0, 0.0, 1000.0),
            (0.66, 0.2, 0.0),
            (0.33, 0.8, 0.0),
            (0.0, 0.1, 1000.0),
        )
    ]


def test_default_profile_is_neutral_and_reports_comparisons_without_using_them():
    profile = build_profile(
        _records(),
        profile_id="test",
        bins=4,
        experimental_stability_density=False,
        weights=WEIGHTS,
    )
    assert profile["status"] == "provisional-neutral"
    assert [point["difficulty"] for point in profile["points"]] == [1.0, 1.0]
    metadata = profile["metadata"]
    assert metadata["comparison_metrics_used_for_density"] is False
    assert metadata["comparison_diagnostics"][
        "comparison_ref2va_video_x0_relative_error"
    ] == pytest.approx(500.0)


def test_explicit_experimental_density_uses_only_production_stability():
    original = build_profile(
        _records(),
        profile_id="test",
        bins=4,
        experimental_stability_density=True,
        weights=WEIGHTS,
    )
    changed = _records()
    for record in changed:
        record["comparison_ref2va_video_x0_relative_error"] *= 10000.0
        record["ref_delta_video_cosine"] = -1.0
    altered = build_profile(
        changed,
        profile_id="test",
        bins=4,
        experimental_stability_density=True,
        weights=WEIGHTS,
    )
    assert original["status"] == "trajectory-stability-experimental"
    assert original["points"] == altered["points"]
    assert len({point["difficulty"] for point in original["points"]}) > 1


def test_experimental_density_requires_four_populated_bins():
    with pytest.raises(ValueError, match="four populated"):
        build_profile(
            _records()[:3],
            profile_id="test",
            bins=4,
            experimental_stability_density=True,
            weights=WEIGHTS,
        )
