from __future__ import annotations

import pytest

from tools.build_profile import build_profile, read_records


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


def test_default_profile_is_neutral_compatibility_and_strips_comparison_data():
    profile = build_profile(
        _records(),
        profile_id="test",
        bins=4,
        experimental_stability_density=False,
        weights=WEIGHTS,
    )
    assert profile["status"] == "neutral-compatibility"
    assert [point["difficulty"] for point in profile["points"]] == [1.0, 1.0]
    metadata = profile["metadata"]
    assert metadata["comparison_metrics_used_for_density"] is False
    assert metadata["comparison_fields_embedded"] is False
    assert metadata["production_scheduler"] == "comfyui_basic_scheduler_beta"
    assert metadata["production_use"] is False
    assert not any(key.startswith("comparison_") for key in metadata)
    assert not any(key.startswith("ref_delta_") for key in metadata)


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
    assert original["metadata"]["production_use"] is False


def test_replay_copies_do_not_duplicate_production_stability_evidence():
    captured = _records()
    fl2va = [
        dict(record, comparison_label="fl2va", comparison_fl2va_video_x0_cosine=0.9)
        for record in captured
    ]
    ref2va = [
        dict(record, comparison_label="ref2va", comparison_ref2va_video_x0_cosine=0.8)
        for record in captured
    ]
    baseline = build_profile(
        captured,
        profile_id="baseline",
        bins=4,
        experimental_stability_density=True,
        weights=WEIGHTS,
    )
    combined = build_profile(
        captured + fl2va + ref2va,
        profile_id="combined",
        bins=4,
        experimental_stability_density=True,
        weights=WEIGHTS,
    )

    assert combined["points"] == baseline["points"]
    metadata = combined["metadata"]
    assert metadata["input_records"] == 12
    assert metadata["unique_production_records"] == 4
    assert metadata["replayed_production_duplicates_removed"] == 8
    assert (
        sum(point["samples"] for point in metadata["binned_production_stability"])
        == 4
    )
    assert metadata["comparison_fields_embedded"] is False


def test_read_records_normalizes_csv_booleans(tmp_path):
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "sigma,actual_model_evaluation,comparison_model_evaluated\n1.0,True,false\n",
        encoding="utf-8",
    )

    assert read_records([path]) == [
        {
            "sigma": 1.0,
            "actual_model_evaluation": True,
            "comparison_model_evaluated": False,
        }
    ]


def test_experimental_density_requires_four_populated_bins():
    with pytest.raises(ValueError, match="four populated"):
        build_profile(
            _records()[:3],
            profile_id="test",
            bins=4,
            experimental_stability_density=True,
            weights=WEIGHTS,
        )


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), float("-inf"), -0.1))
def test_experimental_density_rejects_nonfinite_or_negative_weights(invalid):
    weights = dict(WEIGHTS)
    weights["trajectory_risk"] = invalid
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_profile(
            _records(),
            profile_id="test",
            bins=4,
            experimental_stability_density=True,
            weights=weights,
        )
