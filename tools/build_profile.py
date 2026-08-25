from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def read_records(paths: list[Path]) -> list[dict[str, float | str | bool | None]]:
    records: list[dict[str, float | str | bool | None]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            raise ValueError(f"unsupported telemetry format: {path}")
        for row in rows:
            parsed: dict[str, float | str | bool | None] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                if isinstance(value, bool):
                    parsed[key] = value
                    continue
                if isinstance(value, str) and value.casefold() in {"true", "false"}:
                    parsed[key] = value.casefold() == "true"
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = str(value)
            records.append(parsed)
    return records


def deduplicate_production_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip diagnostic fields and deduplicate replay copies of production rows."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    for record in records:
        production = {
            key: value
            for key, value in record.items()
            if not key.startswith(("comparison_", "ref_delta_"))
        }
        identity = tuple(
            sorted(
                (key, type(value).__name__, repr(value))
                for key, value in production.items()
            )
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(production)
    return unique


def _finite_values(record: dict[str, Any], predicate) -> list[float]:
    return [
        float(value)
        for key, value in record.items()
        if predicate(key) and isinstance(value, float) and math.isfinite(value)
    ]


def _row_max(record: dict[str, Any], predicate) -> float:
    values = _finite_values(record, predicate)
    return max(values) if values else 0.0


def bin_stability_records(
    records: list[dict[str, Any]],
    bins: int,
    *,
    video_shift: float | None = None,
) -> list[dict[str, float]]:
    if bins < 1:
        raise ValueError("bins must be positive")
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
    for record in records:
        sigma = record.get("sigma")
        if not isinstance(sigma, float) or not math.isfinite(sigma):
            continue
        if video_shift is not None:
            denominator = video_shift + sigma * (1.0 - video_shift)
            if denominator <= 0.0:
                continue
            progress = 1.0 - sigma / denominator
        else:
            sigma_min = float(record.get("sigma_min", 0.0))
            sigma_max = float(record.get("sigma_max", 1.0))
            span = sigma_max - sigma_min
            progress = 1.0 - sigma if span <= 0.0 else (sigma_max - sigma) / span
        progress = min(1.0, max(0.0, progress))
        grouped[min(bins - 1, int(progress * bins))].append(record)

    points: list[dict[str, float]] = []
    for index, rows in enumerate(grouped):
        if not rows:
            continue
        trajectory = [
            _row_max(row, lambda key: "trajectory_risk" in key and not key.startswith("comparison_"))
            for row in rows
        ]
        curvature = [_row_max(row, lambda key: key.endswith("_curvature")) for row in rows]
        extrapolation = [
            _row_max(row, lambda key: key.endswith("_extrapolation_error"))
            for row in rows
        ]
        stochastic = [
            _row_max(
                row,
                lambda key: "stochastic_pressure" in key and not key.startswith("comparison_"),
            )
            for row in rows
        ]
        points.append(
            {
                "progress": (index + 0.5) / bins,
                "trajectory_risk": median(trajectory),
                "curvature": median(curvature),
                "extrapolation_error": median(extrapolation),
                "stochastic_pressure": median(stochastic),
                "samples": float(len(rows)),
            }
        )
    return points


def build_stability_density(
    points: list[dict[str, float]],
    *,
    trajectory_weight: float,
    curvature_weight: float,
    extrapolation_weight: float,
    stochastic_pressure_weight: float,
    instability_slope_weight: float,
) -> list[dict[str, float]]:
    if len(points) < 4:
        raise ValueError("experimental stability density needs at least four populated trajectory bins")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            trajectory_weight,
            curvature_weight,
            extrapolation_weight,
            stochastic_pressure_weight,
            instability_slope_weight,
        )
    ):
        raise ValueError("stability-density weights must be finite and non-negative")
    instability = [
        trajectory_weight * point["trajectory_risk"]
        + curvature_weight * point["curvature"]
        + extrapolation_weight * point["extrapolation_error"]
        + stochastic_pressure_weight * point["stochastic_pressure"]
        for point in points
    ]
    slopes: list[float] = []
    for index in range(len(points)):
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        span = points[right]["progress"] - points[left]["progress"]
        slopes.append(
            0.0 if span == 0.0 else abs(instability[right] - instability[left]) / span
        )
    raw = [
        1.0 + value + instability_slope_weight * slope
        for value, slope in zip(instability, slopes)
    ]
    mean = sum(raw) / len(raw)
    density = [min(4.0, max(0.25, value / mean)) for value in raw]
    output = [{"progress": 0.0, "difficulty": density[0]}]
    output.extend(
        {"progress": point["progress"], "difficulty": value}
        for point, value in zip(points, density)
    )
    output.append({"progress": 1.0, "difficulty": density[-1]})
    return output


def build_profile(
    records: list[dict[str, Any]],
    *,
    profile_id: str,
    bins: int,
    experimental_stability_density: bool,
    weights: dict[str, float],
    input_files: list[str] | None = None,
    shared_flow_density: bool = False,
    shared_flow_video_shift: float | None = None,
) -> dict[str, Any]:
    """Build a research scheduler profile from production telemetry only.

    Same-state FL2VA/Ref2VA comparison fields are deliberately stripped before
    binning. They remain diagnostic data and are not embedded in scheduler
    profiles or treated as a scheduler oracle.
    """
    if shared_flow_density and (
        shared_flow_video_shift is None
        or not math.isfinite(shared_flow_video_shift)
        or shared_flow_video_shift <= 0.0
    ):
        raise ValueError("shared-flow density requires a positive finite source video shift")
    production_records = deduplicate_production_records(records)
    points = bin_stability_records(
        production_records,
        bins,
        video_shift=shared_flow_video_shift if shared_flow_density else None,
    )
    if not points:
        raise ValueError("no finite sigma-bearing telemetry records were found")
    if experimental_stability_density:
        density = build_stability_density(
            points,
            trajectory_weight=weights["trajectory_risk"],
            curvature_weight=weights["curvature"],
            extrapolation_weight=weights["extrapolation_error"],
            stochastic_pressure_weight=weights["stochastic_pressure"],
            instability_slope_weight=weights["instability_slope"],
        )
        status = "trajectory-stability-experimental"
    else:
        density = [
            {"progress": 0.0, "difficulty": 1.0},
            {"progress": 1.0, "difficulty": 1.0},
        ]
        status = "neutral-compatibility"
    output_points = density
    if shared_flow_density:
        output_points = [
            {"progress": point["progress"], "density": point["difficulty"]}
            for point in density
        ]
    metadata = {
        "input_files": input_files or [],
        "input_records": len(records),
        "unique_production_records": len(production_records),
        "replayed_production_duplicates_removed": len(records) - len(production_records),
        "populated_bins": len(points),
        "density_source": (
            "production_trajectory_stability_research"
            if experimental_stability_density
            else "neutral_compatibility"
        ),
        "weights": weights if experimental_stability_density else {},
        "binned_production_stability": points,
        "comparison_metrics_used_for_density": False,
        "comparison_fields_embedded": False,
        "production_scheduler": "comfyui_basic_scheduler_beta",
        "production_use": False,
        "base_scheduler": (
            "uniform_linspace"
            if shared_flow_density
            else {"name": "beta", "alpha": 0.6, "beta": 0.6}
        ),
        "profile_semantics": (
            "immutable_offline_shared_base_time_density"
            if shared_flow_density
            else "legacy_beta_prior_density"
        ),
    }
    if shared_flow_density:
        metadata["evidence_source"] = (
            "production_actual_trajectory"
            if experimental_stability_density
            else "neutral_control"
        )
        metadata["source_video_shift"] = shared_flow_video_shift

    return {
        "version": 2 if shared_flow_density else 1,
        "id": profile_id,
        "model_family": "MiniMax-H3 Pruned Ref-Delta Fused",
        "rank": 1024,
        "status": status,
        "domain": (
            "shared-base-time-progress-density"
            if shared_flow_density
            else "video-sigma-progress-over-beta-prior"
        ),
        "points": output_points,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only RefDelta scheduler-profile builder. Production uses ComfyUI "
            "BasicScheduler beta; FL2VA/Ref2VA comparison fields are stripped and never "
            "used as scheduler evidence."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id", default="r1024_scheduler_research")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--experimental-stability-density", action="store_true")
    parser.add_argument(
        "--shared-flow-density",
        action="store_true",
        help="Emit version-2 density for MiniMaxH3UniformFlowScheduler curvature_profile mode.",
    )
    parser.add_argument(
        "--video-shift",
        type=float,
        help="Source run's video shift; required with --shared-flow-density.",
    )
    parser.add_argument("--trajectory-weight", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=0.25)
    parser.add_argument("--extrapolation-weight", type=float, default=0.25)
    parser.add_argument("--stochastic-pressure-weight", type=float, default=0.0)
    parser.add_argument("--instability-slope-weight", type=float, default=0.25)
    args = parser.parse_args()

    records = read_records(args.inputs)
    weights = {
        "trajectory_risk": args.trajectory_weight,
        "curvature": args.curvature_weight,
        "extrapolation_error": args.extrapolation_weight,
        "stochastic_pressure": args.stochastic_pressure_weight,
        "instability_slope": args.instability_slope_weight,
    }
    profile = build_profile(
        records,
        profile_id=args.id,
        bins=args.bins,
        experimental_stability_density=args.experimental_stability_density,
        weights=weights,
        input_files=[str(path) for path in args.inputs],
        shared_flow_density=args.shared_flow_density,
        shared_flow_video_shift=args.video_shift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
