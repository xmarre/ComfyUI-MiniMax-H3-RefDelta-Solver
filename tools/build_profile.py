from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


REFERENCE_FIELDS = (
    "reference_video_x0_relative_error",
    "reference_video_velocity_relative_error",
    "reference_audio_x0_relative_error",
    "reference_audio_velocity_relative_error",
)


def read_records(paths: list[Path]) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
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
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = str(value)
            records.append(parsed)
    return records


def bin_records(records: list[dict[str, float | str]], bins: int, require_reference: bool) -> list[dict[str, float]]:
    grouped: list[list[dict[str, float | str]]] = [[] for _ in range(bins)]
    for record in records:
        sigma = record.get("sigma")
        if not isinstance(sigma, float) or not math.isfinite(sigma):
            continue
        sigma_min = float(record.get("sigma_min", 0.0))
        sigma_max = float(record.get("sigma_max", 1.0))
        span = sigma_max - sigma_min
        progress = 1.0 - sigma if span <= 0.0 else (sigma_max - sigma) / span
        progress = min(1.0, max(0.0, progress))
        index = min(bins - 1, int(progress * bins))
        grouped[index].append(record)

    result: list[dict[str, float]] = []
    for index, rows in enumerate(grouped):
        if not rows:
            continue
        reference_values = [
            max(float(row[field]) for field in REFERENCE_FIELDS if isinstance(row.get(field), float))
            for row in rows
            if any(isinstance(row.get(field), float) for field in REFERENCE_FIELDS)
        ]
        if require_reference and not reference_values:
            continue
        curvature_values = [
            max(float(value) for key, value in row.items() if key.endswith("_curvature") and isinstance(value, float))
            for row in rows
            if any(key.endswith("_curvature") and isinstance(value, float) for key, value in row.items())
        ]
        result.append({
            "progress": (index + 0.5) / bins,
            "error": median(reference_values) if reference_values else 0.0,
            "curvature": median(curvature_values) if curvature_values else 0.0,
            "samples": float(len(rows)),
        })
    return result


def build_density(points: list[dict[str, float]], alpha: float, beta: float, gamma: float) -> list[dict[str, float]]:
    if len(points) < 4:
        raise ValueError("calibration needs at least four populated trajectory bins")
    errors = [point["error"] for point in points]
    slopes = []
    for index, point in enumerate(points):
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        span = points[right]["progress"] - points[left]["progress"]
        slopes.append(0.0 if span == 0.0 else abs(errors[right] - errors[left]) / span)
    raw = [
        1.0 + alpha * point["error"] + beta * slope + gamma * point["curvature"]
        for point, slope in zip(points, slopes)
    ]
    mean = sum(raw) / len(raw)
    density = [min(4.0, max(0.25, value / mean)) for value in raw]
    output = [{"progress": 0.0, "difficulty": density[0]}]
    output.extend({"progress": point["progress"], "difficulty": value} for point, value in zip(points, density))
    output.append({"progress": 1.0, "difficulty": density[-1]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RefDelta scheduler profile from scalar trajectory telemetry.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id", default="r1024_calibrated")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0, help="Reference-error density weight")
    parser.add_argument("--beta", type=float, default=0.25, help="Error-slope density weight")
    parser.add_argument("--gamma", type=float, default=0.25, help="Trajectory-curvature density weight")
    parser.add_argument("--allow-no-reference", action="store_true")
    args = parser.parse_args()

    records = read_records(args.inputs)
    points = bin_records(records, args.bins, require_reference=not args.allow_no_reference)
    density = build_density(points, args.alpha, args.beta, args.gamma)
    profile = {
        "version": 1,
        "id": args.id,
        "model_family": "MiniMax-H3 Pruned Ref-Delta Fused",
        "rank": 1024,
        "status": "calibrated" if not args.allow_no_reference else "trajectory-only-experimental",
        "domain": "video-sigma-progress-over-beta-prior",
        "points": density,
        "metadata": {
            "input_files": [str(path) for path in args.inputs],
            "input_records": len(records),
            "populated_bins": len(points),
            "weights": {"model_error": args.alpha, "error_derivative": args.beta, "trajectory_curvature": args.gamma},
            "base_scheduler": {"name": "beta", "alpha": 0.6, "beta": 0.6},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
