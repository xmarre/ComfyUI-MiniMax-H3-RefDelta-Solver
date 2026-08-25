from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from safetensors import safe_open


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report correlation between AdaLN table curvature and one named telemetry field."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("telemetry_csv", type=Path)
    parser.add_argument(
        "--metric",
        default="trajectory_risk",
        help="Scalar telemetry column to correlate (default: production trajectory_risk)",
    )
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        table = handle.get_tensor("adaln_t_table").float()
    first = table[1:] - table[:-1]
    curvature = torch.linalg.vector_norm(first[1:] - first[:-1], dim=1)
    curvature = torch.cat((curvature[:1], curvature, curvature[-1:]))

    progress, measurements = [], []
    with args.telemetry_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(args.metric)
            if value in (None, "") or not math.isfinite(float(value)):
                continue
            progress.append(min(1.0, max(0.0, 1.0 - float(row["sigma"]))))
            measurements.append(float(value))
    if len(measurements) < 8:
        raise ValueError(f"at least eight finite telemetry points are required for {args.metric!r}")
    indices = torch.tensor(progress).mul(table.shape[0] - 1).round().long().clamp(0, table.shape[0] - 1)
    coefficient = pearson(curvature.index_select(0, indices), torch.tensor(measurements))
    print(f"points={len(measurements)} metric={args.metric} pearson={coefficient:.8f}")


if __name__ == "__main__":
    main()
