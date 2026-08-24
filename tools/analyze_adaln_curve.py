#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(description="Test whether AdaLN table curvature correlates with matched reference error telemetry.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("telemetry_csv", type=Path)
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        table = handle.get_tensor("adaln_t_table").float()
    first = table[1:] - table[:-1]
    curvature = torch.linalg.vector_norm(first[1:] - first[:-1], dim=1)
    curvature = torch.cat((curvature[:1], curvature, curvature[-1:]))

    progress, errors = [], []
    with args.telemetry_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates = [row.get("reference_video_x0_relative_error"), row.get("reference_video_velocity_relative_error")]
            values = [float(value) for value in candidates if value not in (None, "") and math.isfinite(float(value))]
            if not values:
                continue
            progress.append(min(1.0, max(0.0, 1.0 - float(row["sigma"]))))
            errors.append(max(values))
    if len(errors) < 8:
        raise ValueError("at least eight matched reference telemetry points are required")
    indices = torch.tensor(progress).mul(table.shape[0] - 1).round().long().clamp(0, table.shape[0] - 1)
    coefficient = pearson(curvature.index_select(0, indices), torch.tensor(errors))
    print(f"points={len(errors)} pearson_adaln_curvature_vs_reference_error={coefficient:.8f}")


if __name__ == "__main__":
    main()
