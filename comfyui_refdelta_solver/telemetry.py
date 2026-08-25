from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import torch


_SAFE_PREFIX = re.compile(r"[^A-Za-z0-9_.-]+")


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("telemetry only accepts scalar tensors")
        return float(value.detach().cpu())
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported telemetry value {type(value).__name__}")


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_record(value, name))
        else:
            flat[name] = scalar(value)
    return flat


class TelemetryWriter:
    def __init__(self, output_directory: Path, prefix: str, seed: int | None) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        safe_prefix = _SAFE_PREFIX.sub("_", prefix).strip("._") or "refdelta_trajectory"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        seed_part = "unknown" if seed is None else str(seed)
        base = output_directory / f"{safe_prefix}-{stamp}-seed{seed_part}"
        suffix = 1
        while base.with_suffix(".jsonl").exists() or base.with_suffix(".csv").exists():
            base = output_directory / f"{safe_prefix}-{stamp}-seed{seed_part}-{suffix}"
            suffix += 1
        self.jsonl_path = base.with_suffix(".jsonl")
        self.csv_path = base.with_suffix(".csv")
        self.stability_map_directory = base.parent / f"{base.name}-stability-maps"
        self._records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self._records.append(flatten_record(record))

    def write_stability_maps(
        self,
        step: int,
        sigma: torch.Tensor,
        temporal_motion_ratio: torch.Tensor,
        diffusion_change_ratio: torch.Tensor | None,
        restore_mask: torch.Tensor,
        applied_video_gate: torch.Tensor,
        metadata: dict[str, Any],
    ) -> Path:
        """Persist compact channel-reduced actual-step maps only on explicit request."""
        import numpy as np

        self.stability_map_directory.mkdir(parents=True, exist_ok=True)
        path = self.stability_map_directory / f"step-{step:04d}.npz"

        def compact(value: torch.Tensor) -> Any:
            return value.detach().to(device="cpu", dtype=torch.float16).numpy()

        payload = {
            "temporal_motion_ratio": compact(temporal_motion_ratio),
            "restore_mask": compact(restore_mask),
            "applied_video_gate": compact(applied_video_gate),
            "metadata_json": json.dumps(
                {
                    **metadata,
                    "step": step,
                    "sigma": float(sigma.detach().cpu()),
                    "video_shape": list(restore_mask.shape),
                },
                sort_keys=True,
                allow_nan=False,
            ),
        }
        if diffusion_change_ratio is not None:
            payload["diffusion_change_ratio"] = compact(diffusion_change_ratio)
        np.savez_compressed(path, **payload)
        return path

    def close(self) -> None:
        if not self._records:
            return
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        fields = sorted({key for record in self._records for key in record})
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._records)
