from __future__ import annotations

import csv
import json

import torch

from comfyui_refdelta_solver.telemetry import TelemetryWriter, flatten_record


def test_flatten_record_accepts_only_scalars():
    flat = flatten_record({"step": 1, "video": {"risk": torch.tensor(0.25)}})
    assert flat == {"step": 1, "video_risk": 0.25}


def test_writer_outputs_jsonl_and_csv_without_tensors(tmp_path):
    writer = TelemetryWriter(tmp_path, "unsafe prefix/../", 42)
    writer.write({"step": 0, "risk": torch.tensor(0.25), "video": {"x0": torch.tensor(3.0)}})
    writer.write({"step": 1, "risk": torch.tensor(0.5), "video": {"x0": torch.tensor(2.0)}})
    writer.close()
    assert writer.jsonl_path.parent == tmp_path
    assert writer.csv_path.parent == tmp_path
    rows = [json.loads(line) for line in writer.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["risk"] == 0.25
    with writer.csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[1]["video_x0"] == "2.0"


def test_stability_map_export_is_explicit_compact_and_machine_readable(tmp_path):
    import numpy as np

    writer = TelemetryWriter(tmp_path, "maps", 42)
    value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    assert not writer.stability_map_directory.exists()
    path = writer.write_stability_maps(
        3,
        torch.tensor(0.25),
        value,
        value + 1,
        torch.ones_like(value),
        torch.full_like(value, 0.75),
        {"seed": 42, "config": {"mode": "test"}},
    )
    assert path.parent == writer.stability_map_directory
    with np.load(path) as payload:
        assert payload["temporal_motion_ratio"].shape == (1, 1, 2, 2, 2)
        assert payload["temporal_motion_ratio"].dtype == np.float16
        assert payload["diffusion_change_ratio"].shape == (1, 1, 2, 2, 2)
        assert payload["restore_mask"].shape == (1, 1, 2, 2, 2)
        assert payload["applied_video_gate"].shape == (1, 1, 2, 2, 2)
        assert '"step": 3' in str(payload["metadata_json"])
