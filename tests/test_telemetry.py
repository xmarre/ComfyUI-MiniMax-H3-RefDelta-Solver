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

