from __future__ import annotations

import json
from pathlib import Path

from comfyui_refdelta_solver import h3_scheduler


class SingleArgTraversable:
    """Mimic importlib CompatibilityFiles.SpecPath's one-child joinpath API."""

    def __init__(self, path: Path):
        self.path = path

    def joinpath(self, child: str):
        return type(self)(self.path / child)

    def is_file(self) -> bool:
        return self.path.is_file()

    def open(self, mode="r", *args, **kwargs):
        return self.path.open(mode, *args, **kwargs)

    def __str__(self) -> str:
        return str(self.path)


def test_packaged_flow_profile_supports_single_arg_traversable_joinpath(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "compat.json").write_text(
        json.dumps(
            {
                "version": 2,
                "id": "compat",
                "model_family": "MiniMax-H3 audiovisual shifted flow",
                "status": "neutral-research-control",
                "domain": "shared-base-time-progress-density",
                "points": [
                    {"progress": 0.0, "density": 1.0},
                    {"progress": 1.0, "density": 1.0},
                ],
                "metadata": {
                    "evidence_source": "neutral_control",
                    "comparison_metrics_used_for_density": False,
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        h3_scheduler,
        "files",
        lambda package: SingleArgTraversable(tmp_path),
    )

    profile = h3_scheduler.load_flow_profile("compat")

    assert profile.profile_id == "compat"
    assert profile.progress == (0.0, 1.0)
    assert profile.density == (1.0, 1.0)
