from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_custom_node_loads_without_repo_root_on_sys_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    comfyui_path = os.environ.get("COMFYUI_PATH")

    env = os.environ.copy()
    if comfyui_path:
        env["PYTHONPATH"] = comfyui_path
    else:
        env.pop("PYTHONPATH", None)

    code = r'''
import importlib.util
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
assert str(repo_root) not in sys.path

spec = importlib.util.spec_from_file_location(
    "refdelta_custom_node_smoke",
    repo_root / "__init__.py",
    submodule_search_locations=[str(repo_root)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.NODE_CLASS_MAPPINGS
assert module.NODE_DISPLAY_NAME_MAPPINGS
assert "MiniMaxH3RefDeltaProductionSampler" in module.NODE_CLASS_MAPPINGS
assert module.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3RefDeltaProductionSampler"] == "MiniMax H3 RefDelta Stability Sampler"
assert "MiniMaxH3UniformFlowScheduler" in module.NODE_CLASS_MAPPINGS
assert module.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3UniformFlowScheduler"] == "MiniMax H3 Uniform Flow Scheduler [Experimental]"
legacy_scheduler = module.NODE_CLASS_MAPPINGS["MiniMaxH3RefDeltaScheduler"]
uniform_scheduler = module.NODE_CLASS_MAPPINGS["MiniMaxH3UniformFlowScheduler"]
assert legacy_scheduler.INPUT_TYPES()["required"]["profile"][0] == ["r1024_provisional"]
assert uniform_scheduler.INPUT_TYPES()["required"]["profile"][0] == ["h3_uniform_neutral"]
assert uniform_scheduler.INPUT_TYPES()["required"]["steps"][1]["default"] == 19
assert uniform_scheduler.INPUT_TYPES()["required"]["mode"][1]["default"] == "phase_offset_uniform"
assert uniform_scheduler.INPUT_TYPES()["required"]["phase"][1]["default"] == 0.50
assert uniform_scheduler.INPUT_TYPES()["required"]["auto_tail_steps"][1]["default"] is True
assert uniform_scheduler.INPUT_TYPES()["required"]["tail_steps"][1]["default"] == 5
'''

    result = subprocess.run(
        [sys.executable, "-c", code, str(repo_root)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
