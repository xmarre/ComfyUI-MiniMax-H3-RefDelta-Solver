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
