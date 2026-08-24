from __future__ import annotations

import os
import sys

import torch


# Native ComfyUI fixtures run on CPU-only Actions runners. Prime ComfyUI's
# public argument parser once before model_management is imported, then restore
# pytest's arguments immediately.
if os.environ.get("COMFYUI_PATH") and not torch.cuda.is_available():
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [original_argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args
    finally:
        sys.argv[:] = original_argv

    # The reviewed historical matrix pins the same minimal comfy-kitchen as
    # Spectrum. Newer source probes this capability, but this CPU fixture never
    # executes INT8 attention.
    import comfy_kitchen

    if not hasattr(comfy_kitchen, "int8_attention_is_available"):
        comfy_kitchen.int8_attention_is_available = lambda: False
