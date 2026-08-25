from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfyui_refdelta_solver.h3_scheduler import (
    SCHEDULER_MODES,
    base_to_shifted_sigma,
    h3_uniform_flow_sigmas,
    load_flow_profile,
    shifted_sigma_to_base,
)


class InspectionSampling:
    def __init__(self, shift: float, audio_shift: float, timesteps: int = 1000):
        self.shift = shift
        self.audio_shift = audio_shift
        self.multiplier = 1000
        base = torch.arange(1, timesteps + 1, dtype=torch.float64) / timesteps
        self.sigmas = self.sigma(base * self.multiplier).to(torch.float32)
        self.sigma_min = self.sigmas[0]
        self.sigma_max = self.sigmas[-1]

    def sigma(self, timestep):
        base = torch.as_tensor(timestep) / self.multiplier
        return base_to_shifted_sigma(base, self.shift)


def inspect(args) -> list[dict[str, float | int | str]]:
    if args.comfyui_path is not None:
        sys.path.insert(0, str(args.comfyui_path.resolve()))
    if args.mode == "legacy_ddim_uniform" and not torch.cuda.is_available():
        original_argv = sys.argv[:]
        try:
            sys.argv[:] = [original_argv[0], "--cpu"]
            import comfy.options

            comfy.options.enable_args_parsing()
            import comfy.cli_args
        finally:
            sys.argv[:] = original_argv
        import comfy_kitchen

        if not hasattr(comfy_kitchen, "int8_attention_is_available"):
            comfy_kitchen.int8_attention_is_available = lambda: False
    sampling = InspectionSampling(args.video_shift, args.audio_shift)
    profile = load_flow_profile(args.profile, args.profile_path) if args.mode == "curvature_profile" else None
    sigmas = h3_uniform_flow_sigmas(
        sampling,
        args.steps,
        args.denoise,
        args.mode,
        phase=args.phase,
        power=args.power,
        tail_steps=args.tail_steps,
        tail_start=args.tail_start,
        tail_power=args.tail_power,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
        arc_strength=args.arc_strength,
        audio_weight=args.audio_weight,
        structure_fraction=args.structure_fraction,
        mid_power=args.mid_power,
        detail_power=args.detail_power,
        profile=profile,
    ).to(torch.float64)
    base = shifted_sigma_to_base(sigmas, args.video_shift)
    audio = base_to_shifted_sigma(base, args.audio_shift)

    rows = []
    for index in range(sigmas.numel()):
        next_index = min(index + 1, sigmas.numel() - 1)
        rows.append(
            {
                "mode": args.mode,
                "index": index,
                "base_time": float(base[index]),
                "video_sigma": float(sigmas[index]),
                "audio_sigma": float(audio[index]),
                "delta_base_time": float(base[index] - base[next_index]),
                "delta_video_sigma": float(sigmas[index] - sigmas[next_index]),
                "delta_audio_sigma": float(audio[index] - audio[next_index]),
            }
        )
    return rows


def _write(rows, output_format: str, output: Path | None) -> None:
    handle = output.open("w", encoding="utf-8", newline="") if output is not None else sys.stdout
    try:
        if output_format == "json":
            json.dump(rows, handle, indent=2)
            handle.write("\n")
        else:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if output is not None:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MiniMax-H3 shared-flow scheduler coordinates.")
    parser.add_argument("--mode", choices=SCHEDULER_MODES, default="legacy_ddim_uniform")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--phase", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--tail-steps", type=int, default=5)
    parser.add_argument("--tail-start", type=float, default=0.15)
    parser.add_argument("--tail-power", type=float, default=2.0)
    parser.add_argument("--beta-alpha", type=float, default=0.6)
    parser.add_argument("--beta-beta", type=float, default=0.6)
    parser.add_argument("--arc-strength", type=float, default=0.5)
    parser.add_argument("--audio-weight", type=float, default=1.0)
    parser.add_argument("--structure-fraction", type=float, default=0.5)
    parser.add_argument("--mid-power", type=float, default=1.0)
    parser.add_argument("--detail-power", type=float, default=1.0)
    parser.add_argument("--profile", default="h3_uniform_neutral")
    parser.add_argument("--profile-path", type=Path)
    parser.add_argument("--comfyui-path", type=Path, help="Required for legacy_ddim_uniform outside ComfyUI.")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = inspect(args)
    if not rows:
        raise ValueError("the selected denoise value produced an empty schedule")
    _write(rows, args.format, args.output)


if __name__ == "__main__":
    main()
