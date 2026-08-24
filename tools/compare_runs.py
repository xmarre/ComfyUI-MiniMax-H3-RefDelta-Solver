#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from build_profile import read_records


FIELDS = (
    "risk",
    "effective_order",
    "stochastic_multiplier",
    "video_correction_norm",
    "audio_correction_norm",
    "reference_video_x0_relative_error",
    "reference_audio_x0_relative_error",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scalar summaries from RefDelta telemetry runs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.inputs:
        records = read_records([path])
        values = []
        for field in FIELDS:
            present = [float(row[field]) for row in records if isinstance(row.get(field), float)]
            if present:
                values.append(f"{field}={mean(present):.6g}")
        print(f"{path}: steps={len(records)} " + " ".join(values))


if __name__ == "__main__":
    main()

