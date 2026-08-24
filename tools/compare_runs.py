from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

try:
    from .build_profile import read_records
except ImportError:  # Direct ``python tools/compare_runs.py`` execution.
    from build_profile import read_records


FIELDS = (
    "risk",
    "effective_order",
    "stochastic_multiplier",
    "video_correction_norm",
    "audio_correction_norm",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scalar summaries from RefDelta telemetry runs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.inputs:
        records = read_records([path])
        values = []
        diagnostic_fields = sorted(
            {
                key
                for row in records
                for key, value in row.items()
                if key.startswith(("comparison_", "ref_delta_"))
                and isinstance(value, float)
            }
        )
        for field in (*FIELDS, *diagnostic_fields):
            present = [float(row[field]) for row in records if isinstance(row.get(field), float)]
            if present:
                values.append(f"{field}={mean(present):.6g}")
        print(f"{path}: steps={len(records)} " + " ".join(values))


if __name__ == "__main__":
    main()
