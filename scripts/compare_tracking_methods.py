#!/usr/bin/env python3
"""Paired comparison for two RaFT-UAV tracking artifact sets.

This script compares two estimates.csv files against the same truth CSV at the
same truth timestamps.  It writes paired error deltas plus block-bootstrap
confidence intervals, which are more appropriate for autocorrelated trajectory
errors than treating every timestamp as independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import (
    bad_segment_table,
    paired_delta_summary,
    paired_error_delta_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--method-a-estimates", type=Path, required=True)
    parser.add_argument("--method-b-estimates", type=Path, required=True)
    parser.add_argument("--label-a", default="method_a")
    parser.add_argument("--label-b", default="method_b")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paired_method_comparison"))
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=3)
    parser.add_argument("--bootstrap-block-size", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bad-segment-window-s", type=float, default=20.0)
    parser.add_argument("--bad-segment-stride-s", type=float, default=5.0)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth_csv)
    method_a = pd.read_csv(args.method_a_estimates)
    method_b = pd.read_csv(args.method_b_estimates)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deltas = paired_error_delta_frame(
        method_a,
        method_b,
        truth,
        max_time_delta_s=args.max_time_delta_s,
        dimensions=args.dimensions,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    summary = paired_delta_summary(
        deltas,
        block_size=args.bootstrap_block_size,
        resamples=args.bootstrap_resamples,
    )
    summary.update(
        {
            "truth_csv": str(args.truth_csv),
            "method_a_estimates": str(args.method_a_estimates),
            "method_b_estimates": str(args.method_b_estimates),
            "label_a": args.label_a,
            "label_b": args.label_b,
            "dimensions": args.dimensions,
            "max_time_delta_s": args.max_time_delta_s,
        }
    )

    bad = bad_segment_table(
        deltas["time_s"].to_numpy(dtype=float),
        deltas["delta_error_m"].abs().to_numpy(dtype=float),
        window_s=args.bad_segment_window_s,
        stride_s=args.bad_segment_stride_s,
    )
    delta_path = args.output_dir / "paired_delta_by_timestamp.csv"
    summary_path = args.output_dir / "paired_delta_summary.json"
    bad_path = args.output_dir / "paired_delta_bad_segments.csv"
    deltas.to_csv(delta_path, index=False)
    bad.to_csv(bad_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"paired_delta_csv={delta_path}")
    print(f"summary_json={summary_path}")
    print(f"bad_segments_csv={bad_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
