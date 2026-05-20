#!/usr/bin/env python3
"""Mine worst tracking intervals from estimates/truth artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import bad_segment_table, estimate_error_frame, recovery_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bad_segments"))
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--stride-s", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--catastrophic-error-threshold-m", type=float, default=150.0)
    args = parser.parse_args()

    estimates = pd.read_csv(args.estimates_csv)
    truth = pd.read_csv(args.truth_csv)
    errors = estimate_error_frame(estimates, truth, max_time_delta_s=args.max_time_delta_s)
    segments = bad_segment_table(
        errors["time_s"].to_numpy(dtype=float),
        errors["error_3d_m"].to_numpy(dtype=float),
        window_s=args.window_s,
        stride_s=args.stride_s,
        top_k=args.top_k,
    )
    recoveries = recovery_events(
        errors["time_s"].to_numpy(dtype=float),
        errors["error_3d_m"].to_numpy(dtype=float),
        threshold_m=args.catastrophic_error_threshold_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.output_dir / "per_estimate_errors.csv", index=False)
    segments.to_csv(args.output_dir / "bad_segments.csv", index=False)
    recoveries.to_csv(args.output_dir / "catastrophic_recovery_events.csv", index=False)
    print(f"bad_segments_csv={args.output_dir / 'bad_segments.csv'}")
    print(f"recovery_events_csv={args.output_dir / 'catastrophic_recovery_events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
