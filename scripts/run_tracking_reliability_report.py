#!/usr/bin/env python3
"""Build reliability diagnostics from one RaFT-UAV run artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import (
    block_bootstrap_interval,
    calibration_transfer_summary,
    error_attribution_by_source_sequence,
    estimate_error_frame,
    recovery_events,
    residual_whiteness_summary,
    track_purity_summary,
    vertical_horizontal_error_summary,
)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing estimates.csv/selected_radar.csv/diagnostics.csv")
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--catastrophic-error-threshold-m", type=float, default=150.0)
    parser.add_argument("--train-diagnostics-csv", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "reliability_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates = pd.read_csv(args.run_dir / "estimates.csv")
    truth = pd.read_csv(args.truth_csv)
    selected = _read_csv_if_exists(args.run_dir / "selected_radar.csv")
    diagnostics = _read_csv_if_exists(args.run_dir / "diagnostics.csv")

    error_frame = estimate_error_frame(estimates, truth, max_time_delta_s=args.max_time_delta_s)
    vertical_summary = vertical_horizontal_error_summary(estimates, truth, max_time_delta_s=args.max_time_delta_s)
    track_purity = track_purity_summary(selected)
    recovery = recovery_events(
        error_frame["time_s"].to_numpy(dtype=float),
        error_frame["error_3d_m"].to_numpy(dtype=float),
        threshold_m=args.catastrophic_error_threshold_m,
    )
    source_attribution = error_attribution_by_source_sequence(
        error_frame,
        error_frame["error_3d_m"].to_numpy(dtype=float),
    )
    whiteness = residual_whiteness_summary(diagnostics) if not diagnostics.empty else pd.DataFrame()
    error_ci = block_bootstrap_interval(error_frame["error_3d_m"], metric="rmse")

    transfer = pd.DataFrame()
    if args.train_diagnostics_csv is not None and args.train_diagnostics_csv.exists() and not diagnostics.empty:
        transfer = calibration_transfer_summary(pd.read_csv(args.train_diagnostics_csv), diagnostics)
        transfer.to_csv(output_dir / "calibration_transfer.csv", index=False)

    error_frame.to_csv(output_dir / "per_estimate_errors.csv", index=False)
    recovery.to_csv(output_dir / "time_to_recovery_events.csv", index=False)
    source_attribution.to_csv(output_dir / "error_by_source_sequence.csv", index=False)
    whiteness.to_csv(output_dir / "residual_whiteness.csv", index=False)

    summary = {
        "run_dir": str(args.run_dir),
        "truth_csv": str(args.truth_csv),
        "max_time_delta_s": args.max_time_delta_s,
        "catastrophic_error_threshold_m": args.catastrophic_error_threshold_m,
        **vertical_summary,
        **{f"track_{key}": value for key, value in track_purity.items()},
        "catastrophic_error_events": int(len(recovery)),
        "max_recovery_time_s": float(recovery["duration_s"].max()) if not recovery.empty else 0.0,
        "rmse_3d_bootstrap": error_ci.to_dict(),
        "calibration_transfer_rows": int(len(transfer)),
    }
    summary_path = output_dir / "reliability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json={summary_path}")
    print(f"per_estimate_errors_csv={output_dir / 'per_estimate_errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
