"""CLI for oracle-gap failure-budget reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
    write_oracle_gap_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar", type=Path, required=True, help="normalized full radar candidates CSV")
    parser.add_argument("--truth", type=Path, required=True, help="normalized truth CSV")
    parser.add_argument("--selected-radar", type=Path, default=None)
    parser.add_argument("--estimates", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--plausible-candidate-gate-m", type=float, default=50.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--estimate-time-gate-s", type=float, default=2.0)
    parser.add_argument("--drift-error-gate-m", type=float, default=150.0)
    args = parser.parse_args(argv)

    config = OracleGapConfig(
        plausible_candidate_gate_m=args.plausible_candidate_gate_m,
        truth_time_gate_s=args.truth_time_gate_s,
        estimate_time_gate_s=args.estimate_time_gate_s,
        drift_error_gate_m=args.drift_error_gate_m,
    )
    selected = pd.read_csv(args.selected_radar) if args.selected_radar is not None else None
    estimates = pd.read_csv(args.estimates) if args.estimates is not None else None
    frame_rows = decompose_radar_oracle_gap(
        radar=pd.read_csv(args.radar),
        truth=pd.read_csv(args.truth),
        selected_radar=selected,
        estimates=estimates,
        config=config,
    )
    write_oracle_gap_report(
        frame_rows=frame_rows,
        selected_radar=selected,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(f"oracle_gap_csv={args.output_csv}")
    if args.output_json is not None:
        print(f"oracle_gap_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
