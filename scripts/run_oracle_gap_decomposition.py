#!/usr/bin/env python3
"""Build oracle-gap and confidence reports from completed RaFT-UAV runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.evaluation.oracle_gap_decomposition import (  # noqa: E402
    OracleGapConfig,
    confidence_diagnostics,
    decompose_radar_oracle_gap,
    write_oracle_gap_report,
)
from raft_uav.io.aerpaw import (  # noqa: E402
    normalize_radar,
    normalize_truth,
    read_radar_tracks_json,
    read_truth,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle_gap_decomposition"))
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--plausible-gate-m", type=float, default=50.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--estimate-time-gate-s", type=float, default=2.0)
    parser.add_argument("--drift-error-gate-m", type=float, default=150.0)
    args = parser.parse_args(argv)

    cfg = OracleGapConfig(
        plausible_candidate_gate_m=args.plausible_gate_m,
        truth_time_gate_s=args.truth_time_gate_s,
        estimate_time_gate_s=args.estimate_time_gate_s,
        drift_error_gate_m=args.drift_error_gate_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for requested in args.flights:
        flight = select_flight(args.dataset_root, requested)
        if flight.truth_txt is None or flight.radar_json is None:
            continue
        truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
        radar = normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time)
        run_flight_dir = args.run_dir / flight.name
        selected = _read_csv_or_empty(run_flight_dir / "selected_radar.csv")
        estimates = _read_csv_or_empty(run_flight_dir / "estimates.csv")
        frame_rows = decompose_radar_oracle_gap(
            radar=radar,
            truth=truth,
            selected_radar=selected,
            estimates=estimates,
            config=cfg,
        )
        flight_dir = args.output_dir / flight.name
        summary = write_oracle_gap_report(
            frame_rows=frame_rows,
            selected_radar=selected,
            output_csv=flight_dir / "oracle_gap_frames.csv",
            output_json=flight_dir / "oracle_gap_summary.json",
        )
        confidence = confidence_diagnostics(estimates, selected)
        if not confidence.empty:
            confidence.to_csv(flight_dir / "confidence_diagnostics.csv", index=False)
        summary_rows.append({"flight": flight.name, **summary})
    _write_csv(args.output_dir / "oracle_gap_summary.csv", summary_rows)
    (args.output_dir / "oracle_gap_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    print(f"summary_csv={args.output_dir / 'oracle_gap_summary.csv'}")
    return 0


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
