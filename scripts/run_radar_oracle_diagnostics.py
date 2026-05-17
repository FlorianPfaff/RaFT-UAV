"""Run truth-based radar oracle and timestamp-offset diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import (
    best_time_offset,
    nearest_candidate_oracle,
    summarize_oracle_selection,
    time_offset_sweep,
)
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_truth,
    read_radar_tracks_json,
    read_truth,
    select_flight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar_oracle_diagnostics"))
    parser.add_argument("--offset-min-s", type=float, default=-10.0)
    parser.add_argument("--offset-max-s", type=float, default=10.0)
    parser.add_argument("--offset-step-s", type=float, default=0.25)
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    args = parser.parse_args()

    if args.offset_step_s <= 0.0:
        raise ValueError("offset-step-s must be positive")
    flights = args.flight or ["Opt1", "Opt2", "Opt3"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for flight_name in flights:
        rows.append(_run_one(args, flight_name))
    summary = pd.DataFrame.from_records(rows)
    summary_path = args.output_dir / "radar_oracle_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    return 0


def _run_one(args: argparse.Namespace, flight_name: str) -> dict[str, Any]:
    flight = select_flight(args.dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    radar = _inside_truth_window(
        normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
        truth,
    )

    offsets = _offset_grid(args.offset_min_s, args.offset_max_s, args.offset_step_s)
    sweep = time_offset_sweep(radar, truth, offsets, max_time_delta_s=args.max_time_delta_s)
    best_offset = best_time_offset(sweep, metric="mean_3d_error_m")
    nominal_oracle = nearest_candidate_oracle(
        radar, truth, time_offset_s=0.0, max_time_delta_s=args.max_time_delta_s
    )
    best_oracle = nearest_candidate_oracle(
        radar,
        truth,
        time_offset_s=0.0 if best_offset is None else best_offset,
        max_time_delta_s=args.max_time_delta_s,
    )

    frame_count = _radar_frame_count(radar)
    nominal_summary = summarize_oracle_selection(nominal_oracle, frame_count=frame_count)
    best_summary = summarize_oracle_selection(best_oracle, frame_count=frame_count)

    flight_output = args.output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    sweep_path = flight_output / "time_offset_sweep.csv"
    nominal_path = flight_output / "nearest_candidate_oracle_offset0.csv"
    best_path = flight_output / "nearest_candidate_oracle_best_offset.csv"
    metrics_path = flight_output / "oracle_diagnostics.json"
    sweep.to_csv(sweep_path, index=False)
    nominal_oracle.to_csv(nominal_path, index=False)
    best_oracle.to_csv(best_path, index=False)

    metrics = {
        "flight": flight.name,
        "radar_rows": int(len(radar)),
        "radar_frames": int(frame_count),
        "offset_grid_s": {
            "min": float(args.offset_min_s),
            "max": float(args.offset_max_s),
            "step": float(args.offset_step_s),
        },
        "max_time_delta_s": float(args.max_time_delta_s),
        "best_time_offset_s": None if best_offset is None else float(best_offset),
        "nominal_offset": nominal_summary,
        "best_offset": best_summary,
        "files": {
            "time_offset_sweep": str(sweep_path),
            "nearest_candidate_oracle_offset0": str(nominal_path),
            "nearest_candidate_oracle_best_offset": str(best_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"flight={flight.name} radar_frames={frame_count} "
        f"offset0_mean3d={nominal_summary['mean_3d_error_m']:.3f} "
        f"best_offset_s={best_offset} best_mean3d={best_summary['mean_3d_error_m']:.3f} "
        f"metrics_json={metrics_path}"
    )
    return {
        "flight": flight.name,
        "radar_rows": int(len(radar)),
        "radar_frames": int(frame_count),
        "best_time_offset_s": np.nan if best_offset is None else float(best_offset),
        "offset0_mean_3d_error_m": nominal_summary["mean_3d_error_m"],
        "offset0_p95_3d_error_m": nominal_summary["p95_3d_error_m"],
        "offset0_coverage": nominal_summary["coverage"],
        "best_mean_3d_error_m": best_summary["mean_3d_error_m"],
        "best_p95_3d_error_m": best_summary["p95_3d_error_m"],
        "best_coverage": best_summary["coverage"],
        "metrics_json": str(metrics_path),
    }


def _offset_grid(min_s: float, max_s: float, step_s: float) -> np.ndarray:
    if max_s < min_s:
        raise ValueError("offset-max-s must be >= offset-min-s")
    count = int(np.floor((float(max_s) - float(min_s)) / float(step_s))) + 1
    offsets = float(min_s) + np.arange(count, dtype=float) * float(step_s)
    if offsets.size == 0 or offsets[-1] < float(max_s) - 1.0e-9:
        offsets = np.append(offsets, float(max_s))
    return offsets


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _radar_frame_count(radar: pd.DataFrame) -> int:
    if radar.empty:
        return 0
    group_column = "frame_index" if "frame_index" in radar.columns else "time_s"
    return int(radar[group_column].nunique())


if __name__ == "__main__":
    raise SystemExit(main())
