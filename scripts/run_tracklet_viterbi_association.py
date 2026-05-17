"""Run tracklet-Viterbi radar association as a standalone baseline.

This script deliberately avoids changing the main CLI while making the new
association primitive directly evaluable on AERPAW optimization flights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import run_async_cv_baseline
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiConfig,
    select_tracklet_viterbi_path,
)
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    radar_measurements_to_enu,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet_viterbi"))
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=12)
    parser.add_argument("--transition-std-m", type=float, default=60.0)
    parser.add_argument("--velocity-std-mps", type=float, default=15.0)
    parser.add_argument("--switch-penalty", type=float, default=9.0)
    parser.add_argument("--same-track-reward", type=float, default=1.5)
    parser.add_argument("--catprob-weight", type=float, default=6.0)
    parser.add_argument("--track-length-reward", type=float, default=0.35)
    parser.add_argument("--rf-support-weight", type=float, default=0.4)
    parser.add_argument("--rf-support-std-m", type=float, default=250.0)
    parser.add_argument("--rf-time-gate-s", type=float, default=2.0)
    parser.add_argument("--max-speed-mps", type=float, default=55.0)
    parser.add_argument("--speed-penalty-weight", type=float, default=4.0)
    args = parser.parse_args()

    flights = args.flight or ["Opt1", "Opt2", "Opt3"]
    rows: list[dict[str, Any]] = []
    for flight_name in flights:
        rows.append(_run_one(args, flight_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_records(rows)
    summary_path = args.output_dir / "tracklet_viterbi_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    return 0


def _run_one(args: argparse.Namespace, flight_name: str) -> dict[str, Any]:
    flight = select_flight(args.dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)

    rf = pd.DataFrame()
    radar = pd.DataFrame()
    rf_measurements = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        rf_measurements = rf_measurements_to_enu(rf)
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
            truth,
        )

    config = TrackletViterbiConfig(
        max_candidates_per_frame=args.max_candidates_per_frame,
        transition_std_m=args.transition_std_m,
        velocity_std_mps=args.velocity_std_mps,
        switch_penalty=args.switch_penalty,
        same_track_reward=args.same_track_reward,
        catprob_weight=args.catprob_weight,
        track_length_reward=args.track_length_reward,
        rf_support_weight=args.rf_support_weight,
        rf_support_std_m=args.rf_support_std_m,
        rf_time_gate_s=args.rf_time_gate_s,
        max_speed_mps=args.max_speed_mps,
        speed_penalty_weight=args.speed_penalty_weight,
    )
    selected_radar = select_tracklet_viterbi_path(
        radar,
        rf_measurements=rf_measurements,
        candidate_catprob_threshold=args.radar_catprob_threshold,
        config=config,
    )

    measurements = [*rf_measurements, *radar_measurements_to_enu(selected_radar)]
    records = run_async_cv_baseline(
        measurements,
        acceleration_std_mps2=args.acceleration_std,
    )
    if not records:
        raise RuntimeError(f"{flight.name} produced no posterior records")
    records = smooth_tracking_records(
        records,
        method=args.smoother,
        acceleration_std_mps2=args.acceleration_std,
        lag_s=args.smoother_lag_s,
    )
    estimate_frame = _records_to_frame(records)

    flight_output = args.output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    selected_radar.to_csv(flight_output / "selected_radar.csv", index=False)
    estimate_frame.to_csv(flight_output / "estimates.csv", index=False)
    metrics = _metrics(
        flight_name=flight.name,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        estimate_frame=estimate_frame,
        args=args,
    )
    metrics_path = flight_output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"flight={flight.name} selected_radar_rows={len(selected_radar)} "
        f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f} metrics_json={metrics_path}"
    )
    return {
        "flight": flight.name,
        "selected_radar_rows": int(len(selected_radar)),
        "posterior_records": int(len(estimate_frame)),
        "rmse_3d_m": metrics["position_error_3d"]["rmse_m"],
        "p95_3d_m": metrics["position_error_3d"]["p95_m"],
        "rmse_2d_m": metrics["position_error_2d"]["rmse_m"],
        "p95_2d_m": metrics["position_error_2d"]["p95_m"],
        "metrics_json": str(metrics_path),
    }


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "east_m": float(state[0]),
                "north_m": float(state[1]),
                "up_m": float(state[2]),
                "v_east_mps": float(state[3]),
                "v_north_mps": float(state[4]),
                "v_up_mps": float(state[5]),
                "accepted": bool(record.get("accepted", True)),
                "update_action": str(record.get("update_action", "updated")),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _metrics(
    *,
    flight_name: str,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected_radar: pd.DataFrame,
    estimate_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    error_2d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=args.max_eval_time_delta_s,
        dimensions=2,
    )
    error_3d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=args.max_eval_time_delta_s,
        dimensions=3,
    )
    return {
        "flight": flight_name,
        "radar_association": "tracklet-viterbi",
        "radar_catprob_threshold": float(args.radar_catprob_threshold),
        "smoother": {"method": args.smoother, "lag_s": float(args.smoother_lag_s)},
        "tracklet_viterbi": {
            "max_candidates_per_frame": int(args.max_candidates_per_frame),
            "transition_std_m": float(args.transition_std_m),
            "velocity_std_mps": float(args.velocity_std_mps),
            "switch_penalty": float(args.switch_penalty),
            "same_track_reward": float(args.same_track_reward),
            "catprob_weight": float(args.catprob_weight),
            "track_length_reward": float(args.track_length_reward),
            "rf_support_weight": float(args.rf_support_weight),
            "rf_support_std_m": float(args.rf_support_std_m),
            "rf_time_gate_s": float(args.rf_time_gate_s),
            "max_speed_mps": float(args.max_speed_mps),
            "speed_penalty_weight": float(args.speed_penalty_weight),
        },
        "truth_rows": int(len(truth)),
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected_radar)),
        "posterior_records": int(len(estimate_frame)),
        "position_error_2d": summarize_errors(error_2d),
        "position_error_3d": summarize_errors(error_3d),
    }


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


if __name__ == "__main__":
    raise SystemExit(main())
