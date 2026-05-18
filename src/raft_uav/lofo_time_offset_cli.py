"""LOFO-safe RF/radar time-offset calibration and tracking evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)
from raft_uav.calibration.time_offset import (
    TimeOffsetFitResult,
    apply_time_offset,
    fit_measurement_time_offset,
    fit_radar_time_offset,
    make_offset_grid,
)
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


@dataclass(frozen=True)
class _LoadedFlight:
    name: str
    truth: pd.DataFrame
    rf: pd.DataFrame
    radar: pd.DataFrame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav-lofo-time-offset")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_time_offset"))
    parser.add_argument("--offset-min-s", type=float, default=-10.0)
    parser.add_argument("--offset-max-s", type=float, default=10.0)
    parser.add_argument("--offset-step-s", type=float, default=0.25)
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--radar-offset-metric", default="mean_3d_error_m")
    parser.add_argument("--rf-offset-metric", default="mean_2d_error_m")
    parser.add_argument("--disable-radar-offset", action="store_true")
    parser.add_argument("--disable-rf-offset", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--max-candidates-per-frame", type=int, default=8)
    parser.add_argument("--missed-detection-cost", type=float, default=7.0)
    parser.add_argument("--track-switch-cost", type=float, default=8.0)
    parser.add_argument("--catprob-weight", type=float, default=2.5)
    parser.add_argument("--anchor-nis-weight", type=float, default=0.35)
    parser.add_argument("--transition-nis-weight", type=float, default=1.0)
    parser.add_argument("--velocity-nis-weight", type=float, default=0.15)
    parser.add_argument("--max-speed-mps", type=float, default=55.0)
    parser.add_argument("--range-gate-m", type=float, default=850.0)
    parser.add_argument("--disable-rf-anchor", action="store_true")
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    args = parser.parse_args(argv)

    requested = args.flight or ["Opt1", "Opt2", "Opt3"]
    offsets = make_offset_grid(args.offset_min_s, args.offset_max_s, args.offset_step_s)
    flights = [_load_flight(args.dataset_root, name) for name in requested]
    if len(flights) < 2:
        raise ValueError("LOFO calibration needs at least two flights")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for holdout in flights:
        train = [flight for flight in flights if flight.name != holdout.name]
        rows.append(_run_holdout(args, holdout, train, offsets))

    summary = pd.DataFrame.from_records(rows)
    summary_path = args.output_dir / "lofo_time_offset_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    return 0


def _run_holdout(
    args: argparse.Namespace,
    holdout: _LoadedFlight,
    train: list[_LoadedFlight],
    offsets: np.ndarray,
) -> dict[str, Any]:
    out = args.output_dir / holdout.name
    out.mkdir(parents=True, exist_ok=True)

    radar_fit = _fit_radar(args, train, offsets)
    rf_fit = _fit_rf(args, train, offsets)
    radar_offset = 0.0 if args.disable_radar_offset else (radar_fit.best_offset_s or 0.0)
    rf_offset = 0.0 if args.disable_rf_offset else (rf_fit.best_offset_s or 0.0)

    radar_fit.sweep.to_csv(out / "radar_time_offset_sweep.csv", index=False)
    rf_fit.sweep.to_csv(out / "rf_time_offset_sweep.csv", index=False)

    radar = _inside_truth_window(apply_time_offset(holdout.radar, radar_offset), holdout.truth)
    rf = _inside_truth_window(apply_time_offset(holdout.rf, rf_offset), holdout.truth)
    radar.to_csv(out / "radar_time_corrected.csv", index=False)
    rf.to_csv(out / "rf_time_corrected.csv", index=False)

    metrics: dict[str, Any] = {
        "flight": holdout.name,
        "train_flights": [flight.name for flight in train],
        "radar_time_offset_fit": radar_fit.summary(),
        "rf_time_offset_fit": {**rf_fit.summary(), "source": "rf"},
        "applied_radar_time_offset_s": float(radar_offset),
        "applied_rf_time_offset_s": float(rf_offset),
        "tracking": None,
    }
    tracking_summary: dict[str, Any] = {}
    if not args.skip_tracking:
        tracking_summary = _run_tracking(args, holdout, rf, radar, out)
        metrics["tracking"] = tracking_summary

    metrics_path = out / "lofo_time_offset_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"holdout={holdout.name} radar_offset_s={radar_offset:.3f} "
        f"rf_offset_s={rf_offset:.3f} metrics_json={metrics_path}"
    )

    row = {
        "flight": holdout.name,
        "train_flights": ",".join(flight.name for flight in train),
        "radar_offset_s": float(radar_offset),
        "rf_offset_s": float(rf_offset),
        "radar_fit_metric": radar_fit.metric,
        "rf_fit_metric": rf_fit.metric,
        "metrics_json": str(metrics_path),
    }
    for key in ("rmse_3d_m", "p95_3d_m", "mae_3d_m", "selected_radar_rows"):
        if key in tracking_summary:
            row[key] = tracking_summary[key]
    return row


def _fit_radar(args: argparse.Namespace, train: list[_LoadedFlight], offsets: np.ndarray) -> TimeOffsetFitResult:
    pairs = [(flight.radar, flight.truth) for flight in train if not flight.radar.empty]
    if args.disable_radar_offset or not pairs:
        return TimeOffsetFitResult("radar", 0.0, args.radar_offset_metric, pd.DataFrame())
    return fit_radar_time_offset(
        pairs,
        offsets,
        max_time_delta_s=args.max_time_delta_s,
        metric=args.radar_offset_metric,
    )


def _fit_rf(args: argparse.Namespace, train: list[_LoadedFlight], offsets: np.ndarray) -> TimeOffsetFitResult:
    pairs = [(flight.rf, flight.truth) for flight in train if not flight.rf.empty]
    if args.disable_rf_offset or not pairs:
        return TimeOffsetFitResult("rf", 0.0, args.rf_offset_metric, pd.DataFrame())
    fit = fit_measurement_time_offset(
        pairs,
        offsets,
        dimensions=2,
        max_time_delta_s=args.max_time_delta_s,
        metric=args.rf_offset_metric,
    )
    return TimeOffsetFitResult("rf", fit.best_offset_s, fit.metric, fit.sweep)


def _run_tracking(
    args: argparse.Namespace,
    holdout: _LoadedFlight,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    out: Path,
) -> dict[str, Any]:
    rf_measurements = rf_measurements_to_enu(rf)
    config = TrackletViterbiAssociationConfig(
        max_candidates_per_frame=args.max_candidates_per_frame,
        missed_detection_cost=args.missed_detection_cost,
        track_switch_cost=args.track_switch_cost,
        catprob_weight=args.catprob_weight,
        anchor_nis_weight=args.anchor_nis_weight,
        transition_nis_weight=args.transition_nis_weight,
        velocity_nis_weight=args.velocity_nis_weight,
        max_speed_mps=args.max_speed_mps,
        range_gate_m=None if args.range_gate_m <= 0.0 else args.range_gate_m,
        use_rf_anchor=not args.disable_rf_anchor,
    )
    records, selected_radar = run_async_cv_baseline_with_tracklet_viterbi_association(
        rf_measurements=rf_measurements,
        radar=radar,
        acceleration_std_mps2=args.acceleration_std,
        candidate_catprob_threshold=args.radar_catprob_threshold,
        config=config,
    )
    if not records:
        raise RuntimeError(f"{holdout.name} produced no posterior records")
    records = smooth_tracking_records(
        records,
        method=args.smoother,
        acceleration_std_mps2=args.acceleration_std,
        lag_s=args.smoother_lag_s,
    )
    estimates = _records_to_frame(records)
    estimates.to_csv(out / "estimates.csv", index=False)
    selected_radar.to_csv(out / "selected_radar.csv", index=False)
    errors_3d = position_errors_m(
        estimates["time_s"].to_numpy(dtype=float),
        estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        holdout.truth["time_s"].to_numpy(dtype=float),
        holdout.truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        max_time_delta_s=args.max_eval_time_delta_s,
        dimensions=3,
    )
    errors_2d = position_errors_m(
        estimates["time_s"].to_numpy(dtype=float),
        estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        holdout.truth["time_s"].to_numpy(dtype=float),
        holdout.truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        max_time_delta_s=args.max_eval_time_delta_s,
        dimensions=2,
    )
    summary_3d = summarize_errors(errors_3d)
    summary_2d = summarize_errors(errors_2d)
    return {
        "selected_radar_rows": int(len(selected_radar)),
        "posterior_records": int(len(estimates)),
        "position_error_3d": summary_3d,
        "position_error_2d": summary_2d,
        "rmse_3d_m": summary_3d["rmse_m"],
        "mae_3d_m": summary_3d["mae_m"],
        "p95_3d_m": summary_3d["p95_m"],
        "rmse_2d_m": summary_2d["rmse_m"],
        "p95_2d_m": summary_2d["p95_m"],
    }


def _load_flight(dataset_root: Path, name: str) -> _LoadedFlight:
    flight = select_flight(dataset_root, name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
    rf = pd.DataFrame()
    if flight.rf_csv is not None:
        rf = _inside_truth_window(normalize_rf(read_rf_csv(flight.rf_csv), projector, origin_time), truth)
    radar = pd.DataFrame()
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time),
            truth,
        )
    return _LoadedFlight(name=flight.name, truth=truth, rf=rf, radar=radar)


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or truth.empty or "time_s" not in frame.columns:
        return frame
    lower = float(truth["time_s"].min())
    upper = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= lower) & (frame["time_s"] <= upper)].copy()


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "accepted": bool(record.get("accepted", True)),
                "update_action": str(record.get("update_action", "updated")),
                "east_m": float(state[0]),
                "north_m": float(state[1]),
                "up_m": float(state[2]),
                "v_east_mps": float(state[3]),
                "v_north_mps": float(state[4]),
                "v_up_mps": float(state[5]),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
