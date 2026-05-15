#!/usr/bin/env python3
"""Write a compact per-run summary for stateful learned-association sweeps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def count_actions(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "update_action" not in frame.columns:
        return {}
    return {
        str(action): int(count)
        for action, count in frame["update_action"].astype(str).value_counts().sort_index().items()
    }


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_times_s must not be empty")
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


def load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    from raft_uav.io.aerpaw import normalize_truth, read_truth, select_flight

    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def track_ids(frame: pd.DataFrame, length: int) -> np.ndarray:
    if "track_id" not in frame.columns:
        return np.ones(length, dtype=int)
    values = pd.to_numeric(frame["track_id"], errors="coerce").to_numpy(dtype=float)
    out = np.full(length, -1, dtype=int)
    finite = np.isfinite(values)
    out[finite] = values[finite].astype(int)
    return out


def single_target_mot_summary(
    *,
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    max_time_delta_s: float,
    distance_threshold_m: float,
    dimensions: int,
) -> dict[str, Any]:
    required = {"time_s", "east_m", "north_m", "up_m"}
    if estimates.empty or truth.empty:
        return empty_mot_summary(len(truth), len(estimates))
    if not required.issubset(estimates.columns) or not required.issubset(truth.columns):
        return empty_mot_summary(len(truth), len(estimates))

    est_times = estimates["time_s"].to_numpy(dtype=float)
    est_pos = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[:, :dimensions]
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_pos = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[:, :dimensions]
    if est_times.size == 0 or truth_times.size == 0:
        return empty_mot_summary(len(truth), len(estimates))

    nearest = nearest_time_indices(truth_times, est_times)
    dt = np.abs(truth_times[nearest] - est_times)
    dist = np.linalg.norm(est_pos - truth_pos[nearest], axis=1)
    finite = np.isfinite(dt) & np.isfinite(dist)
    matched = finite & (dt <= float(max_time_delta_s)) & (dist <= float(distance_threshold_m))
    matched_truth = nearest[matched]
    matched_tracks = track_ids(estimates, len(estimates))[matched]

    tp = int(np.count_nonzero(matched))
    fp = int(len(estimates) - tp)
    matched_truth_unique = set(int(index) for index in matched_truth.tolist())
    fn = int(len(truth) - len(matched_truth_unique))

    idsw = 0
    fragmentations = 0
    previous_track: int | None = None
    previous_truth: int | None = None
    for truth_index, track_id in sorted(
        zip(matched_truth.tolist(), matched_tracks.tolist(), strict=False), key=lambda item: item[0]
    ):
        if track_id < 0:
            continue
        if previous_track is not None and track_id != previous_track:
            idsw += 1
        if previous_truth is not None and truth_index > previous_truth + 1:
            fragmentations += 1
        previous_track = int(track_id)
        previous_truth = int(truth_index)

    valid_tracks = [int(value) for value in matched_tracks.tolist() if int(value) >= 0]
    if valid_tracks:
        counts = pd.Series(valid_tracks).value_counts()
        dominant_track = int(counts.index[0])
        dominant_matches = int(counts.iloc[0])
    else:
        dominant_track = None
        dominant_matches = 0
    idtp = dominant_matches
    idfp = max(0, fp + tp - idtp)
    idfn = max(0, fn + tp - idtp)
    idf1_den = (2 * idtp) + idfp + idfn
    gt = int(len(truth))
    return {
        "gt": gt,
        "estimates": int(len(estimates)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": int(idsw),
        "fragmentations": int(fragmentations),
        "mota": None if gt == 0 else 1.0 - ((fp + fn + idsw) / gt),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "idf1": None if idf1_den == 0 else (2 * idtp) / idf1_den,
        "fragmentation_per_match": None if tp == 0 else fragmentations / tp,
        "dominant_track_id": dominant_track,
        "dominant_track_matches": int(dominant_matches),
        "unique_matched_track_ids": int(len(set(valid_tracks))),
        "max_time_delta_s": float(max_time_delta_s),
        "distance_threshold_m": float(distance_threshold_m),
        "dimensions": int(dimensions),
    }


def empty_mot_summary(gt: int, estimates: int) -> dict[str, Any]:
    return {
        "gt": int(gt),
        "estimates": int(estimates),
        "tp": 0,
        "fp": int(estimates),
        "fn": int(gt),
        "idsw": 0,
        "fragmentations": 0,
        "mota": None if gt == 0 else 1.0 - (estimates + gt) / gt,
        "idtp": 0,
        "idfp": int(estimates),
        "idfn": int(gt),
        "idf1": None if (estimates + gt) == 0 else 0.0,
        "fragmentation_per_match": None,
        "dominant_track_id": None,
        "dominant_track_matches": 0,
        "unique_matched_track_ids": 0,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    metrics = load_json_if_exists(args.metrics_path)
    diagnostic_summary = load_json_if_exists(args.diagnostic_summary_path)
    diagnostics_actions = count_actions(args.diagnostics_path)
    estimates = read_frame(args.estimates_path)
    selected_radar = read_frame(args.selected_radar_path)

    summary: dict[str, Any] = {
        "flight": args.flight,
        "variant": args.variant,
        "status": "missing_metrics",
        "metrics_path": str(args.metrics_path),
        "diagnostic_summary_path": str(args.diagnostic_summary_path),
        "diagnostics_path": str(args.diagnostics_path),
        "estimates_path": str(args.estimates_path),
        "selected_radar_path": str(args.selected_radar_path),
        "beam_max_hypotheses": int(args.beam_max_hypotheses),
        "beam_max_candidates": int(args.beam_max_candidates),
        "beam_track_switch_cost": float(args.beam_track_switch_cost),
        "beam_missed_detection_cost": float(args.beam_missed_detection_cost),
        "beam_consecutive_miss_cost": float(args.beam_consecutive_miss_cost),
        "beam_missing_track_id_cost": float(args.beam_missing_track_id_cost),
        "beam_lag_s": float(args.beam_lag_s),
        "radar_catprob_threshold": None
        if args.radar_catprob_threshold.lower() == "none"
        else float(args.radar_catprob_threshold),
        "radar_inflation_alpha": float(args.radar_inflation_alpha),
        "association_safety_gate_enabled": args.association_safety_gate_enabled.lower() == "true",
    }
    if metrics is None:
        return summary

    summary.update(
        {
            "status": "ok",
            "radar_association": metrics.get("radar_association"),
            "learned_radar_association_mode": metrics.get("learned_radar_association_mode"),
            "selected_radar_rows": metrics.get("selected_radar_rows"),
            "posterior_records": metrics.get("posterior_records"),
            "accepted_measurements": metrics.get("accepted_measurements"),
            "rejected_measurements": metrics.get("rejected_measurements"),
            "reweighted_measurements": metrics.get("reweighted_measurements"),
            "rmse_2d_m": (metrics.get("position_error_2d") or {}).get("rmse_m"),
            "p95_2d_m": (metrics.get("position_error_2d") or {}).get("p95_m"),
            "rmse_3d_m": (metrics.get("position_error_3d") or {}).get("rmse_m"),
            "p95_3d_m": (metrics.get("position_error_3d") or {}).get("p95_m"),
            "update_action_counts": diagnostics_actions,
            "missed_detection_count": int(diagnostics_actions.get("missed_detection", 0)),
            "rejected_count": int(diagnostics_actions.get("rejected", 0)),
            "inflated_count": int(diagnostics_actions.get("inflated", 0)),
        }
    )
    if isinstance(diagnostic_summary, dict):
        summary["track_switch_count"] = (
            (diagnostic_summary.get("track_switches") or {}).get("selected_radar", {}).get("count")
        )
        summary["covariance_inflation_count"] = (
            (diagnostic_summary.get("covariance_inflation") or {}).get("count")
        )
    try:
        truth = load_truth(args.dataset_root, args.flight)
        summary["selected_radar_mot"] = single_target_mot_summary(
            estimates=selected_radar,
            truth=truth,
            max_time_delta_s=args.max_eval_time_delta_s,
            distance_threshold_m=args.mot_distance_threshold_m,
            dimensions=3,
        )
        summary["estimate_mot"] = single_target_mot_summary(
            estimates=estimates,
            truth=truth,
            max_time_delta_s=args.max_eval_time_delta_s,
            distance_threshold_m=args.mot_distance_threshold_m,
            dimensions=3,
        )
    except Exception as exc:
        summary["mot_summary_error"] = str(exc)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--beam-max-hypotheses", type=int, required=True)
    parser.add_argument("--beam-max-candidates", type=int, required=True)
    parser.add_argument("--beam-track-switch-cost", type=float, required=True)
    parser.add_argument("--beam-missed-detection-cost", type=float, required=True)
    parser.add_argument("--beam-consecutive-miss-cost", type=float, required=True)
    parser.add_argument("--beam-missing-track-id-cost", type=float, required=True)
    parser.add_argument("--beam-lag-s", type=float, required=True)
    parser.add_argument("--radar-catprob-threshold", required=True)
    parser.add_argument("--radar-inflation-alpha", type=float, required=True)
    parser.add_argument("--association-safety-gate-enabled", required=True)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--mot-distance-threshold-m", type=float, default=150.0)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--diagnostic-summary-path", type=Path, required=True)
    parser.add_argument("--diagnostics-path", type=Path, required=True)
    parser.add_argument("--estimates-path", type=Path, required=True)
    parser.add_argument("--selected-radar-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
