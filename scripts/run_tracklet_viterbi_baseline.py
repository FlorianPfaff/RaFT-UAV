"""Run the truth-free tracklet-Viterbi radar association baseline on one flight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records  # noqa: E402
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig  # noqa: E402
from raft_uav.baselines.tracklet_viterbi_result import (  # noqa: E402
    run_async_cv_baseline_with_tracklet_viterbi_result,
)
from raft_uav.evaluation.metrics import nearest_time_indices, position_errors_m, summarize_errors  # noqa: E402
from raft_uav.io.aerpaw import (  # noqa: E402
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet_viterbi"))
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--max-candidates-per-frame", type=int, default=8)
    parser.add_argument("--path-beam-width", type=int, default=1)
    parser.add_argument("--replay-nis-weight", type=float, default=0.0)
    parser.add_argument("--replay-rejection-cost", type=float, default=0.0)
    parser.add_argument("--replay-roughness-weight", type=float, default=0.0)
    parser.add_argument("--missed-detection-cost", type=float, default=7.0)
    parser.add_argument("--track-switch-cost", type=float, default=8.0)
    parser.add_argument("--catprob-weight", type=float, default=2.5)
    parser.add_argument("--anchor-nis-weight", type=float, default=0.35)
    parser.add_argument("--transition-nis-weight", type=float, default=1.0)
    parser.add_argument("--velocity-nis-weight", type=float, default=0.15)
    parser.add_argument("--max-speed-mps", type=float, default=55.0)
    parser.add_argument("--range-gate-m", type=float, default=850.0)
    parser.add_argument("--disable-rf-anchor", action="store_true")
    parser.add_argument("--association-reranker-json", type=Path, default=None)
    parser.add_argument("--association-reranker-weight", type=float, default=1.0)
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=float, default=0.0)
    parser.add_argument("--robust-update", choices=["none", "nis-inflate"], default="nis-inflate")
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    args = parser.parse_args(argv)

    flight = select_flight(args.dataset_root, args.flight)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))

    rf = pd.DataFrame()
    rf_measurements = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(normalize_rf(read_rf_csv(flight.rf_csv), projector, origin_time), truth)
        rf_measurements = rf_measurements_to_enu(rf)

    radar = pd.DataFrame()
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time),
            truth,
        )

    config = TrackletViterbiAssociationConfig(
        max_candidates_per_frame=args.max_candidates_per_frame,
        path_beam_width=args.path_beam_width,
        replay_nis_weight=args.replay_nis_weight,
        replay_rejection_cost=args.replay_rejection_cost,
        replay_roughness_weight=args.replay_roughness_weight,
        missed_detection_cost=args.missed_detection_cost,
        track_switch_cost=args.track_switch_cost,
        catprob_weight=args.catprob_weight,
        anchor_nis_weight=args.anchor_nis_weight,
        transition_nis_weight=args.transition_nis_weight,
        velocity_nis_weight=args.velocity_nis_weight,
        max_speed_mps=args.max_speed_mps,
        range_gate_m=None if args.range_gate_m <= 0.0 else args.range_gate_m,
        use_rf_anchor=not args.disable_rf_anchor,
        association_reranker_path=args.association_reranker_json,
        association_reranker_weight=args.association_reranker_weight,
    )
    robust_updates = None if args.robust_update == "none" else {"rf": args.robust_update, "radar": args.robust_update}
    inflation_alphas = None if robust_updates is None else {"rf": args.rf_inflation_alpha, "radar": args.radar_inflation_alpha}

    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=rf_measurements,
        radar=radar,
        acceleration_std_mps2=args.acceleration_std,
        gate_probabilities_by_source={"rf": args.rf_gate_prob, "radar": args.radar_gate_prob},
        safety_gate_probabilities_by_source={"rf": args.rf_safety_gate_prob, "radar": args.radar_safety_gate_prob},
        robust_update_by_source=robust_updates,
        inflation_alpha_by_source=inflation_alphas,
        max_residual_norms_by_source={
            "rf": None if args.rf_max_residual_m <= 0.0 else args.rf_max_residual_m,
            "radar": None if args.radar_max_residual_m <= 0.0 else args.radar_max_residual_m,
        },
        candidate_catprob_threshold=args.radar_catprob_threshold,
        config=config,
    )
    records = result.records
    selected_radar = result.accepted_radar
    viterbi_selected_radar = result.viterbi_selected_radar
    radar_candidate_ledger = result.radar_candidate_ledger
    if not records:
        raise RuntimeError(f"{flight.name} produced no posterior records")

    records = smooth_tracking_records(records, method=args.smoother, acceleration_std_mps2=args.acceleration_std, lag_s=args.smoother_lag_s)
    estimates = _records_to_frame(records)
    selected_radar_diagnostics = _selected_radar_diagnostics(
        selected_radar,
        truth,
        radar,
        max_time_delta_s=args.max_eval_time_delta_s,
    )
    viterbi_selected_radar_diagnostics = _selected_radar_diagnostics(
        viterbi_selected_radar,
        truth,
        radar,
        max_time_delta_s=args.max_eval_time_delta_s,
    )
    metrics = _metrics(
        flight.name,
        truth,
        rf,
        radar,
        selected_radar,
        estimates,
        viterbi_selected_radar,
        selected_radar_diagnostics,
        viterbi_selected_radar_diagnostics,
        args,
    )
    metrics["radar_candidate_ledger_rows"] = int(len(radar_candidate_ledger))
    if "association_viterbi_selected" in radar_candidate_ledger.columns:
        metrics["radar_candidate_ledger_selected_rows"] = int(
            radar_candidate_ledger["association_viterbi_selected"].astype(bool).sum()
        )
    else:
        metrics["radar_candidate_ledger_selected_rows"] = 0

    out = args.output_dir / flight.name
    out.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(out / "estimates.csv", index=False)
    selected_radar.to_csv(out / "selected_radar.csv", index=False)
    viterbi_selected_radar.to_csv(out / "viterbi_selected_radar.csv", index=False)
    radar_candidate_ledger.to_csv(out / "radar_candidate_ledger.csv", index=False)
    (out / "selected_radar_diagnostics.json").write_text(
        json.dumps(selected_radar_diagnostics, indent=2),
        encoding="utf-8",
    )
    (out / "viterbi_selected_radar_diagnostics.json").write_text(
        json.dumps(viterbi_selected_radar_diagnostics, indent=2),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"flight={flight.name}")
    print("radar_association=tracklet-viterbi")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"posterior_records={len(estimates)}")
    print(f"metrics_json={out / 'metrics.json'}")
    print(f"selected_radar_diagnostics_json={out / 'selected_radar_diagnostics.json'}")
    print(f"viterbi_selected_radar_rows={len(viterbi_selected_radar)}")
    print(f"viterbi_selected_radar_csv={out / 'viterbi_selected_radar.csv'}")
    print(f"viterbi_selected_radar_diagnostics_json={out / 'viterbi_selected_radar_diagnostics.json'}")
    print(f"radar_candidate_ledger_rows={len(radar_candidate_ledger)}")
    print(f"radar_candidate_ledger_csv={out / 'radar_candidate_ledger.csv'}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    print(f"p95_3d_m={metrics['position_error_3d']['p95_m']:.3f}")
    selected_rmse = selected_radar_diagnostics["position_error_3d"]["rmse_m"]
    if np.isfinite(float(selected_rmse)):
        print(f"selected_radar_rmse_3d_m={float(selected_rmse):.3f}")
    return 0


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or truth.empty or "time_s" not in frame.columns:
        return frame
    return frame.loc[(frame["time_s"] >= truth["time_s"].min()) & (frame["time_s"] <= truth["time_s"].max())].copy()


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        filtered_state = record.get("filtered_state")
        filtered = None if filtered_state is None else np.asarray(filtered_state, dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "measurement_dim": int(record.get("measurement_dim", 0)),
                "accepted": bool(record.get("accepted", True)),
                "update_action": str(record.get("update_action", "updated")),
                "nis": _optional_float(record.get("nis")),
                "covariance_scale": _optional_float(record.get("covariance_scale")),
                "east_m": state[0],
                "north_m": state[1],
                "up_m": state[2],
                "v_east_mps": state[3],
                "v_north_mps": state[4],
                "v_up_mps": state[5],
                "filtered_east_m": None if filtered is None else filtered[0],
                "filtered_north_m": None if filtered is None else filtered[1],
                "filtered_up_m": None if filtered is None else filtered[2],
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _metrics(
    name: str,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    estimates: pd.DataFrame,
    viterbi_selected: pd.DataFrame,
    selected_radar_diagnostics: dict[str, object],
    viterbi_selected_radar_diagnostics: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    estimate_positions = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors_2d = position_errors_m(estimate_times, estimate_positions, truth_times, truth_positions, max_time_delta_s=args.max_eval_time_delta_s, dimensions=2)
    errors_3d = position_errors_m(estimate_times, estimate_positions, truth_times, truth_positions, max_time_delta_s=args.max_eval_time_delta_s, dimensions=3)
    accepted = int(estimates["accepted"].sum()) if "accepted" in estimates.columns else len(estimates)
    return {
        "flight": name,
        "radar_association": "tracklet-viterbi",
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected)),
        "viterbi_selected_radar_rows": int(len(viterbi_selected)),
        "posterior_records": int(len(estimates)),
        "accepted_measurements": accepted,
        "rejected_measurements": int(len(estimates) - accepted),
        "position_error_2d": summarize_errors(errors_2d),
        "position_error_3d": summarize_errors(errors_3d),
        "selected_radar_diagnostics": selected_radar_diagnostics,
        "selected_radar_position_error_2d": selected_radar_diagnostics["position_error_2d"],
        "selected_radar_position_error_3d": selected_radar_diagnostics["position_error_3d"],
        "selected_radar_truth_coverage": selected_radar_diagnostics["truth_coverage"],
        "selected_radar_frame_coverage_rate": selected_radar_diagnostics["radar_frame_coverage_rate"],
        "selected_radar_track_switch_count": selected_radar_diagnostics["track_switch_count"],
        "selected_radar_unique_track_ids": selected_radar_diagnostics["unique_track_ids"],
        "tracklet_viterbi": _tracklet_viterbi_config(args),
        "viterbi_selected_radar_diagnostics": viterbi_selected_radar_diagnostics,
        "viterbi_selected_radar_position_error_2d": viterbi_selected_radar_diagnostics["position_error_2d"],
        "viterbi_selected_radar_position_error_3d": viterbi_selected_radar_diagnostics["position_error_3d"],
        "viterbi_selected_radar_truth_coverage": viterbi_selected_radar_diagnostics["truth_coverage"],
        "viterbi_selected_radar_frame_coverage_rate": viterbi_selected_radar_diagnostics["radar_frame_coverage_rate"],
        "viterbi_selected_radar_rejected_rows": int(len(viterbi_selected) - len(selected)),
        "viterbi_selected_radar_rejection_rate": float((len(viterbi_selected) - len(selected)) / len(viterbi_selected))
        if len(viterbi_selected)
        else float("nan"),
        "smoother": {"method": args.smoother, "lag_s": args.smoother_lag_s},
    }


def _selected_radar_diagnostics(
    selected: pd.DataFrame,
    truth: pd.DataFrame,
    radar: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> dict[str, object]:
    """Return direct quality diagnostics for the selected radar path."""

    selected_times, selected_positions = _trajectory_arrays(selected)
    truth_times, truth_positions = _trajectory_arrays(truth)

    if selected_times.size and truth_times.size:
        errors_2d = position_errors_m(
            selected_times,
            selected_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=2,
        )
        errors_3d = position_errors_m(
            selected_times,
            selected_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=3,
        )
    else:
        errors_2d = np.array([], dtype=float)
        errors_3d = np.array([], dtype=float)

    radar_frame_count = _frame_count(radar)
    selected_frame_count = _frame_count(selected)
    row_fraction = float(len(selected) / len(radar)) if len(radar) else float("nan")
    frame_fraction = (
        float(selected_frame_count / radar_frame_count)
        if radar_frame_count
        else float("nan")
    )
    return {
        "selected_radar_rows": int(len(selected)),
        "radar_rows": int(len(radar)),
        "selected_radar_row_fraction": row_fraction,
        "radar_frame_count": int(radar_frame_count),
        "selected_radar_frame_count": int(selected_frame_count),
        "radar_frame_coverage_rate": frame_fraction,
        "truth_coverage": _truth_coverage(
            truth_times,
            selected_times,
            max_time_delta_s=max_time_delta_s,
        ),
        "position_error_2d": _summarize_errors_with_tail(errors_2d),
        "position_error_3d": _summarize_errors_with_tail(errors_3d),
        "track_switch_count": _track_switch_count(selected),
        "unique_track_ids": _unique_track_id_count(selected),
        "association_anchor_nis": _numeric_column_stats(selected, "association_anchor_nis"),
        "association_nis": _numeric_column_stats(selected, "association_nis"),
        "association_score": _numeric_column_stats(selected, "association_score"),
    }


def _tracklet_viterbi_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the tracklet-Viterbi configuration stored with metrics."""

    return {
        "candidate_threshold": args.radar_catprob_threshold,
        "max_candidates_per_frame": args.max_candidates_per_frame,
        "path_beam_width": args.path_beam_width,
        "replay_nis_weight": args.replay_nis_weight,
        "replay_rejection_cost": args.replay_rejection_cost,
        "replay_roughness_weight": args.replay_roughness_weight,
        "missed_detection_cost": args.missed_detection_cost,
        "track_switch_cost": args.track_switch_cost,
        "catprob_weight": args.catprob_weight,
        "anchor_nis_weight": args.anchor_nis_weight,
        "transition_nis_weight": args.transition_nis_weight,
        "velocity_nis_weight": args.velocity_nis_weight,
        "max_speed_mps": args.max_speed_mps,
        "range_gate_m": None if args.range_gate_m <= 0.0 else args.range_gate_m,
        "use_rf_anchor": not args.disable_rf_anchor,
        "association_reranker_json": str(args.association_reranker_json)
        if args.association_reranker_json is not None
        else None,
        "association_reranker_weight": args.association_reranker_weight,
    }


def _summarize_errors_with_tail(errors_m: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            "count": 0.0,
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "p50_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "p99_m": float("nan"),
            "max_m": float("nan"),
        }
    return {
        "count": float(errors.size),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mae_m": float(np.mean(np.abs(errors))),
        "p50_m": float(np.percentile(errors, 50)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "p99_m": float(np.percentile(errors, 99)),
        "max_m": float(np.max(errors)),
    }


def _truth_coverage(
    truth_times_s: np.ndarray,
    estimate_times_s: np.ndarray,
    *,
    max_time_delta_s: float,
) -> dict[str, float | int]:
    truth_times = np.asarray(truth_times_s, dtype=float).reshape(-1)
    estimate_times = np.asarray(estimate_times_s, dtype=float).reshape(-1)
    if truth_times.size == 0:
        return {"truth_rows": 0, "covered_truth_rows": 0, "truth_coverage_rate": float("nan")}
    if estimate_times.size == 0:
        return {
            "truth_rows": int(truth_times.size),
            "covered_truth_rows": 0,
            "truth_coverage_rate": 0.0,
        }
    indices = nearest_time_indices(estimate_times, truth_times)
    dt_s = np.abs(estimate_times[indices] - truth_times)
    covered = int(np.count_nonzero(dt_s <= float(max_time_delta_s)))
    return {
        "truth_rows": int(truth_times.size),
        "covered_truth_rows": covered,
        "truth_coverage_rate": float(covered / truth_times.size),
    }


def _trajectory_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or not {"time_s", "east_m", "north_m", "up_m"}.issubset(frame.columns):
        return np.array([], dtype=float), np.empty((0, 3), dtype=float)
    values = frame[["time_s", "east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    return values[:, 0], values[:, 1:4]


def _frame_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    column = "frame_index" if "frame_index" in frame.columns else "time_s"
    if column not in frame.columns:
        return int(len(frame))
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return int(np.unique(values).size)


def _track_switch_count(selected: pd.DataFrame) -> int:
    if selected.empty or "track_id" not in selected.columns:
        return 0
    ordered = selected.sort_values(
        [column for column in ("time_s", "frame_index", "track_index") if column in selected.columns]
    )
    track_ids = pd.to_numeric(ordered["track_id"], errors="coerce").to_numpy(dtype=float)
    track_ids = track_ids[np.isfinite(track_ids)].astype(int)
    if track_ids.size < 2:
        return 0
    return int(np.count_nonzero(track_ids[1:] != track_ids[:-1]))


def _unique_track_id_count(selected: pd.DataFrame) -> int:
    if selected.empty or "track_id" not in selected.columns:
        return 0
    track_ids = pd.to_numeric(selected["track_id"], errors="coerce").dropna().to_numpy(dtype=float)
    track_ids = track_ids[np.isfinite(track_ids)].astype(int)
    return int(np.unique(track_ids).size)


def _numeric_column_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.empty or column not in frame.columns:
        return _empty_numeric_column_stats()
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_numeric_column_stats()
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _empty_numeric_column_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "mean": float("nan"),
        "p50": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())