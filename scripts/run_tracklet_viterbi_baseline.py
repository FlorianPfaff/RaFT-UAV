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
from raft_uav.baselines.tracklet_viterbi import (  # noqa: E402
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors  # noqa: E402
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
    robust_updates = None if args.robust_update == "none" else {"rf": args.robust_update, "radar": args.robust_update}
    inflation_alphas = None if robust_updates is None else {"rf": args.rf_inflation_alpha, "radar": args.radar_inflation_alpha}

    records, selected_radar = run_async_cv_baseline_with_tracklet_viterbi_association(
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
    if not records:
        raise RuntimeError(f"{flight.name} produced no posterior records")

    records = smooth_tracking_records(records, method=args.smoother, acceleration_std_mps2=args.acceleration_std, lag_s=args.smoother_lag_s)
    estimates = _records_to_frame(records)
    metrics = _metrics(flight.name, truth, rf, radar, selected_radar, estimates, args)

    out = args.output_dir / flight.name
    out.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(out / "estimates.csv", index=False)
    selected_radar.to_csv(out / "selected_radar.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"flight={flight.name}")
    print("radar_association=tracklet-viterbi")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"posterior_records={len(estimates)}")
    print(f"metrics_json={out / 'metrics.json'}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    print(f"p95_3d_m={metrics['position_error_3d']['p95_m']:.3f}")
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


def _metrics(name: str, truth: pd.DataFrame, rf: pd.DataFrame, radar: pd.DataFrame, selected: pd.DataFrame, estimates: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
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
        "posterior_records": int(len(estimates)),
        "accepted_measurements": accepted,
        "rejected_measurements": int(len(estimates) - accepted),
        "position_error_2d": summarize_errors(errors_2d),
        "position_error_3d": summarize_errors(errors_3d),
        "smoother": {"method": args.smoother, "lag_s": args.smoother_lag_s},
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
