"""CLI for running the CV baseline with learned radar association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.baselines.learned_radar_association import (
    run_async_cv_baseline_with_learned_radar_association,
)
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records
from raft_uav.cli import (
    _baseline_metrics,
    _hypotheses_to_frame,
    _inside_truth_window,
    _records_to_frame,
    _write_trajectory_plot,
)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-run-learned-radar-association",
        description="run the CV RF/radar tracker with a learned radar association likelihood",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="JSON model produced by raft-uav-train-radar-association",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/learned-radar-association"))
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--radar-xy-std", type=float, default=25.0)
    parser.add_argument("--radar-z-std", type=float, default=35.0)
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="none")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--enable-gating", action="store_true")
    parser.add_argument("--robust-update", choices=["none", "nis-inflate"], default="none")
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=1.0)
    parser.add_argument("--radar-inflation-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.enable_gating and args.robust_update != "none":
        raise ValueError("--enable-gating and --robust-update are mutually exclusive")

    flight = select_flight(args.dataset_root, args.flight)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)

    rf = pd.DataFrame()
    rf_measurements = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        rf_measurements = rf_measurements_to_enu(rf)

    radar = _inside_truth_window(
        normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
        truth,
    )

    gate_probabilities = None
    robust_updates = None
    inflation_alphas = None
    if args.enable_gating or args.robust_update != "none":
        gate_probabilities = {"rf": args.rf_gate_prob, "radar": args.radar_gate_prob}
    if args.robust_update != "none":
        robust_updates = {"rf": args.robust_update, "radar": args.robust_update}
        inflation_alphas = {"rf": args.rf_inflation_alpha, "radar": args.radar_inflation_alpha}

    records, selected_radar = run_async_cv_baseline_with_learned_radar_association(
        rf_measurements=rf_measurements,
        radar=radar,
        model=args.model,
        acceleration_std_mps2=args.acceleration_std,
        radar_xy_std_m=args.radar_xy_std,
        radar_z_std_m=args.radar_z_std,
        gate_probabilities_by_source=gate_probabilities,
        robust_update_by_source=robust_updates,
        inflation_alpha_by_source=inflation_alphas,
        candidate_catprob_threshold=args.radar_catprob_threshold,
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
    diagnostics_columns = [
        "time_s",
        "source",
        "measurement_dim",
        "accepted",
        "update_action",
        "nis",
        "gate_threshold",
        "covariance_scale",
        "inflation_alpha",
        "residual_norm_m",
    ]
    diagnostics_frame = estimate_frame[diagnostics_columns].copy()

    flight_output = args.output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    estimates_path = flight_output / "estimates.csv"
    diagnostics_path = flight_output / "diagnostics.csv"
    selected_radar_path = flight_output / "selected_radar.csv"
    hypotheses_path = flight_output / "hypotheses.csv"
    metrics_path = flight_output / "metrics.json"
    plot_path = flight_output / "trajectory.png"

    estimate_frame.to_csv(estimates_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    selected_radar.to_csv(selected_radar_path, index=False)
    _hypotheses_to_frame(records).to_csv(hypotheses_path, index=False)

    metrics = _baseline_metrics(
        flight_name=flight.name,
        flight=flight,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        estimate_frame=estimate_frame,
        acceleration_std=args.acceleration_std,
        radar_association="learned-likelihood",
        radar_catprob_threshold=args.radar_catprob_threshold,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        track_switch_nis_ratio=0.5,
        geometry_velocity_std=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        track_bank_max_hypotheses=16,
        track_bank_max_assignments=16,
        track_bank_max_candidates=16,
        track_bank_gate_prob=0.9999999,
        track_bank_detection_prob=0.999,
        track_bank_clutter_intensity=1.0e-12,
        track_bank_prune_delta=80.0,
        smoother=args.smoother,
        smoother_lag_s=args.smoother_lag_s,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
        enable_gating=args.enable_gating,
        robust_update=args.robust_update,
        rf_gate_prob=args.rf_gate_prob,
        radar_gate_prob=args.radar_gate_prob,
        rf_inflation_alpha=args.rf_inflation_alpha,
        radar_inflation_alpha=args.radar_inflation_alpha,
    )
    metrics["learned_radar_association_model"] = str(args.model)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_trajectory_plot(plot_path, truth, rf, selected_radar, estimate_frame, flight.name)

    print(f"flight={flight.name}")
    print("radar_association=learned-likelihood")
    print(f"model_json={args.model}")
    print(f"posterior_records={len(records)}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"metrics_json={metrics_path}")
    print(f"estimates_csv={estimates_path}")
    print(f"selected_radar_csv={selected_radar_path}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
