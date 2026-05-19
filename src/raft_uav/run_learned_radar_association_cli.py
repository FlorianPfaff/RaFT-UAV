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
from raft_uav.baselines.stateful_learned_radar_association import (
    StatefulAssociationConfig,
    run_async_cv_baseline_with_stateful_learned_radar_association,
)
from raft_uav.cli import (
    _baseline_metrics,
    _hypotheses_to_frame,
    _inside_truth_window,
    _records_to_frame,
    _write_trajectory_plot,
)
from raft_uav.evaluation.diagnostics import build_diagnostic_summary
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/learned-radar-association"),
    )
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument(
        "--disable-radar-catprob-threshold",
        action="store_true",
        help="keep all radar rows before learned association scoring",
    )
    parser.add_argument("--radar-xy-std", type=float, default=25.0)
    parser.add_argument("--radar-z-std", type=float, default=35.0)
    parser.add_argument(
        "--association-mode",
        choices=["per-frame", "stateful-beam"],
        default="per-frame",
        help="learned radar association policy",
    )
    parser.add_argument("--beam-max-hypotheses", type=int, default=16)
    parser.add_argument("--beam-max-candidates", type=int, default=6)
    parser.add_argument("--beam-missed-detection-cost", type=float, default=4.0)
    parser.add_argument("--beam-consecutive-miss-cost", type=float, default=0.5)
    parser.add_argument("--beam-track-switch-cost", type=float, default=3.0)
    parser.add_argument("--beam-missing-track-id-cost", type=float, default=1.0)
    parser.add_argument(
        "--beam-lag-s",
        type=float,
        default=20.0,
        help="association look-ahead horizon before old decisions are committed",
    )
    parser.add_argument(
        "--disable-beam-missed-detection",
        action="store_true",
        help="do not keep miss branches in stateful-beam mode",
    )
    parser.add_argument("--smoother", choices=SMOOTHER_MODES, default="none")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--enable-gating", action="store_true")
    parser.add_argument("--robust-update", choices=["none", "nis-inflate"], default="none")
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument(
        "--disable-association-safety-gate",
        action="store_true",
        help="disable the hard RF/radar safety gate that turns impossible updates into misses",
    )
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument(
        "--rf-max-residual-m",
        type=float,
        default=750.0,
        help=(
            "Euclidean RF residual safety cap; with a safety NIS gate it rejects only "
            "statistically implausible updates, <=0 disables it"
        ),
    )
    parser.add_argument(
        "--radar-max-residual-m",
        type=float,
        default=0.0,
        help=(
            "Euclidean radar residual safety cap; with a safety NIS gate it rejects only "
            "statistically implausible updates, <=0 disables it"
        ),
    )
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
    safety_gate_probabilities = None
    max_residual_norms = None
    robust_updates = None
    inflation_alphas = None
    if args.enable_gating or args.robust_update != "none":
        gate_probabilities = {"rf": args.rf_gate_prob, "radar": args.radar_gate_prob}
    if not args.disable_association_safety_gate:
        safety_gate_probabilities = {
            "rf": args.rf_safety_gate_prob,
            "radar": args.radar_safety_gate_prob,
        }
        max_residual_norms = {
            "rf": None if args.rf_max_residual_m <= 0.0 else args.rf_max_residual_m,
            "radar": None
            if args.radar_max_residual_m <= 0.0
            else args.radar_max_residual_m,
        }
    if args.robust_update != "none":
        robust_updates = {"rf": args.robust_update, "radar": args.robust_update}
        inflation_alphas = {"rf": args.rf_inflation_alpha, "radar": args.radar_inflation_alpha}

    candidate_catprob_threshold = (
        None if args.disable_radar_catprob_threshold else args.radar_catprob_threshold
    )

    if args.association_mode == "stateful-beam":
        association_name = "stateful-learned-likelihood"
        records, selected_radar = run_async_cv_baseline_with_stateful_learned_radar_association(
            rf_measurements=rf_measurements,
            radar=radar,
            model=args.model,
            acceleration_std_mps2=args.acceleration_std,
            radar_xy_std_m=args.radar_xy_std,
            radar_z_std_m=args.radar_z_std,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=StatefulAssociationConfig(
                max_hypotheses=args.beam_max_hypotheses,
                max_candidates_per_hypothesis=args.beam_max_candidates,
                missed_detection_cost=args.beam_missed_detection_cost,
                consecutive_miss_cost=args.beam_consecutive_miss_cost,
                track_switch_cost=args.beam_track_switch_cost,
                missing_track_id_cost=args.beam_missing_track_id_cost,
                allow_missed_detection=not args.disable_beam_missed_detection,
                lag_s=args.beam_lag_s,
            ),
        )
    else:
        association_name = "learned-likelihood"
        records, selected_radar = run_async_cv_baseline_with_learned_radar_association(
            rf_measurements=rf_measurements,
            radar=radar,
            model=args.model,
            acceleration_std_mps2=args.acceleration_std,
            radar_xy_std_m=args.radar_xy_std,
            radar_z_std_m=args.radar_z_std,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
            candidate_catprob_threshold=candidate_catprob_threshold,
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
        "safety_gate_threshold",
        "residual_gate_threshold_m",
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
    diagnostic_summary_path = flight_output / "diagnostic_summary.json"
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
        radar_association=association_name,
        radar_catprob_threshold=(
            float("nan") if args.disable_radar_catprob_threshold else args.radar_catprob_threshold
        ),
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        track_switch_nis_ratio=0.5,
        geometry_velocity_std=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        track_bank_max_hypotheses=args.beam_max_hypotheses,
        track_bank_max_assignments=args.beam_max_candidates,
        track_bank_max_candidates=args.beam_max_candidates,
        track_bank_gate_prob=0.9999999,
        track_bank_detection_prob=0.999,
        track_bank_clutter_intensity=1.0e-12,
        track_bank_prune_delta=80.0,
        stable_segment_min_frames=100,
        stable_segment_max_transition_speed_mps=65.0,
        stable_segment_range_gate_m=800.0,
        stable_segment_interpolation_max_gap_s=5.0,
        stable_segment_interpolation_max_speed_mps=65.0,
        stable_segment_interpolation_std_scale=2.0,
        smoother=args.smoother,
        smoother_lag_s=args.smoother_lag_s,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
        enable_gating=args.enable_gating,
        robust_update=args.robust_update,
        rf_gate_prob=args.rf_gate_prob,
        radar_gate_prob=args.radar_gate_prob,
        enable_association_safety_gate=not args.disable_association_safety_gate,
        rf_safety_gate_prob=args.rf_safety_gate_prob,
        radar_safety_gate_prob=args.radar_safety_gate_prob,
        rf_max_residual_m=args.rf_max_residual_m,
        radar_max_residual_m=args.radar_max_residual_m,
        rf_inflation_alpha=args.rf_inflation_alpha,
        radar_inflation_alpha=args.radar_inflation_alpha,
    )
    metrics["learned_radar_association_model"] = str(args.model)
    metrics["learned_radar_association_mode"] = args.association_mode
    if args.disable_radar_catprob_threshold:
        metrics["radar_catprob_threshold"] = None
    if args.association_mode == "stateful-beam":
        metrics["stateful_learned_association"] = {
            "max_hypotheses": int(args.beam_max_hypotheses),
            "max_candidates_per_hypothesis": int(args.beam_max_candidates),
            "missed_detection_cost": float(args.beam_missed_detection_cost),
            "consecutive_miss_cost": float(args.beam_consecutive_miss_cost),
            "track_switch_cost": float(args.beam_track_switch_cost),
            "missing_track_id_cost": float(args.beam_missing_track_id_cost),
            "allow_missed_detection": not args.disable_beam_missed_detection,
            "lag_s": float(args.beam_lag_s),
        }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    diagnostic_summary = build_diagnostic_summary(
        estimate_frame=estimate_frame,
        selected_radar=selected_radar,
        truth=truth,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
    )
    diagnostic_summary_path.write_text(
        json.dumps(diagnostic_summary, indent=2),
        encoding="utf-8",
    )
    _write_trajectory_plot(plot_path, truth, rf, selected_radar, estimate_frame, flight.name)

    print(f"flight={flight.name}")
    print(f"radar_association={association_name}")
    print(f"learned_radar_association_mode={args.association_mode}")
    print(f"model_json={args.model}")
    print(f"posterior_records={len(records)}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"metrics_json={metrics_path}")
    print(f"diagnostic_summary_json={diagnostic_summary_path}")
    print(f"estimates_csv={estimates_path}")
    print(f"selected_radar_csv={selected_radar_path}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
