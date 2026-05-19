"""LOFO-safe SOTA leaderboard runner for RaFT-UAV.

The runner executes each method on a held-out flight while fitting any learned or
calibrated artifacts only on the remaining flights.  It then recomputes all
reporting metrics on a common truth-time grid and writes both method-order
aggregate summaries and a coverage-aware leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.evaluation.metrics import nearest_time_indices, position_errors_m
from raft_uav.io.aerpaw import discover_flights, normalize_truth, read_truth, select_flight

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MethodSpec:
    """Description of one leakage-safe leaderboard method."""

    name: str
    label: str
    runner: str
    family: str
    radar_association: str = ""
    smoother: str = "none"
    robust_update: str = "none"
    learned_association_mode: str = ""
    requires_learned_radar_model: bool = False
    requires_hetero_model: bool = False
    disable_rf_anchor: bool = False
    disable_radar_catprob_threshold: bool = False
    offline_diagnostic: bool = False


@dataclass(frozen=True)
class RunEvaluation:
    """Per-fold evaluation payload used for pooled aggregation."""

    row: dict[str, object]
    errors_2d_m: np.ndarray
    errors_3d_m: np.ndarray
    covered_truth_rows: int
    truth_rows: int


METHODS: dict[str, MethodSpec] = {
    "cv_catprob": MethodSpec(
        name="cv_catprob",
        label="CV catprob",
        runner="baseline",
        family="cv",
        radar_association="catprob",
    ),
    "cv_rf_anchored_nis_fixed_lag": MethodSpec(
        name="cv_rf_anchored_nis_fixed_lag",
        label="CV RF-anchored NIS fixed-lag",
        runner="baseline",
        family="cv",
        radar_association="rf-anchored-nis",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "cv_rf_gated_nis_fixed_lag": MethodSpec(
        name="cv_rf_gated_nis_fixed_lag",
        label="CV RF-gated NIS fixed-lag",
        runner="baseline",
        family="cv",
        radar_association="rf-gated-nis",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "cv_pda_fixed_lag": MethodSpec(
        name="cv_pda_fixed_lag",
        label="CV PDA fixed-lag",
        runner="baseline",
        family="cv",
        radar_association="pda-mixture",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "cv_track_bank_fixed_lag": MethodSpec(
        name="cv_track_bank_fixed_lag",
        label="CV MHT track-bank fixed-lag",
        runner="baseline",
        family="cv",
        radar_association="track-bank",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "cv_stable_segments_hybrid_fixed_lag": MethodSpec(
        name="cv_stable_segments_hybrid_fixed_lag",
        label="CV stable-segment hybrid fixed-lag",
        runner="baseline",
        family="cv",
        radar_association="stable-segments-hybrid",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "learned_per_frame_fixed_lag": MethodSpec(
        name="learned_per_frame_fixed_lag",
        label="Learned radar association fixed-lag",
        runner="learned",
        family="learned-association",
        smoother="fixed-lag",
        robust_update="nis-inflate",
        learned_association_mode="per-frame",
        requires_learned_radar_model=True,
    ),
    "learned_stateful_beam_fixed_lag": MethodSpec(
        name="learned_stateful_beam_fixed_lag",
        label="Stateful learned radar association fixed-lag",
        runner="learned",
        family="learned-association",
        smoother="fixed-lag",
        robust_update="nis-inflate",
        learned_association_mode="stateful-beam",
        requires_learned_radar_model=True,
    ),
    "tracklet_viterbi_online": MethodSpec(
        name="tracklet_viterbi_online",
        label="Tracklet-Viterbi online",
        runner="tracklet-viterbi",
        family="tracklet-viterbi",
        smoother="none",
        robust_update="nis-inflate",
    ),
    "tracklet_viterbi_fixed_lag": MethodSpec(
        name="tracklet_viterbi_fixed_lag",
        label="Tracklet-Viterbi fixed-lag",
        runner="tracklet-viterbi",
        family="tracklet-viterbi",
        smoother="fixed-lag",
        robust_update="nis-inflate",
    ),
    "tracklet_viterbi_no_rf_anchor_fixed_lag": MethodSpec(
        name="tracklet_viterbi_no_rf_anchor_fixed_lag",
        label="Tracklet-Viterbi no RF-anchor fixed-lag",
        runner="tracklet-viterbi",
        family="tracklet-viterbi",
        smoother="fixed-lag",
        robust_update="nis-inflate",
        disable_rf_anchor=True,
    ),
    "tracklet_viterbi_rts_offline": MethodSpec(
        name="tracklet_viterbi_rts_offline",
        label="Tracklet-Viterbi full RTS diagnostic",
        runner="tracklet-viterbi",
        family="tracklet-viterbi",
        smoother="rts",
        robust_update="nis-inflate",
        offline_diagnostic=True,
    ),
    "imm_catprob": MethodSpec(
        name="imm_catprob",
        label="IMM catprob",
        runner="imm",
        family="imm",
    ),
    "imm_catprob_robust": MethodSpec(
        name="imm_catprob_robust",
        label="IMM catprob robust",
        runner="imm",
        family="imm",
        robust_update="nis-inflate",
    ),
    "hetero_cv_fixed_lag": MethodSpec(
        name="hetero_cv_fixed_lag",
        label="Heteroscedastic CV fixed-lag",
        runner="hetero",
        family="heteroscedastic-covariance",
        smoother="fixed-lag",
        requires_hetero_model=True,
    ),
}

DEFAULT_METHODS = [
    "cv_catprob",
    "cv_rf_anchored_nis_fixed_lag",
    "learned_stateful_beam_fixed_lag",
    "tracklet_viterbi_fixed_lag",
    "imm_catprob_robust",
    "hetero_cv_fixed_lag",
]

REFERENCE_ROWS = {
    "aerpaw_cv_kf_full_coverage": {
        "method": "aerpaw_cv_kf_full_coverage",
        "label": "AERPAW RF+radar CV-KF reference, 100% coverage",
        "runner": "reference",
        "family": "published-reference",
        "folds": 0,
        "truth_coverage_rate": 1.0,
        "error_3d_mae_m": 21.9,
        "reference_metric": "reported mean 3D error at 100% coverage",
        "reference_only": True,
    },
    "aerpaw_cv_kf_updated_coverage": {
        "method": "aerpaw_cv_kf_updated_coverage",
        "label": "AERPAW RF+radar CV-KF reference, updated estimates",
        "runner": "reference",
        "family": "published-reference",
        "folds": 0,
        "truth_coverage_rate": 0.952,
        "error_3d_mae_m": 21.6,
        "reference_metric": "reported mean 3D error at 95.2% updated-estimate coverage",
        "reference_only": True,
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested methods in strict leave-one-flight-out mode."""

    parser = argparse.ArgumentParser(
        prog="raft-uav-lofo-leaderboard",
        description=(
            "run RaFT-UAV method families in leakage-safe leave-one-flight-out mode "
            "and write fold summaries, aggregate summaries, and a coverage-aware leaderboard"
        ),
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_sota_leaderboard"))
    parser.add_argument("--flights", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=DEFAULT_METHODS,
        help="method rows to run; methods that fit models use only non-held-out flights",
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=float, default=0.0)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--learned-l2", type=float, default=1.0e-3)
    parser.add_argument("--learned-max-iter", type=int, default=500)
    parser.add_argument("--learned-positive-gate-m", type=float, default=50.0)
    parser.add_argument("--learned-truth-gate-m", type=float, default=150.0)
    parser.add_argument("--learned-truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--beam-max-hypotheses", type=int, default=16)
    parser.add_argument("--beam-max-candidates", type=int, default=6)
    parser.add_argument("--beam-missed-detection-cost", type=float, default=4.0)
    parser.add_argument("--beam-consecutive-miss-cost", type=float, default=0.5)
    parser.add_argument("--beam-track-switch-cost", type=float, default=3.0)
    parser.add_argument("--beam-missing-track-id-cost", type=float, default=1.0)
    parser.add_argument("--beam-lag-s", type=float, default=20.0)
    parser.add_argument("--tracklet-max-candidates-per-frame", type=int, default=8)
    parser.add_argument("--tracklet-missed-detection-cost", type=float, default=7.0)
    parser.add_argument("--tracklet-track-switch-cost", type=float, default=8.0)
    parser.add_argument("--tracklet-catprob-weight", type=float, default=2.5)
    parser.add_argument("--tracklet-anchor-nis-weight", type=float, default=0.35)
    parser.add_argument("--tracklet-transition-nis-weight", type=float, default=1.0)
    parser.add_argument("--tracklet-velocity-nis-weight", type=float, default=0.15)
    parser.add_argument("--tracklet-max-speed-mps", type=float, default=55.0)
    parser.add_argument("--tracklet-range-gate-m", type=float, default=850.0)
    parser.add_argument("--target-coverage", type=float, default=1.0)
    parser.add_argument(
        "--coverage-penalty-m",
        type=float,
        default=1000.0,
        help=(
            "meters added to the leaderboard score per unit coverage shortfall; "
            "default makes a 1 percentage-point coverage loss cost 10 m"
        ),
    )
    parser.add_argument(
        "--include-reference-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include fixed AERPAW reference rows in leaderboard.csv only",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    flights = _selected_flight_names(args.dataset_root, args.flights)
    methods = [METHODS[name] for name in args.methods]
    if len(flights) < 2 and any(_needs_training(method) for method in methods):
        raise ValueError("LOFO model fitting needs at least two flights")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    evaluations: dict[str, list[RunEvaluation]] = {method.name: [] for method in methods}

    for heldout in flights:
        train_flights = [flight for flight in flights if flight != heldout]
        fold_dir = args.output_dir / f"heldout_{_slug(heldout)}"
        model_paths = _prepare_fold_models(args, methods, train_flights, fold_dir)
        for method in methods:
            run_dir = fold_dir / method.name
            metrics_path = run_dir / heldout / "metrics.json"
            if not (args.skip_existing and metrics_path.exists()):
                _run_method(args, method, heldout, run_dir, model_paths)
            evaluation = _evaluate_run(
                dataset_root=args.dataset_root,
                flight=heldout,
                method=method,
                metrics_path=metrics_path,
                max_eval_time_delta_s=args.max_eval_time_delta_s,
                train_flights=train_flights,
                target_coverage=args.target_coverage,
                coverage_penalty_m=args.coverage_penalty_m,
            )
            fold_rows.append(evaluation.row)
            evaluations[method.name].append(evaluation)

    aggregate_rows = _aggregate_method_rows(
        methods,
        evaluations,
        target_coverage=args.target_coverage,
        coverage_penalty_m=args.coverage_penalty_m,
    )
    leaderboard_rows = _leaderboard_rows(
        aggregate_rows,
        include_reference_rows=args.include_reference_rows,
        target_coverage=args.target_coverage,
        coverage_penalty_m=args.coverage_penalty_m,
    )

    _write_csv(args.output_dir / "fold_summary.csv", fold_rows)
    _write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    _write_csv(args.output_dir / "leaderboard.csv", leaderboard_rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "flights": flights,
                "target_coverage": float(args.target_coverage),
                "coverage_penalty_m": float(args.coverage_penalty_m),
                "methods": [asdict(method) for method in methods],
                "fold_rows": fold_rows,
                "aggregate_rows": aggregate_rows,
                "leaderboard_rows": leaderboard_rows,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(fold_rows)} fold rows to {args.output_dir / 'fold_summary.csv'}")
    print(
        f"wrote {len(aggregate_rows)} aggregate rows to "
        f"{args.output_dir / 'aggregate_summary.csv'}"
    )
    print(
        f"wrote {len(leaderboard_rows)} leaderboard rows to "
        f"{args.output_dir / 'leaderboard.csv'}"
    )
    return 0


def _prepare_fold_models(
    args: argparse.Namespace,
    methods: Sequence[MethodSpec],
    train_flights: Sequence[str],
    fold_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if any(method.requires_learned_radar_model for method in methods):
        path = fold_dir / "models" / "learned_radar_association.json"
        _train_learned_radar_model(args, train_flights, path)
        paths["learned_radar"] = path
    if any(method.requires_hetero_model for method in methods):
        path = fold_dir / "models" / "heteroscedastic_uncertainty.json"
        _train_heteroscedastic_model(args, train_flights, path)
        paths["hetero"] = path
    return paths


def _train_learned_radar_model(
    args: argparse.Namespace,
    train_flights: Sequence[str],
    model_path: Path,
) -> None:
    if args.skip_existing and model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command: list[object] = [
        sys.executable,
        "-m",
        "raft_uav.train_radar_association_cli",
        args.dataset_root,
        "--output-model",
        model_path,
        "--acceleration-std",
        args.acceleration_std,
        "--radar-catprob-threshold",
        args.candidate_threshold,
        "--positive-gate-m",
        args.learned_positive_gate_m,
        "--truth-gate-m",
        args.learned_truth_gate_m,
        "--truth-time-gate-s",
        args.learned_truth_time_gate_s,
        "--l2",
        args.learned_l2,
        "--max-iter",
        args.learned_max_iter,
    ]
    for flight in train_flights:
        command.extend(["--flight", flight])
    _run(command)


def _train_heteroscedastic_model(
    args: argparse.Namespace,
    train_flights: Sequence[str],
    model_path: Path,
) -> None:
    if args.skip_existing and model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command: list[object] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_heteroscedastic_uncertainty.py"),
        args.dataset_root,
        "--output",
        model_path,
        "--ridge-lambda",
        args.ridge_lambda,
        "--max-time-delta-s",
        args.max_eval_time_delta_s,
    ]
    for flight in train_flights:
        command.extend(["--flight", flight])
    _run(command)


def _run_method(
    args: argparse.Namespace,
    method: MethodSpec,
    flight: str,
    run_dir: Path,
    model_paths: dict[str, Path],
) -> None:
    if method.runner == "baseline":
        command: list[object] = [
            sys.executable,
            "-m",
            "raft_uav.cli",
            "run-baseline",
            args.dataset_root,
            "--flight",
            flight,
            "--output-dir",
            run_dir,
            "--acceleration-std",
            args.acceleration_std,
            "--radar-association",
            method.radar_association,
            "--radar-catprob-threshold",
            args.candidate_threshold,
            "--smoother",
            method.smoother,
            "--smoother-lag-s",
            args.fixed_lag_s,
            "--max-eval-time-delta-s",
            args.max_eval_time_delta_s,
        ]
        _append_robust_options(command, args, method)
        _run(command)
        return

    if method.runner == "learned":
        model = model_paths.get("learned_radar")
        if model is None:
            raise RuntimeError("learned radar model was not prepared")
        command = [
            sys.executable,
            "-m",
            "raft_uav.run_learned_radar_association_cli",
            args.dataset_root,
            "--flight",
            flight,
            "--model",
            model,
            "--output-dir",
            run_dir,
            "--association-mode",
            method.learned_association_mode,
            "--radar-catprob-threshold",
            args.candidate_threshold,
            "--acceleration-std",
            args.acceleration_std,
            "--smoother",
            method.smoother,
            "--smoother-lag-s",
            args.fixed_lag_s,
            "--max-eval-time-delta-s",
            args.max_eval_time_delta_s,
            "--beam-max-hypotheses",
            args.beam_max_hypotheses,
            "--beam-max-candidates",
            args.beam_max_candidates,
            "--beam-missed-detection-cost",
            args.beam_missed_detection_cost,
            "--beam-consecutive-miss-cost",
            args.beam_consecutive_miss_cost,
            "--beam-track-switch-cost",
            args.beam_track_switch_cost,
            "--beam-missing-track-id-cost",
            args.beam_missing_track_id_cost,
            "--beam-lag-s",
            args.beam_lag_s,
        ]
        if method.disable_radar_catprob_threshold:
            command.append("--disable-radar-catprob-threshold")
        _append_robust_options(command, args, method)
        _run(command)
        return

    if method.runner == "tracklet-viterbi":
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_tracklet_viterbi_baseline.py"),
            args.dataset_root,
            "--flight",
            flight,
            "--output-dir",
            run_dir,
            "--acceleration-std",
            args.acceleration_std,
            "--radar-catprob-threshold",
            args.candidate_threshold,
            "--max-candidates-per-frame",
            args.tracklet_max_candidates_per_frame,
            "--missed-detection-cost",
            args.tracklet_missed_detection_cost,
            "--track-switch-cost",
            args.tracklet_track_switch_cost,
            "--catprob-weight",
            args.tracklet_catprob_weight,
            "--anchor-nis-weight",
            args.tracklet_anchor_nis_weight,
            "--transition-nis-weight",
            args.tracklet_transition_nis_weight,
            "--velocity-nis-weight",
            args.tracklet_velocity_nis_weight,
            "--max-speed-mps",
            args.tracklet_max_speed_mps,
            "--range-gate-m",
            args.tracklet_range_gate_m,
            "--smoother",
            method.smoother,
            "--smoother-lag-s",
            args.fixed_lag_s,
            "--max-eval-time-delta-s",
            args.max_eval_time_delta_s,
        ]
        _append_robust_options(command, args, method)
        if method.disable_rf_anchor:
            command.append("--disable-rf-anchor")
        _run(command)
        return

    if method.runner == "imm":
        command = [
            sys.executable,
            "-m",
            "raft_uav.imm_cli",
            args.dataset_root,
            "--flight",
            flight,
            "--output-dir",
            run_dir,
            "--tracker",
            "imm",
            "--radar-selection",
            "catprob",
            "--radar-catprob-threshold",
            args.candidate_threshold,
            "--acceleration-std",
            args.acceleration_std,
            "--max-eval-time-delta-s",
            args.max_eval_time_delta_s,
        ]
        _append_robust_options(command, args, method)
        _run(command)
        return

    if method.runner == "hetero":
        model = model_paths.get("hetero")
        if model is None:
            raise RuntimeError("heteroscedastic uncertainty model was not prepared")
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_heteroscedastic_baseline.py"),
            args.dataset_root,
            "--flight",
            flight,
            "--uncertainty-model",
            model,
            "--output-dir",
            run_dir,
            "--radar-selection",
            "catprob",
            "--radar-catprob-threshold",
            args.candidate_threshold,
            "--acceleration-std",
            args.acceleration_std,
            "--max-eval-time-delta-s",
            args.max_eval_time_delta_s,
        ]
        if method.smoother != "none":
            command.extend(["--smoother", method.smoother, "--smoother-lag-s", args.fixed_lag_s])
        _run(command)
        return

    raise ValueError(f"unknown method runner {method.runner!r}")


def _append_robust_options(
    command: list[object],
    args: argparse.Namespace,
    method: MethodSpec,
) -> None:
    command.extend(
        [
            "--robust-update",
            method.robust_update,
            "--rf-gate-prob",
            args.rf_gate_prob,
            "--radar-gate-prob",
            args.radar_gate_prob,
            "--rf-safety-gate-prob",
            args.rf_safety_gate_prob,
            "--radar-safety-gate-prob",
            args.radar_safety_gate_prob,
            "--rf-max-residual-m",
            args.rf_max_residual_m,
            "--radar-max-residual-m",
            args.radar_max_residual_m,
            "--rf-inflation-alpha",
            args.rf_inflation_alpha,
            "--radar-inflation-alpha",
            args.radar_inflation_alpha,
        ]
    )


def _evaluate_run(
    *,
    dataset_root: Path,
    flight: str,
    method: MethodSpec,
    metrics_path: Path,
    max_eval_time_delta_s: float,
    train_flights: Sequence[str],
    target_coverage: float,
    coverage_penalty_m: float,
) -> RunEvaluation:
    metrics = _load_metrics(metrics_path)
    estimates = pd.read_csv(metrics_path.parent / "estimates.csv")
    selected_radar = _read_optional_csv(metrics_path.parent / "selected_radar.csv")
    truth = _load_truth(dataset_root, flight)

    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    estimate_positions = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors_2d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=2,
    )
    errors_3d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=3,
    )
    coverage = _truth_coverage(truth_times, estimate_times, max_time_delta_s=max_eval_time_delta_s)
    coverage_rate = _finite_or_nan(coverage["truth_coverage_rate"])
    error_2d = _summarize_scalar_errors(errors_2d)
    error_3d = _summarize_scalar_errors(errors_3d)
    score = _coverage_score(error_3d["mae_m"], coverage_rate, target_coverage, coverage_penalty_m)

    smoother = metrics.get("smoother") or {}
    robust_update = metrics.get("robust_update") or {}
    diagnostics = metrics.get("selected_radar_diagnostics") or {}
    row: dict[str, object] = {
        "heldout_flight": flight,
        "train_flights": ";".join(train_flights),
        "method": method.name,
        "label": method.label,
        "runner": method.runner,
        "family": method.family,
        "radar_association": metrics.get(
            "radar_association",
            metrics.get("radar_selection", method.radar_association),
        ),
        "robust_update": _robust_name(robust_update) or method.robust_update,
        "smoother": smoother.get("method", "") if isinstance(smoother, dict) else method.smoother,
        "smoother_lag_s": smoother.get("lag_s", "") if isinstance(smoother, dict) else "",
        "lofo_safe": True,
        "offline_diagnostic": bool(method.offline_diagnostic),
        "reference_only": False,
        "posterior_records": int(metrics.get("posterior_records", len(estimates))),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", len(selected_radar))),
        "accepted_measurements": int(metrics.get("accepted_measurements", 0)),
        "rejected_measurements": int(metrics.get("rejected_measurements", 0)),
        "selected_radar_frame_count": _frame_count(selected_radar),
        "selected_radar_track_switch_count": _track_switch_count(selected_radar),
        "selected_radar_unique_track_ids": _unique_track_id_count(selected_radar),
        "score_3d_mean_m": score,
        "target_coverage": float(target_coverage),
        "coverage_penalty_m": float(coverage_penalty_m),
        "coverage_shortfall_rate": _coverage_shortfall(coverage_rate, target_coverage),
        "metrics_path": str(metrics_path),
    }
    if isinstance(diagnostics, dict):
        row["selected_radar_truth_coverage_rate"] = _nested_float(
            diagnostics,
            "truth_coverage",
            "truth_coverage_rate",
        )
        row["selected_radar_frame_coverage_rate"] = _optional_float(
            diagnostics.get("radar_frame_coverage_rate")
        )
    row.update(_prefixed_summary("error_2d", error_2d))
    row.update(_prefixed_summary("error_3d", error_3d))
    row.update(coverage)
    row.update(_nis_summary(estimates))
    return RunEvaluation(
        row=row,
        errors_2d_m=errors_2d,
        errors_3d_m=errors_3d,
        covered_truth_rows=int(coverage["covered_truth_rows"]),
        truth_rows=int(coverage["truth_rows"]),
    )


def _aggregate_method_rows(
    methods: Sequence[MethodSpec],
    evaluations: dict[str, Sequence[RunEvaluation]],
    *,
    target_coverage: float,
    coverage_penalty_m: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        runs = list(evaluations.get(method.name, []))
        errors_2d = _concat([run.errors_2d_m for run in runs])
        errors_3d = _concat([run.errors_3d_m for run in runs])
        truth_rows = int(sum(run.truth_rows for run in runs))
        covered = int(sum(run.covered_truth_rows for run in runs))
        coverage_rate = float(covered / truth_rows) if truth_rows else float("nan")
        error_2d = _summarize_scalar_errors(errors_2d)
        error_3d = _summarize_scalar_errors(errors_3d)
        row: dict[str, object] = {
            "method": method.name,
            "label": method.label,
            "runner": method.runner,
            "family": method.family,
            "folds": len(runs),
            "lofo_safe": True,
            "offline_diagnostic": bool(method.offline_diagnostic),
            "reference_only": False,
            "posterior_records": int(sum(int(run.row.get("posterior_records", 0)) for run in runs)),
            "selected_radar_rows": int(
                sum(int(run.row.get("selected_radar_rows", 0)) for run in runs)
            ),
            "truth_rows": truth_rows,
            "covered_truth_rows": covered,
            "truth_coverage_rate": coverage_rate,
            "target_coverage": float(target_coverage),
            "coverage_penalty_m": float(coverage_penalty_m),
            "coverage_shortfall_rate": _coverage_shortfall(coverage_rate, target_coverage),
        }
        row.update(_prefixed_summary("error_2d", error_2d))
        row.update(_prefixed_summary("error_3d", error_3d))
        row["score_3d_mean_m"] = _coverage_score(
            error_3d["mae_m"],
            coverage_rate,
            target_coverage,
            coverage_penalty_m,
        )
        row["score_3d_rmse_m"] = _coverage_score(
            error_3d["rmse_m"],
            coverage_rate,
            target_coverage,
            coverage_penalty_m,
        )
        rows.append(row)
    _assign_ranks(rows, score_key="score_3d_mean_m", rank_key="rank_score_3d")
    _assign_ranks(rows, score_key="error_3d_rmse_m", rank_key="rank_rmse_3d")
    return rows


def _leaderboard_rows(
    aggregate_rows: Sequence[dict[str, object]],
    *,
    include_reference_rows: bool,
    target_coverage: float,
    coverage_penalty_m: float,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in aggregate_rows]
    if include_reference_rows:
        for reference in REFERENCE_ROWS.values():
            row = dict(reference)
            coverage_rate = _finite_or_nan(row["truth_coverage_rate"])
            row["target_coverage"] = float(target_coverage)
            row["coverage_penalty_m"] = float(coverage_penalty_m)
            row["coverage_shortfall_rate"] = _coverage_shortfall(coverage_rate, target_coverage)
            row["score_3d_mean_m"] = _coverage_score(
                _finite_or_nan(row.get("error_3d_mae_m")),
                coverage_rate,
                target_coverage,
                coverage_penalty_m,
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            _sort_number(row.get("score_3d_mean_m")),
            bool(row.get("offline_diagnostic", False)),
            -_sort_number(row.get("truth_coverage_rate"), default=0.0),
            str(row.get("method", "")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["leaderboard_rank"] = rank
    return rows


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


def _summarize_scalar_errors(errors_m: np.ndarray) -> dict[str, float]:
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


def _selected_flight_names(dataset_root: Path, requested: Sequence[str] | None) -> list[str]:
    if requested:
        return [select_flight(dataset_root, name).name for name in requested]
    return [
        flight.name for flight in discover_flights(dataset_root) if flight.truth_txt is not None
    ]


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def _load_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _nis_summary(estimates: pd.DataFrame) -> dict[str, object]:
    if "nis" not in estimates.columns:
        return {}
    out: dict[str, object] = {}
    source = (
        estimates["source"]
        if "source" in estimates.columns
        else pd.Series(["all"] * len(estimates))
    )
    for name, group in estimates.groupby(source):
        values = pd.to_numeric(group["nis"], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[f"nis_{name}_count"] = int(values.size)
            out[f"nis_{name}_mean"] = float(np.mean(values))
            out[f"nis_{name}_p95"] = float(np.percentile(values, 95))
    return out


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
    sort_columns = [
        column for column in ("time_s", "frame_index", "track_index") if column in selected.columns
    ]
    ordered = selected.sort_values(sort_columns) if sort_columns else selected
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


def _coverage_score(
    error_m: float,
    coverage_rate: float,
    target_coverage: float,
    coverage_penalty_m: float,
) -> float:
    if not np.isfinite(error_m):
        return float("inf")
    return float(error_m + coverage_penalty_m * _coverage_shortfall(coverage_rate, target_coverage))


def _coverage_shortfall(coverage_rate: float, target_coverage: float) -> float:
    if not np.isfinite(coverage_rate):
        return float(target_coverage)
    return float(max(0.0, float(target_coverage) - float(coverage_rate)))


def _assign_ranks(rows: list[dict[str, object]], *, score_key: str, rank_key: str) -> None:
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            _sort_number(item[1].get(score_key)),
            bool(item[1].get("offline_diagnostic", False)),
            -_sort_number(item[1].get("truth_coverage_rate"), default=0.0),
            str(item[1].get("method", "")),
        ),
    )
    for rank, (original_index, _) in enumerate(ranked, start=1):
        rows[original_index][rank_key] = rank


def _sort_number(value: object, *, default: float = float("inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _optional_float(value: object) -> float:
    number = _finite_or_nan(value)
    return number if np.isfinite(number) else float("nan")


def _nested_float(mapping: dict[str, object], first: str, second: str) -> float:
    child = mapping.get(first)
    if not isinstance(child, dict):
        return float("nan")
    return _optional_float(child.get(second))


def _robust_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        method = value.get("method")
        return "" if method is None else str(method)
    return ""


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    valid = [
        np.asarray(array, dtype=float).reshape(-1)
        for array in arrays
        if np.asarray(array).size
    ]
    return np.concatenate(valid) if valid else np.array([], dtype=float)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _needs_training(method: MethodSpec) -> bool:
    return method.requires_learned_radar_model or method.requires_hetero_model


def _run(command: Sequence[object]) -> None:
    command_text = [str(item) for item in command]
    print(" ".join(command_text), flush=True)
    subprocess.run(command_text, check=True, env=_subprocess_env())


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
