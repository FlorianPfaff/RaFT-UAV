"""Leakage-safe leave-one-flight-out SOTA evaluation runner for RaFT-UAV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402
from raft_uav.calibration.nis_covariance import (  # noqa: E402
    ENV_NIS_COVARIANCE_CALIBRATION_JSON,
    NIS_COVARIANCE_CALIBRATION_METHODS,
    fit_nis_covariance_calibration_from_paths,
    write_nis_covariance_calibration,
)
from raft_uav.evaluation.oracle_gap_decomposition import (  # noqa: E402
    selected_track_stability_metrics,
)
from raft_uav.evaluation.metrics import nearest_time_indices, position_errors_m  # noqa: E402
from raft_uav.io.aerpaw import discover_flights, normalize_truth, read_truth, select_flight  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    """Description of one protocol method."""

    name: str
    runner: str
    label: str
    association: str = "catprob"
    fixed_lag: bool = False
    robust: bool = False
    rts: bool = False
    nis_calibrated: bool = False
    needs_association_model: bool = False
    replay_tracker: str = "cv"


@dataclass(frozen=True)
class RunEvaluation:
    """Per-fold evaluation payload used for pooled aggregation."""

    row: dict[str, object]
    errors_2d_m: np.ndarray
    errors_3d_m: np.ndarray
    covered_truth_rows: int
    truth_rows: int


METHODS: dict[str, MethodSpec] = {
    "cv_catprob": MethodSpec("cv_catprob", "baseline", "CV catprob"),
    "cv_prediction_nis_fixed_lag": MethodSpec(
        "cv_prediction_nis_fixed_lag",
        "baseline",
        "CV prediction-NIS fixed-lag",
        association="prediction-nis",
        fixed_lag=True,
        robust=True,
    ),
    "cv_rf_anchored_nis_fixed_lag": MethodSpec(
        "cv_rf_anchored_nis_fixed_lag",
        "baseline",
        "CV RF-anchored NIS fixed-lag",
        association="rf-anchored-nis",
        fixed_lag=True,
        robust=True,
    ),
    "cv_rf_gated_nis_fixed_lag": MethodSpec(
        "cv_rf_gated_nis_fixed_lag",
        "baseline",
        "CV RF-gated NIS fixed-lag",
        association="rf-gated-nis",
        fixed_lag=True,
        robust=True,
    ),
    "cv_pda_fixed_lag": MethodSpec(
        "cv_pda_fixed_lag",
        "baseline",
        "CV PDA fixed-lag",
        association="pda-mixture",
        fixed_lag=True,
        robust=True,
    ),
    "cv_track_bank_fixed_lag": MethodSpec(
        "cv_track_bank_fixed_lag",
        "baseline",
        "CV MHT track-bank fixed-lag",
        association="track-bank",
        fixed_lag=True,
        robust=True,
    ),
    "cv_stable_segments_fixed_lag": MethodSpec(
        "cv_stable_segments_fixed_lag",
        "baseline",
        "CV stable radar segments fixed-lag",
        association="stable-segments",
        fixed_lag=True,
        robust=True,
    ),
    "cv_stable_segments_hybrid_fixed_lag": MethodSpec(
        "cv_stable_segments_hybrid_fixed_lag",
        "baseline",
        "CV stable radar segments hybrid fixed-lag",
        association="stable-segments-hybrid",
        fixed_lag=True,
        robust=True,
    ),
    "cv_stable_segments_interpolated_fixed_lag": MethodSpec(
        "cv_stable_segments_interpolated_fixed_lag",
        "baseline",
        "CV interpolated stable radar segments fixed-lag",
        association="stable-segments-interpolated",
        fixed_lag=True,
        robust=True,
    ),
    "imm_catprob": MethodSpec("imm_catprob", "imm", "IMM catprob"),
    "imm_catprob_robust": MethodSpec(
        "imm_catprob_robust", "imm", "IMM catprob robust", robust=True
    ),
    "imm_catprob_fixed_lag": MethodSpec(
        "imm_catprob_fixed_lag",
        "imm",
        "IMM catprob fixed-lag",
        fixed_lag=True,
    ),
    "imm_catprob_rts": MethodSpec(
        "imm_catprob_rts",
        "imm",
        "IMM catprob RTS upper bound",
        rts=True,
    ),
    "imm_tracklet_viterbi_fixed_lag": MethodSpec(
        "imm_tracklet_viterbi_fixed_lag",
        "tracklet",
        "IMM tracklet-Viterbi fixed-lag",
        association="tracklet-viterbi",
        fixed_lag=True,
        robust=True,
    ),
    "imm_learned_tracklet_viterbi_fixed_lag": MethodSpec(
        "imm_learned_tracklet_viterbi_fixed_lag",
        "learned_tracklet",
        "IMM learned tracklet-Viterbi fixed-lag",
        association="learned-tracklet-viterbi",
        fixed_lag=True,
        robust=True,
        needs_association_model=True,
        replay_tracker="imm",
    ),
    "hetero_imm_learned_tracklet_viterbi_fixed_lag": MethodSpec(
        "hetero_imm_learned_tracklet_viterbi_fixed_lag",
        "hetero_learned_tracklet",
        "LOFO heteroscedastic IMM learned tracklet-Viterbi fixed-lag",
        association="learned-tracklet-viterbi",
        fixed_lag=True,
        robust=True,
        needs_association_model=True,
        replay_tracker="imm",
    ),
    "hetero_cv": MethodSpec("hetero_cv", "hetero", "Heteroscedastic CV"),
    "hetero_cv_fixed_lag": MethodSpec(
        "hetero_cv_fixed_lag", "hetero", "Heteroscedastic CV fixed-lag", fixed_lag=True
    ),
    "hetero_cv_lofo_nis_fixed_lag": MethodSpec(
        "hetero_cv_lofo_nis_fixed_lag",
        "hetero",
        "LOFO NIS-calibrated heteroscedastic CV fixed-lag",
        fixed_lag=True,
        nis_calibrated=True,
    ),
    "hetero_imm_tracklet_viterbi_fixed_lag": MethodSpec(
        "hetero_imm_tracklet_viterbi_fixed_lag",
        "hetero_tracklet",
        "LOFO heteroscedastic IMM tracklet-Viterbi fixed-lag",
        association="tracklet-viterbi",
        fixed_lag=True,
        robust=True,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run all requested held-out folds and write leaderboard artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/leave_flight_out_sota"))
    parser.add_argument("--flights", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=[
            "cv_catprob",
            "cv_rf_anchored_nis_fixed_lag",
            "cv_rf_gated_nis_fixed_lag",
            "cv_track_bank_fixed_lag",
            "cv_stable_segments_fixed_lag",
            "cv_stable_segments_hybrid_fixed_lag",
            "cv_stable_segments_interpolated_fixed_lag",
            "imm_catprob",
            "imm_catprob_fixed_lag",
            "imm_catprob_rts",
            "imm_tracklet_viterbi_fixed_lag",
            "hetero_cv_lofo_nis_fixed_lag",
        ],
    )
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument(
        "--nis-calibration-method",
        choices=NIS_COVARIANCE_CALIBRATION_METHODS,
        default="mean",
    )
    parser.add_argument("--nis-calibration-quantile", type=float, default=0.95)
    parser.add_argument("--nis-calibration-min-samples", type=int, default=20)
    parser.add_argument("--nis-calibration-min-scale", type=float, default=0.25)
    parser.add_argument("--nis-calibration-max-scale", type=float, default=25.0)
    parser.add_argument("--nis-calibration-include-rejected", action="store_true")
    parser.add_argument(
        "--learned-association-teacher",
        choices=["oracle", "prediction-nis", "track-continuity", "none"],
        default="prediction-nis",
    )
    parser.add_argument(
        "--learned-association-disable-catprob-threshold",
        action="store_true",
        help="train learned radar association on all candidates instead of hard catProb-filtered candidates",
    )
    parser.add_argument("--tracklet-learned-unary-weight", type=float, default=1.0)
    parser.add_argument("--tracklet-hand-unary-weight", type=float, default=0.25)
    parser.add_argument("--enable-soft-catprob-retention", action="store_true")
    parser.add_argument("--enable-radar-velocity-update", action="store_true")
    parser.add_argument("--nis-covariance-calibration-json", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    flights = _selected_flight_names(args.dataset_root, args.flights)
    methods = [METHODS[name] for name in args.methods]
    if len(flights) < 2 and any(
        _uses_heteroscedastic_uncertainty(method) for method in methods
    ):
        raise ValueError("heteroscedastic leave-flight-out evaluation needs at least two flights")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    evaluations: dict[str, list[RunEvaluation]] = {method.name: [] for method in methods}

    for heldout in flights:
        train_flights = [flight for flight in flights if flight != heldout]
        fold_dir = args.output_dir / f"heldout_{_slug(heldout)}"
        model_path = fold_dir / "models" / "heteroscedastic_uncertainty.json"
        association_model_path = fold_dir / "models" / "radar_association_likelihood.json"
        if any(_uses_heteroscedastic_uncertainty(method) for method in methods):
            _train_uncertainty_model(args, train_flights, model_path)
        if any(method.needs_association_model for method in methods):
            _train_association_model(
                args,
                train_flights,
                association_model_path,
                excluded_flight=heldout,
            )
        for method in methods:
            nis_calibration_path: Path | None = None
            if method.nis_calibrated:
                nis_calibration_path = _prepare_nis_covariance_calibration(
                    args,
                    train_flights,
                    fold_dir,
                    model_path,
                )
            run_dir = fold_dir / method.name
            metrics_path = common.metrics_json_path(run_dir, heldout)
            if not (args.skip_existing and metrics_path.exists()):
                _run_method(args, method, heldout, run_dir, model_path, nis_calibration_path)
            evaluation = evaluate_run(
                dataset_root=args.dataset_root,
                flight=heldout,
                method=method,
                metrics_path=metrics_path,
                max_eval_time_delta_s=args.max_eval_time_delta_s,
                train_flights=train_flights,
            )
            fold_rows.append(evaluation.row)
            evaluations[method.name].append(evaluation)

    aggregate_rows = _aggregate_method_rows(methods, evaluations)
    _write_csv(args.output_dir / "fold_summary.csv", fold_rows)
    _write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "flights": flights,
                "methods": [method.__dict__ for method in methods],
                "fold_rows": fold_rows,
                "aggregate_rows": aggregate_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(fold_rows)} fold rows to {args.output_dir / 'fold_summary.csv'}")
    print(f"wrote {len(aggregate_rows)} aggregate rows to {args.output_dir / 'aggregate_summary.csv'}")
    return 0


def evaluate_run(
    *,
    dataset_root: Path,
    flight: str,
    method: MethodSpec,
    metrics_path: Path,
    max_eval_time_delta_s: float,
    train_flights: Sequence[str],
) -> RunEvaluation:
    """Load one run artifact set and recompute leakage-safe reporting metrics."""

    metrics = common.load_metrics(metrics_path)
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
    coverage = truth_coverage(truth_times, estimate_times, max_time_delta_s=max_eval_time_delta_s)
    smoother = metrics.get("smoother") or {}
    robust_update = metrics.get("robust_update") or {}
    row: dict[str, object] = {
        "heldout_flight": flight,
        "train_flights": ";".join(train_flights),
        "method": method.name,
        "label": method.label,
        "runner": method.runner,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", method.association)),
        "robust_update": _robust_name(robust_update),
        "smoother": smoother.get("method", "") if isinstance(smoother, dict) else "",
        "smoother_lag_s": smoother.get("lag_s", "") if isinstance(smoother, dict) else "",
        "posterior_records": int(metrics.get("posterior_records", len(estimates))),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", 0)),
        "accepted_measurements": int(metrics.get("accepted_measurements", 0)),
        "rejected_measurements": int(metrics.get("rejected_measurements", 0)),
        "nis_covariance_calibrated": bool(metrics.get("nis_covariance_calibrated", False)),
        "nis_covariance_calibration": metrics.get("nis_covariance_calibration", ""),
        "metrics_path": str(metrics_path),
    }
    row.update(_prefixed_summary("error_2d", summarize_scalar_errors(errors_2d)))
    row.update(_prefixed_summary("error_3d", summarize_scalar_errors(errors_3d)))
    row.update(coverage)
    row.update(_nis_summary(estimates))
    stability = selected_track_stability_metrics(selected_radar)
    if selected_radar is None:
        stability.pop("selected_radar_rows", None)
    row.update(stability)
    return RunEvaluation(
        row=row,
        errors_2d_m=errors_2d,
        errors_3d_m=errors_3d,
        covered_truth_rows=int(coverage["covered_truth_rows"]),
        truth_rows=int(coverage["truth_rows"]),
    )


def truth_coverage(
    truth_times_s: np.ndarray,
    estimate_times_s: np.ndarray,
    *,
    max_time_delta_s: float,
) -> dict[str, float | int]:
    """Return fraction of truth timestamps covered by a nearby estimate."""

    truth_times = np.asarray(truth_times_s, dtype=float).reshape(-1)
    estimate_times = np.asarray(estimate_times_s, dtype=float).reshape(-1)
    if truth_times.size == 0:
        return {"truth_rows": 0, "covered_truth_rows": 0, "truth_coverage_rate": float("nan")}
    if estimate_times.size == 0:
        return {"truth_rows": int(truth_times.size), "covered_truth_rows": 0, "truth_coverage_rate": 0.0}
    indices = nearest_time_indices(estimate_times, truth_times)
    dt_s = np.abs(estimate_times[indices] - truth_times)
    covered = int(np.count_nonzero(dt_s <= float(max_time_delta_s)))
    return {
        "truth_rows": int(truth_times.size),
        "covered_truth_rows": covered,
        "truth_coverage_rate": float(covered / truth_times.size),
    }


def summarize_scalar_errors(errors_m: np.ndarray) -> dict[str, float]:
    """Summarize scalar errors with tail metrics for SOTA-style tables."""

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


def _aggregate_method_rows(
    methods: Sequence[MethodSpec],
    evaluations: dict[str, Sequence[RunEvaluation]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        runs = list(evaluations.get(method.name, []))
        errors_2d = _concat([run.errors_2d_m for run in runs])
        errors_3d = _concat([run.errors_3d_m for run in runs])
        truth_rows = int(sum(run.truth_rows for run in runs))
        covered = int(sum(run.covered_truth_rows for run in runs))
        track_switch_count = int(sum(int(run.row.get("track_switch_count", 0)) for run in runs))
        finite_track_id_rows = int(sum(int(run.row.get("finite_track_id_rows", 0)) for run in runs))
        row: dict[str, object] = {
            "method": method.name,
            "label": method.label,
            "runner": method.runner,
            "nis_covariance_calibrated": bool(method.nis_calibrated),
            "folds": len(runs),
            "posterior_records": int(sum(int(run.row.get("posterior_records", 0)) for run in runs)),
            "selected_radar_rows": int(sum(int(run.row.get("selected_radar_rows", 0)) for run in runs)),
            "finite_track_id_rows": finite_track_id_rows,
            "unique_selected_track_ids_fold_sum": int(
                sum(int(run.row.get("unique_selected_track_ids", 0)) for run in runs)
            ),
            "track_switch_count": track_switch_count,
            "track_switch_rate": float(
                track_switch_count / max(finite_track_id_rows - len(runs), 1)
            ),
            "truth_rows": truth_rows,
            "covered_truth_rows": covered,
            "truth_coverage_rate": float(covered / truth_rows) if truth_rows else float("nan"),
            "dominant_track_fraction_mean": _mean_finite(
                [run.row.get("dominant_track_fraction") for run in runs]
            ),
            "selected_track_entropy_mean": _mean_finite(
                [run.row.get("selected_track_entropy") for run in runs]
            ),
        }
        row.update(_prefixed_summary("error_2d", summarize_scalar_errors(errors_2d)))
        row.update(_prefixed_summary("error_3d", summarize_scalar_errors(errors_3d)))
        rows.append(row)
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            float(item[1].get("error_3d_rmse_m", float("inf"))),
            -float(item[1].get("truth_coverage_rate", 0.0)),
        ),
    )
    for rank, (original_index, _) in enumerate(ranked, start=1):
        rows[original_index]["rank_rmse_3d"] = rank
    return rows


def _run_method(
    args: argparse.Namespace,
    method: MethodSpec,
    flight: str,
    run_dir: Path,
    model_path: Path,
    nis_calibration_path: Path | None = None,
) -> None:
    options: list[object] = ["--acceleration-std", args.acceleration_std]
    if method.robust:
        options.extend(common.robust_update_options(args))
    if method.fixed_lag:
        options.extend(common.smoother_options("fixed-lag", args.fixed_lag_s))
    if method.rts:
        options.extend(common.smoother_options("rts", args.fixed_lag_s))
    if method.runner == "baseline":
        options.extend(["--radar-catprob-threshold", args.candidate_threshold])
        common.run_baseline(
            dataset_root=args.dataset_root,
            flight=flight,
            output_dir=run_dir,
            association=method.association,
            extra_options=options,
        )
        return
    if method.runner == "tracklet":
        command = [
            sys.executable,
            "-m",
            "raft_uav.tracklet_viterbi_cli",
            "run-baseline",
            str(args.dataset_root),
            "--flight",
            flight,
            "--output-dir",
            str(run_dir),
            "--radar-association",
            method.association,
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            *[str(option) for option in options],
        ]
        _run(command)
        return
    if method.runner == "learned_tracklet":
        model_path = run_dir.parent / "models" / "radar_association_likelihood.json"
        command = [
            sys.executable,
            "-m",
            "raft_uav.tracklet_viterbi_cli",
            "run-baseline",
            str(args.dataset_root),
            "--flight",
            flight,
            "--output-dir",
            str(run_dir),
            "--radar-association",
            method.association,
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            method.replay_tracker,
            "--tracklet-association-model",
            str(model_path),
            "--tracklet-learned-unary-weight",
            str(args.tracklet_learned_unary_weight),
            "--tracklet-hand-unary-weight",
            str(args.tracklet_hand_unary_weight),
            *[str(option) for option in options],
        ]
        _run(command, env_overrides=_runtime_env_updates(args))
        return
    if method.runner == "hetero_learned_tracklet":
        association_model_path = run_dir.parent / "models" / "radar_association_likelihood.json"
        command = [
            sys.executable,
            "-m",
            "raft_uav.heteroscedastic_tracklet_viterbi_cli",
            "run-baseline",
            str(args.dataset_root),
            "--flight",
            flight,
            "--uncertainty-model",
            str(model_path),
            "--output-dir",
            str(run_dir),
            "--radar-association",
            method.association,
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            method.replay_tracker,
            "--tracklet-association-model",
            str(association_model_path),
            "--tracklet-learned-unary-weight",
            str(args.tracklet_learned_unary_weight),
            "--tracklet-hand-unary-weight",
            str(args.tracklet_hand_unary_weight),
            *[str(option) for option in options],
        ]
        _run(command, env_overrides=_runtime_env_updates(args))
        return
    if method.runner == "imm":
        command = [
            sys.executable,
            "-m",
            "raft_uav.imm_cli",
            str(args.dataset_root),
            "--flight",
            flight,
            "--output-dir",
            str(run_dir),
            "--tracker",
            "imm",
            "--radar-selection",
            "catprob",
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            *[str(option) for option in options],
        ]
        _run(command)
        return
    if method.runner == "hetero_tracklet":
        command = [
            sys.executable,
            "-m",
            "raft_uav.heteroscedastic_tracklet_viterbi_cli",
            "run-baseline",
            str(args.dataset_root),
            "--flight",
            flight,
            "--uncertainty-model",
            str(model_path),
            "--output-dir",
            str(run_dir),
            "--radar-association",
            method.association,
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            *[str(option) for option in options],
        ]
        _run(command)
        return
    if method.runner == "hetero":
        command = [
            sys.executable,
            "scripts/run_heteroscedastic_baseline.py",
            str(args.dataset_root),
            "--flight",
            flight,
            "--uncertainty-model",
            str(model_path),
            "--output-dir",
            str(run_dir),
            "--radar-selection",
            "catprob",
            "--radar-catprob-threshold",
            str(args.candidate_threshold),
            "--acceleration-std",
            str(args.acceleration_std),
        ]
        if method.fixed_lag:
            command.extend(["--smoother", "fixed-lag", "--smoother-lag-s", str(args.fixed_lag_s)])
        env_overrides: dict[str, str] | None = None
        if method.nis_calibrated:
            if nis_calibration_path is None:
                raise RuntimeError(f"{method.name} requires a NIS calibration JSON")
            env_overrides = {ENV_NIS_COVARIANCE_CALIBRATION_JSON: str(nis_calibration_path)}
        _run(command, env_overrides=env_overrides)
        return
    raise ValueError(f"unknown method runner {method.runner!r}")


def _uses_heteroscedastic_uncertainty(method: MethodSpec) -> bool:
    return method.runner in {"hetero", "hetero_tracklet", "hetero_learned_tracklet"}


def _train_uncertainty_model(args: argparse.Namespace, train_flights: Sequence[str], model_path: Path) -> None:
    if args.skip_existing and model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/train_heteroscedastic_uncertainty.py",
        str(args.dataset_root),
        "--output",
        str(model_path),
        "--ridge-lambda",
        str(args.ridge_lambda),
        "--max-time-delta-s",
        str(args.max_eval_time_delta_s),
    ]
    for flight in train_flights:
        command.extend(["--flight", flight])
    _run(command)


def _train_association_model(
    args: argparse.Namespace,
    train_flights: Sequence[str],
    model_path: Path,
    *,
    excluded_flight: str,
) -> None:
    if args.skip_existing and model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "raft_uav.train_radar_association_cli",
        str(args.dataset_root),
        "--output-model",
        str(model_path),
        "--teacher-association",
        args.learned_association_teacher,
        "--exclude-flight",
        excluded_flight,
    ]
    if args.learned_association_disable_catprob_threshold:
        command.append("--disable-radar-catprob-threshold")
    else:
        command.extend(["--radar-catprob-threshold", str(args.candidate_threshold)])
    for flight in train_flights:
        command.extend(["--flight", flight])
    _run(command)


def _prepare_nis_covariance_calibration(
    args: argparse.Namespace,
    train_flights: Sequence[str],
    fold_dir: Path,
    model_path: Path,
) -> Path:
    """Fit a held-out-safe NIS covariance calibration for a LOFO fold."""

    calibration_path = fold_dir / "models" / "nis_covariance_calibration.json"
    summary_path = fold_dir / "models" / "nis_covariance_calibration_summary.csv"
    if args.skip_existing and calibration_path.exists():
        return calibration_path

    diagnostics_root = fold_dir / "nis_calibration" / "heteroscedastic_uncalibrated"
    diagnostics_paths: list[Path] = []
    for flight in train_flights:
        diagnostics_path = diagnostics_root / flight / "diagnostics.csv"
        if not (args.skip_existing and diagnostics_path.exists()):
            _run_heteroscedastic_calibration_diagnostics(
                args=args,
                flight=flight,
                output_dir=diagnostics_root,
                model_path=model_path,
            )
        if not diagnostics_path.exists():
            raise FileNotFoundError(f"missing NIS diagnostics for {flight}: {diagnostics_path}")
        diagnostics_paths.append(diagnostics_path)

    payload = fit_nis_covariance_calibration_from_paths(
        diagnostics_paths,
        method=args.nis_calibration_method,
        quantile=args.nis_calibration_quantile,
        min_samples=args.nis_calibration_min_samples,
        min_scale=args.nis_calibration_min_scale,
        max_scale=args.nis_calibration_max_scale,
        accepted_only=not args.nis_calibration_include_rejected,
    )
    payload["train_flights"] = list(train_flights)
    payload["uncertainty_model"] = str(model_path)
    write_nis_covariance_calibration(payload, calibration_path)
    _write_nis_calibration_summary_csv(payload, summary_path)
    return calibration_path


def _run_heteroscedastic_calibration_diagnostics(
    *,
    args: argparse.Namespace,
    flight: str,
    output_dir: Path,
    model_path: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/run_heteroscedastic_baseline.py",
        str(args.dataset_root),
        "--flight",
        flight,
        "--uncertainty-model",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--radar-selection",
        "catprob",
        "--radar-catprob-threshold",
        str(args.candidate_threshold),
        "--acceleration-std",
        str(args.acceleration_std),
    ]
    _run(command)


def _write_nis_calibration_summary_csv(payload: Mapping[str, object], path: Path) -> None:
    groups = payload.get("groups")
    rows = []
    if isinstance(groups, Mapping):
        for key, group in groups.items():
            if isinstance(group, Mapping):
                rows.append({"group": key, **dict(group)})
    if rows:
        _write_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("group\n", encoding="utf-8")


def _runtime_env_updates(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.enable_soft_catprob_retention:
        env["RAFT_UAV_SOFT_CATPROB_RETENTION"] = "1"
    if args.enable_radar_velocity_update:
        env["RAFT_UAV_RADAR_UPDATE_USES_VELOCITY"] = "1"
    if args.nis_covariance_calibration_json is not None:
        env[ENV_NIS_COVARIANCE_CALIBRATION_JSON] = str(args.nis_covariance_calibration_json)
    return env


def _run(command: Sequence[str], *, env_overrides: Mapping[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    env = common.subprocess_env()
    env.pop(ENV_NIS_COVARIANCE_CALIBRATION_JSON, None)
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    subprocess.run(list(command), check=True, env=env)


def _selected_flight_names(dataset_root: Path, requested: Sequence[str] | None) -> list[str]:
    if requested:
        return [select_flight(dataset_root, name).name for name in requested]
    return [flight.name for flight in discover_flights(dataset_root) if flight.truth_txt is not None]


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _mean_finite(values: Sequence[object]) -> float:
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            parsed.append(number)
    return float(np.mean(parsed)) if parsed else float("nan")


def _robust_name(value: object) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("method") or ""
    return ""


def _nis_summary(estimates: pd.DataFrame) -> dict[str, object]:
    if "nis" not in estimates.columns:
        return {}
    out: dict[str, object] = {}
    source = estimates["source"] if "source" in estimates.columns else pd.Series(["all"] * len(estimates))
    for name, group in estimates.groupby(source):
        values = pd.to_numeric(group["nis"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            out[f"nis_{name}_count"] = int(values.size)
            out[f"nis_{name}_mean"] = float(np.mean(values))
            out[f"nis_{name}_p95"] = float(np.percentile(values, 95))
    return out


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(array, dtype=float).reshape(-1) for array in arrays if np.asarray(array).size]
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


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
