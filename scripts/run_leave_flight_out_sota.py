"""Leakage-safe leave-one-flight-out SOTA evaluation runner for RaFT-UAV."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402
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
    "hetero_cv": MethodSpec("hetero_cv", "hetero", "Heteroscedastic CV"),
    "hetero_cv_fixed_lag": MethodSpec(
        "hetero_cv_fixed_lag", "hetero", "Heteroscedastic CV fixed-lag", fixed_lag=True
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
            "hetero_cv_fixed_lag",
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
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    flights = _selected_flight_names(args.dataset_root, args.flights)
    methods = [METHODS[name] for name in args.methods]
    if len(flights) < 2 and any(method.runner == "hetero" for method in methods):
        raise ValueError("heteroscedastic leave-flight-out evaluation needs at least two flights")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    evaluations: dict[str, list[RunEvaluation]] = {method.name: [] for method in methods}

    for heldout in flights:
        train_flights = [flight for flight in flights if flight != heldout]
        fold_dir = args.output_dir / f"heldout_{_slug(heldout)}"
        model_path = fold_dir / "models" / "heteroscedastic_uncertainty.json"
        for method in methods:
            if method.runner == "hetero":
                _train_uncertainty_model(args, train_flights, model_path)
            run_dir = fold_dir / method.name
            metrics_path = common.metrics_json_path(run_dir, heldout)
            if not (args.skip_existing and metrics_path.exists()):
                _run_method(args, method, heldout, run_dir, model_path)
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
        "metrics_path": str(metrics_path),
    }
    row.update(_prefixed_summary("error_2d", summarize_scalar_errors(errors_2d)))
    row.update(_prefixed_summary("error_3d", summarize_scalar_errors(errors_3d)))
    row.update(coverage)
    row.update(_nis_summary(estimates))
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
        row: dict[str, object] = {
            "method": method.name,
            "label": method.label,
            "runner": method.runner,
            "folds": len(runs),
            "posterior_records": int(sum(int(run.row.get("posterior_records", 0)) for run in runs)),
            "selected_radar_rows": int(sum(int(run.row.get("selected_radar_rows", 0)) for run in runs)),
            "truth_rows": truth_rows,
            "covered_truth_rows": covered,
            "truth_coverage_rate": float(covered / truth_rows) if truth_rows else float("nan"),
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


def _run_method(args: argparse.Namespace, method: MethodSpec, flight: str, run_dir: Path, model_path: Path) -> None:
    options: list[object] = ["--acceleration-std", args.acceleration_std]
    if method.robust:
        options.extend(common.robust_update_options(args))
    if method.fixed_lag:
        options.extend(common.smoother_options("fixed-lag", args.fixed_lag_s))
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
            "--acceleration-std",
            str(args.acceleration_std),
        ]
        if method.robust:
            command.extend(str(option) for option in common.robust_update_options(args))
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
        _run(command)
        return
    raise ValueError(f"unknown method runner {method.runner!r}")


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


def _run(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(list(command), check=True, env=common.subprocess_env())


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
