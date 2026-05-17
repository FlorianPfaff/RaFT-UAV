"""Leave-one-flight-out aggregate runner for tracklet-Viterbi association.

This script gives the offline truth-free tracklet-Viterbi method the same
fold/aggregate reporting shape as ``run_leave_flight_out_sota.py`` without
requiring it to be exposed as an online ``run-baseline`` association mode.
"""

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
class RunEvaluation:
    """Per-fold errors and summary row for aggregate reporting."""

    row: dict[str, object]
    errors_2d_m: np.ndarray
    errors_3d_m: np.ndarray
    covered_truth_rows: int
    truth_rows: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run tracklet-Viterbi on requested held-out flights and aggregate metrics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet_viterbi_lofo"))
    parser.add_argument("--flights", nargs="*", default=None)
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
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
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=float, default=0.0)
    parser.add_argument("--robust-update", choices=["none", "nis-inflate"], default="nis-inflate")
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    flights = _selected_flight_names(args.dataset_root, args.flights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations: list[RunEvaluation] = []
    for heldout in flights:
        run_dir = args.output_dir / "cv_tracklet_viterbi_fixed_lag"
        metrics_path = common.metrics_json_path(run_dir, heldout)
        if not (args.skip_existing and metrics_path.exists()):
            _run_tracklet_viterbi(args, heldout, run_dir)
        evaluations.append(
            _evaluate_run(
                dataset_root=args.dataset_root,
                flight=heldout,
                metrics_path=metrics_path,
                max_eval_time_delta_s=args.max_eval_time_delta_s,
                train_flights=[flight for flight in flights if flight != heldout],
            )
        )

    fold_rows = [evaluation.row for evaluation in evaluations]
    aggregate_rows = [_aggregate_row(evaluations)]
    _write_csv(args.output_dir / "fold_summary.csv", fold_rows)
    _write_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "flights": flights,
                "method": "cv_tracklet_viterbi_fixed_lag",
                "fold_rows": fold_rows,
                "aggregate_rows": aggregate_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(fold_rows)} fold rows to {args.output_dir / 'fold_summary.csv'}")
    print(f"wrote aggregate row to {args.output_dir / 'aggregate_summary.csv'}")
    return 0


def _run_tracklet_viterbi(args: argparse.Namespace, flight: str, run_dir: Path) -> None:
    command: list[object] = [
        sys.executable,
        "scripts/run_tracklet_viterbi_baseline.py",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        run_dir,
        "--acceleration-std",
        args.acceleration_std,
        "--radar-catprob-threshold",
        args.candidate_threshold,
        "--max-candidates-per-frame",
        args.max_candidates_per_frame,
        "--missed-detection-cost",
        args.missed_detection_cost,
        "--track-switch-cost",
        args.track_switch_cost,
        "--catprob-weight",
        args.catprob_weight,
        "--anchor-nis-weight",
        args.anchor_nis_weight,
        "--transition-nis-weight",
        args.transition_nis_weight,
        "--velocity-nis-weight",
        args.velocity_nis_weight,
        "--max-speed-mps",
        args.max_speed_mps,
        "--range-gate-m",
        args.range_gate_m,
        "--smoother",
        "fixed-lag",
        "--smoother-lag-s",
        args.fixed_lag_s,
        "--max-eval-time-delta-s",
        args.max_eval_time_delta_s,
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
        "--robust-update",
        args.robust_update,
        "--rf-inflation-alpha",
        args.rf_inflation_alpha,
        "--radar-inflation-alpha",
        args.radar_inflation_alpha,
    ]
    if args.disable_rf_anchor:
        command.append("--disable-rf-anchor")
    _run(command)


def _evaluate_run(
    *,
    dataset_root: Path,
    flight: str,
    metrics_path: Path,
    max_eval_time_delta_s: float,
    train_flights: Sequence[str],
) -> RunEvaluation:
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
    coverage = _truth_coverage(truth_times, estimate_times, max_time_delta_s=max_eval_time_delta_s)
    smoother = metrics.get("smoother") or {}
    row: dict[str, object] = {
        "heldout_flight": flight,
        "train_flights": ";".join(train_flights),
        "method": "cv_tracklet_viterbi_fixed_lag",
        "label": "CV tracklet-Viterbi fixed-lag",
        "runner": "tracklet-viterbi",
        "radar_association": metrics.get("radar_association", "tracklet-viterbi"),
        "robust_update": metrics.get("robust_update", {}).get("method", "")
        if isinstance(metrics.get("robust_update"), dict)
        else metrics.get("robust_update", ""),
        "smoother": smoother.get("method", "") if isinstance(smoother, dict) else "",
        "smoother_lag_s": smoother.get("lag_s", "") if isinstance(smoother, dict) else "",
        "posterior_records": int(metrics.get("posterior_records", len(estimates))),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", 0)),
        "accepted_measurements": int(metrics.get("accepted_measurements", 0)),
        "rejected_measurements": int(metrics.get("rejected_measurements", 0)),
        "metrics_path": str(metrics_path),
    }
    row.update(_prefixed_summary("error_2d", _summarize_scalar_errors(errors_2d)))
    row.update(_prefixed_summary("error_3d", _summarize_scalar_errors(errors_3d)))
    row.update(coverage)
    row.update(_nis_summary(estimates))
    return RunEvaluation(
        row=row,
        errors_2d_m=errors_2d,
        errors_3d_m=errors_3d,
        covered_truth_rows=int(coverage["covered_truth_rows"]),
        truth_rows=int(coverage["truth_rows"]),
    )


def _aggregate_row(evaluations: Sequence[RunEvaluation]) -> dict[str, object]:
    errors_2d = _concat([evaluation.errors_2d_m for evaluation in evaluations])
    errors_3d = _concat([evaluation.errors_3d_m for evaluation in evaluations])
    truth_rows = int(sum(evaluation.truth_rows for evaluation in evaluations))
    covered = int(sum(evaluation.covered_truth_rows for evaluation in evaluations))
    row: dict[str, object] = {
        "method": "cv_tracklet_viterbi_fixed_lag",
        "label": "CV tracklet-Viterbi fixed-lag",
        "runner": "tracklet-viterbi",
        "folds": len(evaluations),
        "posterior_records": int(sum(int(e.row.get("posterior_records", 0)) for e in evaluations)),
        "selected_radar_rows": int(sum(int(e.row.get("selected_radar_rows", 0)) for e in evaluations)),
        "truth_rows": truth_rows,
        "covered_truth_rows": covered,
        "truth_coverage_rate": float(covered / truth_rows) if truth_rows else float("nan"),
    }
    row.update(_prefixed_summary("error_2d", _summarize_scalar_errors(errors_2d)))
    row.update(_prefixed_summary("error_3d", _summarize_scalar_errors(errors_3d)))
    row["rank_rmse_3d"] = 1
    return row


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
        return {"truth_rows": int(truth_times.size), "covered_truth_rows": 0, "truth_coverage_rate": 0.0}
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
    return [flight.name for flight in discover_flights(dataset_root) if flight.truth_txt is not None]


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


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


def _run(command: Sequence[object]) -> None:
    command_text = [str(item) for item in command]
    print(" ".join(command_text), flush=True)
    subprocess.run(command_text, check=True, env=common.subprocess_env())


if __name__ == "__main__":
    raise SystemExit(main())
