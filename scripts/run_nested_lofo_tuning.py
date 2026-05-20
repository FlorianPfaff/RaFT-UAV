#!/usr/bin/env python3
"""Nested leave-one-flight-out hyperparameter tuning for RaFT-UAV methods."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.experiments.config import write_resolved_experiment_config  # noqa: E402
from raft_uav.io.aerpaw import discover_flights, select_flight  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nested_lofo_tuning"))
    parser.add_argument("--flights", nargs="*", default=None)
    parser.add_argument("--candidates-json", type=Path, default=None)
    parser.add_argument("--metric", default="position_error_3d.rmse_m")
    parser.add_argument("--aggregate", choices=["mean", "median", "max"], default="mean")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    flights = _selected_flights(args.dataset_root, args.flights)
    if len(flights) < 3:
        raise ValueError("nested LOFO tuning needs at least three flights")
    candidates = _load_candidates(args.candidates_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_experiment_config(
        args.output_dir / "resolved_experiment_config.json",
        argv=list(sys.argv if argv is None else argv),
        extra={"flights": flights, "candidates": candidates, "metric": args.metric},
    )

    training_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for holdout in flights:
        train_flights = [flight for flight in flights if flight != holdout]
        candidate_scores: list[dict[str, object]] = []
        for candidate in candidates:
            values: list[float] = []
            for train_flight in train_flights:
                metrics_path = _run_candidate(args, candidate, holdout, train_flight, split="train")
                metric_value = _read_metric(metrics_path, args.metric)
                values.append(metric_value)
                training_rows.append(
                    {
                        "holdout_flight": holdout,
                        "train_flight": train_flight,
                        "candidate": candidate["name"],
                        "metric": args.metric,
                        "metric_value": metric_value,
                        "metrics_json": str(metrics_path),
                    }
                )
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                candidate_scores.append(
                    {
                        "candidate": candidate["name"],
                        "aggregate_metric_value": _aggregate(finite, args.aggregate),
                        "finite_train_flights": int(finite.size),
                    }
                )
        if not candidate_scores:
            summary_rows.append({"holdout_flight": holdout, "status": "no_valid_candidate"})
            continue
        selected = sorted(candidate_scores, key=lambda row: (row["aggregate_metric_value"], row["candidate"]))[0]
        selected_candidate = next(candidate for candidate in candidates if candidate["name"] == selected["candidate"])
        holdout_metrics = _run_candidate(args, selected_candidate, holdout, holdout, split="holdout")
        summary_rows.append(
            {
                "holdout_flight": holdout,
                "train_flights": ";".join(train_flights),
                "selected_candidate": selected["candidate"],
                "selection_metric": args.metric,
                "selection_aggregate": args.aggregate,
                "training_metric_value": selected["aggregate_metric_value"],
                "holdout_metric_value": _read_metric(holdout_metrics, args.metric),
                "holdout_metrics_json": str(holdout_metrics),
                "status": "ok",
            }
        )
    _write_csv(args.output_dir / "nested_lofo_training_rows.csv", training_rows)
    _write_csv(args.output_dir / "nested_lofo_summary.csv", summary_rows)
    print(f"summary_csv={args.output_dir / 'nested_lofo_summary.csv'}")
    return 0


def _selected_flights(dataset_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [select_flight(dataset_root, item).name for item in requested]
    return [flight.name for flight in discover_flights(dataset_root) if flight.truth_txt is not None]


def _load_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [
            {"name": "cv_catprob", "runner": "baseline", "association": "catprob", "options": []},
            {
                "name": "cv_track_bank_fixed_lag",
                "runner": "baseline",
                "association": "track-bank",
                "options": ["--smoother", "fixed-lag", "--smoother-lag-s", "20"],
            },
            {
                "name": "tracklet_viterbi_range_covariance_imm",
                "runner": "tracklet",
                "association": "tracklet-viterbi",
                "options": [
                    "--tracklet-variant",
                    "range-covariance",
                    "--tracklet-replay-tracker",
                    "imm",
                    "--smoother",
                    "fixed-lag",
                    "--smoother-lag-s",
                    "20",
                ],
            },
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidates JSON must be a list")
    return [dict(item) for item in payload]


def _run_candidate(args: argparse.Namespace, candidate: dict[str, Any], holdout: str, flight: str, *, split: str) -> Path:
    run_dir = args.output_dir / holdout / split / str(candidate["name"])
    metrics_path = run_dir / flight / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return metrics_path
    module = "raft_uav.tracklet_viterbi_cli" if candidate.get("runner") == "tracklet" else "raft_uav.cli"
    command = [
        sys.executable,
        "-m",
        module,
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(run_dir),
    ]
    association = candidate.get("association")
    if association:
        command.extend(["--radar-association", str(association)])
    command.extend(str(option) for option in candidate.get("options", []))
    env = _subprocess_env()
    env.update({str(k): str(v) for k, v in dict(candidate.get("env", {}) or {}).items()})
    if args.dry_run:
        print(" ".join(command))
        return metrics_path
    subprocess.run(command, check=True, env=env)
    return metrics_path


def _read_metric(metrics_path: Path, dotted_key: str) -> float:
    if not metrics_path.exists():
        return float("nan")
    value: Any = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return float("nan")
        value = value[key]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _aggregate(values: np.ndarray, method: str) -> float:
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "max":
        return float(np.max(values))
    raise ValueError(method)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
