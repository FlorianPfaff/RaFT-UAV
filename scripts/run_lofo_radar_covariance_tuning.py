#!/usr/bin/env python3
"""Leave-one-flight-out tuning for range-angle radar covariance parameters."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RadarCovarianceCandidate:
    """One runtime radar-covariance setting."""

    candidate_id: str
    range_std_m: float
    azimuth_std_deg: float
    elevation_std_deg: float
    min_std_m: float
    max_std_m: float

    def environment(self) -> dict[str, str]:
        """Return RAFT_UAV_RADAR_* variables consumed by the covariance hook."""

        return {
            "RAFT_UAV_RADAR_COVARIANCE_MODE": "range-angle",
            "RAFT_UAV_RADAR_RANGE_STD_M": _format_float(self.range_std_m),
            "RAFT_UAV_RADAR_AZIMUTH_STD_DEG": _format_float(self.azimuth_std_deg),
            "RAFT_UAV_RADAR_ELEVATION_STD_DEG": _format_float(self.elevation_std_deg),
            "RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M": _format_float(self.min_std_m),
            "RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M": _format_float(self.max_std_m),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_radar_covariance"))
    parser.add_argument("--baseline-script", type=Path, default=Path("scripts/run_tracklet_viterbi_baseline.py"))
    parser.add_argument("--metric", default="position_error_3d.rmse_m")
    parser.add_argument("--aggregate", choices=["mean", "median", "max"], default="mean")
    parser.add_argument("--range-std-m", default="3,5,10,20")
    parser.add_argument("--azimuth-std-deg", default="1,2,3,4")
    parser.add_argument("--elevation-std-deg", default="1,2,3,4")
    parser.add_argument("--min-std-m", default="3")
    parser.add_argument("--max-std-m", default="150,250,400")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-arg",
        action="append",
        default=[],
        help="Extra token forwarded to run_tracklet_viterbi_baseline.py; repeat as needed.",
    )
    args = parser.parse_args(argv)

    flights = args.flight or ["Opt1", "Opt2", "Opt3"]
    if len(flights) < 2:
        raise ValueError("LOFO covariance tuning requires at least two flights")

    candidates = _candidate_grid(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for holdout in flights:
        train_flights = [flight for flight in flights if flight != holdout]
        train_rows = _training_rows(args, candidates, train_flights, holdout)
        all_rows.extend(train_rows)
        sweep = pd.DataFrame.from_records(train_rows)
        holdout_dir = args.output_dir / holdout
        holdout_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = holdout_dir / "training_covariance_sweep.csv"
        sweep.to_csv(sweep_path, index=False)

        selected = _select_candidate(sweep, metric_column="metric_value", aggregate=args.aggregate)
        if selected is None:
            summary_rows.append(_failed_summary_row(holdout, train_flights, sweep_path))
            continue

        candidate = next(c for c in candidates if c.candidate_id == selected["candidate_id"])
        metrics_path = _run_baseline(args, candidate, holdout, "holdout", holdout)
        holdout_metric = _read_metric(metrics_path, args.metric)
        selection_path = holdout_dir / "selected_covariance.json"
        selection_payload = {
            "holdout_flight": holdout,
            "training_flights": train_flights,
            "selection_metric": args.metric,
            "selection_aggregate": args.aggregate,
            "selected_candidate": asdict(candidate),
            "training_metric_value": selected["aggregate_metric_value"],
            "holdout_metric_value": holdout_metric,
            "training_sweep_csv": str(sweep_path),
            "holdout_metrics_json": str(metrics_path),
        }
        selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "holdout_flight": holdout,
                "training_flights": ",".join(train_flights),
                **asdict(candidate),
                "selection_metric": args.metric,
                "selection_aggregate": args.aggregate,
                "training_metric_value": selected["aggregate_metric_value"],
                "holdout_metric_value": holdout_metric,
                **_metric_summary(metrics_path),
                "training_sweep_csv": str(sweep_path),
                "selected_covariance_json": str(selection_path),
                "holdout_metrics_json": str(metrics_path),
            }
        )

    all_rows_path = args.output_dir / "lofo_radar_covariance_all_training_rows.csv"
    summary_path = args.output_dir / "lofo_radar_covariance_summary.csv"
    pd.DataFrame.from_records(all_rows).to_csv(all_rows_path, index=False)
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    print(f"all_training_rows_csv={all_rows_path}")
    return 0


def _candidate_grid(args: argparse.Namespace) -> list[RadarCovarianceCandidate]:
    fields = itertools.product(
        _parse_float_list(args.range_std_m),
        _parse_float_list(args.azimuth_std_deg),
        _parse_float_list(args.elevation_std_deg),
        _parse_float_list(args.min_std_m),
        _parse_float_list(args.max_std_m),
    )
    candidates = [
        RadarCovarianceCandidate(f"cov{idx:04d}", r, a, e, lo, hi)
        for idx, (r, a, e, lo, hi) in enumerate(fields)
        if hi >= lo
    ]
    if not candidates:
        raise ValueError("covariance grid is empty")
    return candidates


def _training_rows(
    args: argparse.Namespace,
    candidates: list[RadarCovarianceCandidate],
    train_flights: list[str],
    holdout: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for flight in train_flights:
            metrics_path = _run_baseline(args, candidate, flight, "train", holdout)
            rows.append(
                {
                    "holdout_flight": holdout,
                    "train_flight": flight,
                    **asdict(candidate),
                    "metric": args.metric,
                    "metric_value": _read_metric(metrics_path, args.metric),
                    "metrics_json": str(metrics_path),
                }
            )
    return rows


def _run_baseline(
    args: argparse.Namespace,
    candidate: RadarCovarianceCandidate,
    flight: str,
    split: str,
    holdout: str,
) -> Path:
    output_dir = args.output_dir / holdout / split / candidate.candidate_id
    metrics_path = output_dir / flight / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return metrics_path
    command = [
        sys.executable,
        str(args.baseline_script),
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        *args.baseline_arg,
    ]
    env = os.environ.copy()
    env.update(candidate.environment())
    if args.dry_run:
        print(" ".join(command))
        return metrics_path
    subprocess.run(command, check=True, env=env)
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing expected metrics file: {metrics_path}")
    return metrics_path


def _select_candidate(
    sweep: pd.DataFrame,
    *,
    metric_column: str,
    aggregate: str,
) -> dict[str, Any] | None:
    if sweep.empty or metric_column not in sweep.columns:
        return None
    rows = []
    for candidate_id, group in sweep.groupby("candidate_id", sort=False):
        values = pd.to_numeric(group[metric_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if aggregate == "mean":
            value = float(np.mean(values))
        elif aggregate == "median":
            value = float(np.median(values))
        elif aggregate == "max":
            value = float(np.max(values))
        else:  # pragma: no cover
            raise ValueError(f"unknown aggregate {aggregate!r}")
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "aggregate_metric_value": value,
                "finite_train_flights": int(values.size),
            }
        )
    if not rows:
        return None
    return sorted(rows, key=lambda row: (row["aggregate_metric_value"], row["candidate_id"]))[0]


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


def _metric_summary(metrics_path: Path) -> dict[str, float]:
    return {
        "holdout_rmse_3d_m": _read_metric(metrics_path, "position_error_3d.rmse_m"),
        "holdout_p95_3d_m": _read_metric(metrics_path, "position_error_3d.p95_m"),
        "holdout_mae_3d_m": _read_metric(metrics_path, "position_error_3d.mae_m"),
        "holdout_max_3d_m": _read_metric(metrics_path, "position_error_3d.max_m"),
        "selected_radar_rmse_3d_m": _read_metric(
            metrics_path, "selected_radar_position_error_3d.rmse_m"
        ),
        "selected_radar_p95_3d_m": _read_metric(
            metrics_path, "selected_radar_position_error_3d.p95_m"
        ),
    }


def _failed_summary_row(holdout: str, train_flights: list[str], sweep_path: Path) -> dict[str, Any]:
    return {
        "holdout_flight": holdout,
        "training_flights": ",".join(train_flights),
        "candidate_id": "",
        "training_metric_value": float("nan"),
        "holdout_metric_value": float("nan"),
        "training_sweep_csv": str(sweep_path),
    }


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty numeric list: {raw!r}")
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"all values must be finite and positive: {raw!r}")
    return values


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
