#!/usr/bin/env python3
"""Build a SOTA-readiness report for RaFT-UAV tracking experiments.

The report combines truth-free tracking baselines with truth-based oracle
rows.  It is intended to answer whether the current bottleneck is association,
time alignment, filtering, or the available radar candidates themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.evaluation.metrics import position_errors_m  # noqa: E402
from raft_uav.evaluation.radar_oracle_diagnostics import (  # noqa: E402
    best_time_offset,
    nearest_candidate_oracle,
    summarize_oracle_selection,
    time_offset_sweep,
)
from raft_uav.io.aerpaw import (  # noqa: E402
    normalize_radar,
    normalize_truth,
    read_radar_tracks_json,
    read_truth,
    select_flight,
)

DEFAULT_FLIGHTS = ["Opt1", "Opt2", "Opt3"]
DEFAULT_METHODS = [
    "catprob",
    "prediction-nis",
    "rf-anchored-nis",
    "rf-gated-nis",
    "track-continuity",
    "geometry-score",
    "pda-mixture",
    "track-bank",
    "stable-segments",
    "stable-segments-interpolated",
    "tracklet-viterbi",
]
ORACLE_METHODS = ["oracle-nearest-candidate-offset0", "oracle-nearest-candidate-best-offset"]
PAPER_PRIMARY_METRIC = "mean_3d_error_m"
REPORT_COLUMNS = [
    "flight",
    "method",
    "row_type",
    "paper_primary_metric",
    "matched_count",
    "eval_sample_count",
    "coverage",
    "mean_3d_error_m",
    "std_3d_error_m",
    "max_3d_error_m",
    "p95_3d_error_m",
    "rmse_3d_error_m",
    "mean_2d_error_m",
    "std_2d_error_m",
    "max_2d_error_m",
    "p95_2d_error_m",
    "rmse_2d_error_m",
    "posterior_records",
    "selected_radar_rows",
    "radar_frame_count",
    "missed_radar_frame_count",
    "track_switch_count",
    "selected_cat_prob_mean",
    "association_anchor_nis_p95",
    "association_anchor_gate_rejected_count",
    "association_score_p95",
    "rejected_measurements",
    "reweighted_measurements",
    "applied_radar_time_offset_s",
    "applied_rf_time_offset_s",
    "best_oracle_time_offset_s",
    "metrics_path",
    "estimates_path",
    "selected_radar_path",
    "oracle_sweep_path",
]
PAPER_LEADERBOARD_COLUMNS = [
    "rank",
    "flight",
    "method",
    "row_type",
    "paper_primary_metric",
    "matched_count",
    "eval_sample_count",
    "coverage",
    "mean_3d_error_m",
    "std_3d_error_m",
    "max_3d_error_m",
    "p95_3d_error_m",
    "rmse_3d_error_m",
    "mean_2d_error_m",
    "std_2d_error_m",
    "max_2d_error_m",
    "p95_2d_error_m",
    "rmse_2d_error_m",
    "posterior_records",
    "selected_radar_rows",
    "radar_frame_count",
    "missed_radar_frame_count",
    "track_switch_count",
    "selected_cat_prob_mean",
    "association_anchor_nis_p95",
    "association_anchor_gate_rejected_count",
    "association_score_p95",
    "rejected_measurements",
    "reweighted_measurements",
    "applied_radar_time_offset_s",
    "applied_rf_time_offset_s",
    "best_oracle_time_offset_s",
    "metrics_path",
    "estimates_path",
    "selected_radar_path",
    "oracle_sweep_path",
]
ROW_TYPE_SORT = {"tracking": 0, "oracle": 1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sota_readiness"))
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--leaderboard-output", type=Path, default=None)
    parser.add_argument("--flights", nargs="*", default=DEFAULT_FLIGHTS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-oracles", action="store_true")
    parser.add_argument("--include-lofo", action="store_true")
    parser.add_argument(
        "--lofo-summary",
        type=Path,
        default=Path("outputs/lofo_time_offset/lofo_time_offset_summary.csv"),
    )
    parser.add_argument("--smoother", default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=20.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--offset-min-s", type=float, default=-10.0)
    parser.add_argument("--offset-max-s", type=float, default=10.0)
    parser.add_argument("--offset-step-s", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = args.summary_output or args.output_dir / "sota_readiness_report.csv"
    leaderboard_output = args.leaderboard_output or summary_output.with_name(
        f"{summary_output.stem}_leaderboard.csv"
    )

    rows: list[dict[str, object]] = []
    if not args.skip_tracking:
        for method in args.methods:
            for flight in args.flights:
                rows.append(run_or_collect_tracking(args, method, flight))
    if not args.skip_oracles:
        for flight in args.flights:
            rows.extend(oracle_rows(args, flight))
    if args.include_lofo:
        rows.extend(lofo_rows(args.lofo_summary))

    if not rows:
        raise RuntimeError("No report rows were produced")
    rows = sort_report_rows(rows)
    write_csv(summary_output, rows, preferred_columns=REPORT_COLUMNS)
    write_json(summary_output.with_suffix(".json"), rows)
    leaderboard_rows = build_leaderboard_rows(rows)
    write_csv(leaderboard_output, leaderboard_rows, preferred_columns=PAPER_LEADERBOARD_COLUMNS)
    write_json(leaderboard_output.with_suffix(".json"), leaderboard_rows)
    print(f"summary_csv={summary_output}")
    print(f"summary_json={summary_output.with_suffix('.json')}")
    print(f"leaderboard_csv={leaderboard_output}")
    print(f"leaderboard_json={leaderboard_output.with_suffix('.json')}")
    return 0


def run_or_collect_tracking(
    args: argparse.Namespace, method: str, flight: str
) -> dict[str, object]:
    run_dir = args.output_dir / "runs" / method
    metrics_path = run_dir / flight / "metrics.json"
    estimates_path = run_dir / flight / "estimates.csv"
    selected_path = run_dir / flight / "selected_radar.csv"
    if not (args.skip_existing and metrics_path.exists() and estimates_path.exists()):
        if method == "tracklet-viterbi":
            run_tracklet_viterbi(args, flight, run_dir)
        else:
            run_baseline(args, method, flight, run_dir)
    metrics = load_json(metrics_path)
    estimates = read_csv_or_empty(estimates_path)
    selected = read_csv_or_empty(selected_path)
    truth = load_truth(args.dataset_root, flight)
    row = base_row(method=method, flight=flight, row_type="tracking")
    row.update(extract_tracking_metadata(metrics, selected, estimates))
    row.update(error_summary_from_estimates(estimates, truth, args.max_eval_time_delta_s))
    row["metrics_path"] = str(metrics_path)
    row["estimates_path"] = str(estimates_path)
    row["selected_radar_path"] = str(selected_path)
    return row


def run_baseline(args: argparse.Namespace, method: str, flight: str, output_dir: Path) -> None:
    command: list[str] = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        method,
        "--radar-catprob-threshold",
        str(args.radar_catprob_threshold),
        "--smoother",
        args.smoother,
        "--smoother-lag-s",
        str(args.smoother_lag_s),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=subprocess_env())


def run_tracklet_viterbi(args: argparse.Namespace, flight: str, output_dir: Path) -> None:
    command: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tracklet_viterbi_baseline.py"),
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-catprob-threshold",
        str(args.radar_catprob_threshold),
        "--smoother",
        args.smoother,
        "--smoother-lag-s",
        str(args.smoother_lag_s),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=subprocess_env())


def oracle_rows(args: argparse.Namespace, flight: str) -> list[dict[str, object]]:
    truth = load_truth(args.dataset_root, flight)
    radar = load_radar(args.dataset_root, flight)
    frame_count = radar_frame_count(radar)
    offsets = offset_grid(args.offset_min_s, args.offset_max_s, args.offset_step_s)
    sweep = time_offset_sweep(
        radar,
        truth,
        offsets,
        max_time_delta_s=args.max_eval_time_delta_s,
    )
    oracle_dir = args.output_dir / "oracles" / flight
    oracle_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = oracle_dir / "nearest_candidate_time_offset_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    rows: list[dict[str, object]] = []
    for method, offset in (
        ("oracle-nearest-candidate-offset0", 0.0),
        ("oracle-nearest-candidate-best-offset", best_time_offset(sweep) or 0.0),
    ):
        selected = nearest_candidate_oracle(
            radar,
            truth,
            time_offset_s=float(offset),
            max_time_delta_s=args.max_eval_time_delta_s,
        )
        selected_path = oracle_dir / f"{method}.csv"
        selected.to_csv(selected_path, index=False)
        summary = summarize_oracle_selection(selected, frame_count=frame_count)
        summary["matched_count"] = summary.get("count", 0.0)
        summary["eval_sample_count"] = frame_count
        row = base_row(method=method, flight=flight, row_type="oracle")
        row.update(paper_error_columns(summary))
        row.update(selected_radar_diagnostics(selected, frame_count=frame_count))
        row["best_oracle_time_offset_s"] = round(float(offset), 6)
        row["metrics_path"] = ""
        row["estimates_path"] = ""
        row["selected_radar_path"] = str(selected_path)
        row["oracle_sweep_path"] = str(sweep_path)
        rows.append(row)
    return rows


def lofo_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        print(f"lofo_summary_missing={path}", flush=True)
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, item in frame.iterrows():
        flight = str(item.get("flight", ""))
        row = base_row(
            method="tracklet-viterbi-lofo-time-offset", flight=flight, row_type="tracking"
        )
        row["matched_count"] = integer(
            first_existing(item, ("matched_count", "count", "n_matched"))
        )
        row["eval_sample_count"] = integer(
            first_existing(item, ("eval_sample_count", "posterior_records", "n_estimates"))
        )
        row["coverage"] = rounded(item.get("coverage"))
        row["rmse_3d_error_m"] = rounded(item.get("rmse_3d_m"))
        row["p95_3d_error_m"] = rounded(item.get("p95_3d_m"))
        row["mean_3d_error_m"] = rounded(item.get("mae_3d_m"))
        row["std_3d_error_m"] = rounded(item.get("std_3d_m"))
        row["max_3d_error_m"] = rounded(item.get("max_3d_m"))
        row["applied_radar_time_offset_s"] = rounded(item.get("radar_offset_s"))
        row["applied_rf_time_offset_s"] = rounded(item.get("rf_offset_s"))
        row["metrics_path"] = str(item.get("metrics_json", ""))
        rows.append(row)
    return rows


def base_row(*, method: str, flight: str, row_type: str) -> dict[str, object]:
    return {
        "flight": flight,
        "method": method,
        "row_type": row_type,
        "paper_primary_metric": PAPER_PRIMARY_METRIC,
        "matched_count": "",
        "eval_sample_count": "",
        "coverage": "",
        "mean_3d_error_m": "",
        "std_3d_error_m": "",
        "rmse_3d_error_m": "",
        "p95_3d_error_m": "",
        "max_3d_error_m": "",
        "mean_2d_error_m": "",
        "std_2d_error_m": "",
        "rmse_2d_error_m": "",
        "p95_2d_error_m": "",
        "max_2d_error_m": "",
        "posterior_records": "",
        "selected_radar_rows": "",
        "radar_frame_count": "",
        "missed_radar_frame_count": "",
        "track_switch_count": "",
        "selected_cat_prob_mean": "",
        "association_anchor_nis_p95": "",
        "association_anchor_gate_rejected_count": "",
        "association_score_p95": "",
        "rejected_measurements": "",
        "reweighted_measurements": "",
        "applied_radar_time_offset_s": "",
        "applied_rf_time_offset_s": "",
        "best_oracle_time_offset_s": "",
        "metrics_path": "",
        "estimates_path": "",
        "selected_radar_path": "",
        "oracle_sweep_path": "",
    }


def extract_tracking_metadata(
    metrics: dict[str, Any], selected: pd.DataFrame, estimates: pd.DataFrame
) -> dict[str, object]:
    frame_count = int(metrics.get("radar_frames", 0) or metrics.get("radar_rows", 0) or 0)
    out: dict[str, object] = {
        "posterior_records": int(metrics.get("posterior_records", len(estimates))),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", len(selected))),
        "radar_frame_count": frame_count,
        "rejected_measurements": int(metrics.get("rejected_measurements", 0)),
        "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
    }
    out.update(selected_radar_diagnostics(selected, frame_count=frame_count))
    return out


def selected_radar_diagnostics(
    selected: pd.DataFrame, *, frame_count: int = 0
) -> dict[str, object]:
    if selected.empty:
        return {
            "selected_radar_rows": 0,
            "missed_radar_frame_count": frame_count if frame_count else "",
            "track_switch_count": 0,
            "selected_cat_prob_mean": "",
            "association_anchor_nis_p95": "",
            "association_anchor_gate_rejected_count": 0,
            "association_score_p95": "",
        }
    track_switches = 0
    if "track_id" in selected.columns:
        ids = pd.to_numeric(selected["track_id"], errors="coerce").dropna().to_numpy(dtype=float)
        track_switches = int(np.sum(ids[1:] != ids[:-1])) if ids.size > 1 else 0
    missed = max(0, frame_count - len(selected)) if frame_count else ""
    return {
        "selected_radar_rows": int(len(selected)),
        "missed_radar_frame_count": missed,
        "track_switch_count": track_switches,
        "selected_cat_prob_mean": mean_column(selected, "cat_prob_uav"),
        "association_anchor_nis_p95": percentile_column(selected, "association_anchor_nis", 95),
        "association_anchor_gate_rejected_count": sum_column(
            selected,
            "association_anchor_gate_rejected_count",
        ),
        "association_score_p95": percentile_column(selected, "association_score", 95),
    }


def error_summary_from_estimates(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    max_time_delta_s: float,
) -> dict[str, object]:
    if estimates.empty:
        return paper_error_columns({"matched_count": 0, "eval_sample_count": 0})
    times = estimates["time_s"].to_numpy(dtype=float)
    positions = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors_3d = position_errors_m(
        times,
        positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
        dimensions=3,
    )
    errors_2d = position_errors_m(
        times,
        positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
        dimensions=2,
    )
    summary = {
        "matched_count": int(errors_3d.size),
        "eval_sample_count": int(len(estimates)),
        "coverage": safe_divide(float(errors_3d.size), float(len(estimates))),
        **stats(errors_3d, "3d"),
        **stats(errors_2d, "2d"),
    }
    return paper_error_columns(summary)


def paper_error_columns(summary: dict[str, Any]) -> dict[str, object]:
    matched_count = summary.get("matched_count", summary.get("count"))
    keys = [
        "coverage",
        "mean_3d_error_m",
        "std_3d_error_m",
        "rmse_3d_error_m",
        "p95_3d_error_m",
        "max_3d_error_m",
        "mean_2d_error_m",
        "std_2d_error_m",
        "rmse_2d_error_m",
        "p95_2d_error_m",
        "max_2d_error_m",
    ]
    out: dict[str, object] = {
        "matched_count": integer(matched_count),
        "eval_sample_count": integer(summary.get("eval_sample_count")),
    }
    out.update({key: rounded(summary.get(key)) for key in keys})
    return out


def stats(errors: np.ndarray, suffix: str) -> dict[str, float]:
    values = np.asarray(errors, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"mean_{suffix}_error_m": float("nan"),
            f"std_{suffix}_error_m": float("nan"),
            f"rmse_{suffix}_error_m": float("nan"),
            f"p95_{suffix}_error_m": float("nan"),
            f"max_{suffix}_error_m": float("nan"),
        }
    return {
        f"mean_{suffix}_error_m": float(np.mean(values)),
        f"std_{suffix}_error_m": float(np.std(values)),
        f"rmse_{suffix}_error_m": float(np.sqrt(np.mean(values**2))),
        f"p95_{suffix}_error_m": float(np.percentile(values, 95)),
        f"max_{suffix}_error_m": float(np.max(values)),
    }


def build_leaderboard_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return a paper-metric leaderboard ranked by mean 3D error within each flight."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        entry = {
            column: row.get(column, "")
            for column in PAPER_LEADERBOARD_COLUMNS
            if column != "rank"
        }
        entry["paper_primary_metric"] = PAPER_PRIMARY_METRIC
        key = (str(entry.get("flight", "")), str(entry.get("row_type", "")))
        grouped.setdefault(key, []).append(entry)

    leaderboard: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item[0], ROW_TYPE_SORT.get(item[1], 99), item[1])):
        rank = 1
        ordered = sorted(
            grouped[key],
            key=lambda row: (
                metric_sort_value(row, PAPER_PRIMARY_METRIC),
                str(row.get("method", "")),
            ),
        )
        for entry in ordered:
            metric_value = finite_float(entry.get(PAPER_PRIMARY_METRIC))
            entry["rank"] = rank if metric_value is not None else ""
            if metric_value is not None:
                rank += 1
            leaderboard.append(entry)
    return leaderboard


def sort_report_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("flight", "")),
            ROW_TYPE_SORT.get(str(row.get("row_type", "")), 99),
            metric_sort_value(row, PAPER_PRIMARY_METRIC),
            str(row.get("method", "")),
        ),
    )


def metric_sort_value(row: dict[str, object], metric: str) -> float:
    value = finite_float(row.get(metric))
    return value if value is not None else float("inf")


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def first_existing(row: pd.Series, names: tuple[str, ...]) -> object:
    for name in names:
        value = row.get(name)
        if finite_float(value) is not None:
            return value
    return ""


def load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def load_radar(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
    radar = normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time)
    lower, upper = float(truth["time_s"].min()), float(truth["time_s"].max())
    return radar.loc[(radar["time_s"] >= lower) & (radar["time_s"] <= upper)].copy()


def radar_frame_count(radar: pd.DataFrame) -> int:
    if radar.empty:
        return 0
    column = "frame_index" if "frame_index" in radar.columns else "time_s"
    return int(radar[column].nunique())


def offset_grid(min_s: float, max_s: float, step_s: float) -> np.ndarray:
    if step_s <= 0.0:
        raise ValueError("offset-step-s must be positive")
    if max_s < min_s:
        raise ValueError("offset-max-s must be >= offset-min-s")
    count = int(np.floor((max_s - min_s) / step_s)) + 1
    values = min_s + np.arange(count, dtype=float) * step_s
    if values.size == 0 or values[-1] < max_s - 1e-9:
        values = np.append(values, max_s)
    return values


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    preferred_columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    if preferred_columns is not None:
        for column in preferred_columns:
            if any(column in row for row in rows) and column not in columns:
                columns.append(column)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def mean_column(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return rounded(float(np.mean(values))) if values.size else ""


def percentile_column(frame: pd.DataFrame, column: str, percentile: float) -> object:
    if column not in frame.columns:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return rounded(float(np.percentile(values, percentile))) if values.size else ""


def sum_column(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return int(np.sum(values)) if values.size else ""


def rounded(value: object) -> object:
    if value is None or value == "":
        return ""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(out):
        return ""
    try:
        return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))
    except (InvalidOperation, ValueError):
        return round(out, 3)


def integer(value: object) -> object:
    numeric = finite_float(value)
    return int(round(numeric)) if numeric is not None else ""


def safe_divide(numerator: float, denominator: float) -> object:
    if denominator <= 0.0:
        return ""
    return rounded(numerator / denominator)


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


if __name__ == "__main__":
    raise SystemExit(main())
