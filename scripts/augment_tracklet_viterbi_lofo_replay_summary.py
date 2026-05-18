"""Augment tracklet-Viterbi LOFO summaries with full Viterbi replay diagnostics.

The main tracklet runner writes both

``selected_radar.csv``
    Kalman-accepted replay updates only.

``viterbi_selected_radar.csv``
    All non-miss Viterbi-selected radar rows, including Kalman-rejected rows.

This script post-processes an existing ``run_tracklet_viterbi_lofo.py`` output
folder and adds exact fold/aggregate diagnostics for the full Viterbi-selected
path.  It is deliberately non-destructive by default and writes new summary
files next to the existing LOFO summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.evaluation.metrics import nearest_time_indices, position_errors_m  # noqa: E402
from raft_uav.io.aerpaw import normalize_truth, read_truth, select_flight  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/tracklet_viterbi_lofo"))
    parser.add_argument("--fold-summary", type=Path, default=None)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)

    fold_summary_path = args.fold_summary or args.summary_dir / "fold_summary.csv"
    if not fold_summary_path.exists():
        raise FileNotFoundError(f"missing fold summary: {fold_summary_path}")

    fold_rows = _read_csv_rows(fold_summary_path)
    augmented_rows: list[dict[str, object]] = []
    all_errors_2d: list[np.ndarray] = []
    all_errors_3d: list[np.ndarray] = []
    total_truth_rows = 0
    total_covered_truth_rows = 0
    total_viterbi_rows = 0
    total_rejected_rows = 0
    total_radar_frames = 0
    total_viterbi_frames = 0

    for row in fold_rows:
        flight = str(row.get("heldout_flight", ""))
        if not flight:
            raise ValueError("fold summary rows must contain heldout_flight")
        metrics_path = Path(str(row.get("metrics_path", "")))
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics file for {flight}: {metrics_path}")
        run_dir = metrics_path.parent
        replay_path = run_dir / "viterbi_selected_radar.csv"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        replay = _read_optional_csv(replay_path)
        truth = _load_truth(args.dataset_root, flight)

        augmented, errors_2d, errors_3d, coverage = _augment_fold_row(
            row,
            metrics=metrics,
            replay=replay,
            truth=truth,
            max_time_delta_s=args.max_eval_time_delta_s,
        )
        augmented_rows.append(augmented)
        all_errors_2d.append(errors_2d)
        all_errors_3d.append(errors_3d)
        total_truth_rows += int(coverage["truth_rows"])
        total_covered_truth_rows += int(coverage["covered_truth_rows"])
        total_viterbi_rows += int(augmented["viterbi_selected_radar_rows"])
        total_rejected_rows += int(augmented["viterbi_selected_radar_rejected_rows"])
        total_radar_frames += int(augmented["viterbi_selected_radar_radar_frame_count"])
        total_viterbi_frames += int(augmented["viterbi_selected_radar_frame_count"])

    aggregate_row = _aggregate_row(
        augmented_rows,
        errors_2d=_concat(all_errors_2d),
        errors_3d=_concat(all_errors_3d),
        truth_rows=total_truth_rows,
        covered_truth_rows=total_covered_truth_rows,
        viterbi_rows=total_viterbi_rows,
        rejected_rows=total_rejected_rows,
        radar_frames=total_radar_frames,
        viterbi_frames=total_viterbi_frames,
    )

    if args.in_place:
        fold_out = fold_summary_path
        aggregate_out = args.summary_dir / "aggregate_summary.csv"
    else:
        fold_out = args.summary_dir / "fold_summary_with_viterbi_replay.csv"
        aggregate_out = args.summary_dir / "aggregate_viterbi_replay_summary.csv"
    _write_csv(fold_out, augmented_rows)
    _write_csv(aggregate_out, [aggregate_row])
    print(f"wrote {len(augmented_rows)} augmented fold rows to {fold_out}")
    print(f"wrote Viterbi replay aggregate row to {aggregate_out}")
    return 0


def _augment_fold_row(
    row: dict[str, object],
    *,
    metrics: dict[str, object],
    replay: pd.DataFrame,
    truth: pd.DataFrame,
    max_time_delta_s: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict[str, float | int]]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    replay_times, replay_positions = _trajectory_arrays(replay)
    if replay_times.size and truth_times.size:
        errors_2d = position_errors_m(
            replay_times,
            replay_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=2,
        )
        errors_3d = position_errors_m(
            replay_times,
            replay_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=3,
        )
    else:
        errors_2d = np.array([], dtype=float)
        errors_3d = np.array([], dtype=float)
    coverage = _truth_coverage(truth_times, replay_times, max_time_delta_s=max_time_delta_s)

    diagnostics = metrics.get("viterbi_selected_radar_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    replay_rows = int(metrics.get("viterbi_selected_radar_rows", len(replay)))
    rejected_rows = int(
        metrics.get(
            "viterbi_selected_radar_rejected_rows",
            _rejected_replay_count(replay),
        )
    )
    radar_frame_count = _optional_int(diagnostics.get("radar_frame_count"), default=0)
    replay_frame_count = _optional_int(
        diagnostics.get("selected_radar_frame_count"),
        default=_frame_count(replay),
    )
    frame_coverage = _optional_float(diagnostics.get("radar_frame_coverage_rate"))
    if frame_coverage is None and radar_frame_count:
        frame_coverage = float(replay_frame_count / radar_frame_count)

    augmented = dict(row)
    augmented.update(
        {
            "viterbi_selected_radar_rows": replay_rows,
            "viterbi_selected_radar_rejected_rows": rejected_rows,
            "viterbi_selected_radar_rejection_rate": float(rejected_rows / replay_rows)
            if replay_rows
            else float("nan"),
            "viterbi_selected_radar_covered_truth_rows": int(coverage["covered_truth_rows"]),
            "viterbi_selected_radar_truth_coverage_rate": float(coverage["truth_coverage_rate"]),
            "viterbi_selected_radar_frame_count": replay_frame_count,
            "viterbi_selected_radar_radar_frame_count": radar_frame_count,
            "viterbi_selected_radar_frame_coverage_rate": _nan_if_none(frame_coverage),
        }
    )
    augmented.update(
        _prefixed_summary("viterbi_selected_radar_error_2d", _summarize_scalar_errors(errors_2d))
    )
    augmented.update(
        _prefixed_summary("viterbi_selected_radar_error_3d", _summarize_scalar_errors(errors_3d))
    )
    augmented.update(_replay_stat_columns(replay))
    return augmented, errors_2d, errors_3d, coverage


def _aggregate_row(
    fold_rows: list[dict[str, object]],
    *,
    errors_2d: np.ndarray,
    errors_3d: np.ndarray,
    truth_rows: int,
    covered_truth_rows: int,
    viterbi_rows: int,
    rejected_rows: int,
    radar_frames: int,
    viterbi_frames: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": "cv_tracklet_viterbi_fixed_lag",
        "label": "CV tracklet-Viterbi fixed-lag",
        "runner": "tracklet-viterbi",
        "folds": len(fold_rows),
        "viterbi_selected_radar_rows": viterbi_rows,
        "viterbi_selected_radar_rejected_rows": rejected_rows,
        "viterbi_selected_radar_rejection_rate": float(rejected_rows / viterbi_rows)
        if viterbi_rows
        else float("nan"),
        "truth_rows": truth_rows,
        "viterbi_selected_radar_covered_truth_rows": covered_truth_rows,
        "viterbi_selected_radar_truth_coverage_rate": float(covered_truth_rows / truth_rows)
        if truth_rows
        else float("nan"),
        "viterbi_selected_radar_frame_count": viterbi_frames,
        "viterbi_selected_radar_radar_frame_count": radar_frames,
        "viterbi_selected_radar_frame_coverage_rate": float(viterbi_frames / radar_frames)
        if radar_frames
        else float("nan"),
    }
    row.update(
        _prefixed_summary("viterbi_selected_radar_error_2d", _summarize_scalar_errors(errors_2d))
    )
    row.update(
        _prefixed_summary("viterbi_selected_radar_error_3d", _summarize_scalar_errors(errors_3d))
    )
    return row


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def _trajectory_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty or not {"time_s", "east_m", "north_m", "up_m"}.issubset(frame.columns):
        return np.array([], dtype=float), np.empty((0, 3), dtype=float)
    values = frame[["time_s", "east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    return values[:, 0], values[:, 1:4]


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


def _replay_stat_columns(replay: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for column in (
        "association_replay_nis",
        "association_replay_residual_norm_m",
        "association_replay_covariance_scale",
    ):
        stats = _numeric_column_stats(replay, column)
        short = column.removeprefix("association_replay_")
        for name, value in stats.items():
            out[f"viterbi_selected_radar_replay_{short}_{name}"] = value
    return out


def _numeric_column_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.empty or column not in frame.columns:
        return _empty_numeric_column_stats()
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_numeric_column_stats()
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _empty_numeric_column_stats() -> dict[str, float]:
    return {
        "count": 0.0,
        "mean": float("nan"),
        "p50": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }


def _rejected_replay_count(replay: pd.DataFrame) -> int:
    if replay.empty or "association_replay_accepted" not in replay.columns:
        return 0
    accepted = replay["association_replay_accepted"]
    if accepted.dtype == bool:
        return int((~accepted).sum())
    normalized = accepted.astype(str).str.lower().isin(("true", "1", "yes"))
    return int((~normalized).sum())


def _frame_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    column = "frame_index" if "frame_index" in frame.columns else "time_s"
    if column not in frame.columns:
        return int(len(frame))
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return int(np.unique(values).size)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(array, dtype=float).reshape(-1) for array in arrays if np.asarray(array).size]
    return np.concatenate(valid) if valid else np.array([], dtype=float)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: object, *, default: int) -> int:
    number = _optional_float(value)
    return default if number is None else int(number)


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
