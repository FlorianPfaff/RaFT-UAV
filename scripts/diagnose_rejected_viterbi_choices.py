"""Classify Viterbi-selected radar choices against truth and Kalman replay.

The tracklet-Viterbi runner writes ``viterbi_selected_radar.csv`` with every
non-miss Viterbi radar choice and replay annotations such as
``association_replay_accepted``.  This script adds the missing truth-facing
classification:

``good_association_good_gate``
    The Viterbi-selected row is close to truth and Kalman replay accepted it.

``good_association_bad_gate``
    The Viterbi-selected row is close to truth but Kalman replay rejected it.

``bad_association_good_gate``
    The Viterbi-selected row is far from truth but Kalman replay accepted it.

``bad_association_bad_gate``
    The Viterbi-selected row is far from truth and Kalman replay rejected it.

The resulting per-row and summary CSVs separate association quality from replay
or gating behavior.
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

from raft_uav.evaluation.metrics import nearest_time_indices  # noqa: E402
from raft_uav.io.aerpaw import normalize_truth, read_truth, select_flight  # noqa: E402

CLASS_GOOD_ASSOCIATION_GOOD_GATE = "good_association_good_gate"
CLASS_GOOD_ASSOCIATION_BAD_GATE = "good_association_bad_gate"
CLASS_BAD_ASSOCIATION_GOOD_GATE = "bad_association_good_gate"
CLASS_BAD_ASSOCIATION_BAD_GATE = "bad_association_bad_gate"

CLASS_LABELS = (
    CLASS_GOOD_ASSOCIATION_GOOD_GATE,
    CLASS_GOOD_ASSOCIATION_BAD_GATE,
    CLASS_BAD_ASSOCIATION_GOOD_GATE,
    CLASS_BAD_ASSOCIATION_BAD_GATE,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/tracklet_viterbi_lofo"))
    parser.add_argument("--fold-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--good-association-threshold-m", type=float, default=75.0)
    parser.add_argument("--write-per-fold", action="store_true")
    args = parser.parse_args(argv)

    fold_summary_path = args.fold_summary or args.summary_dir / "fold_summary.csv"
    if not fold_summary_path.exists():
        raise FileNotFoundError(f"missing fold summary: {fold_summary_path}")
    output_dir = args.output_dir or args.summary_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_csv_rows(fold_summary_path)
    all_diagnostics: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, object]] = []
    for row in rows:
        flight = str(row.get("heldout_flight", ""))
        if not flight:
            raise ValueError("fold summary rows must contain heldout_flight")
        metrics_path = Path(str(row.get("metrics_path", "")))
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics file for {flight}: {metrics_path}")
        run_dir = metrics_path.parent
        replay_path = run_dir / "viterbi_selected_radar.csv"
        replay = _read_optional_csv(replay_path)
        truth = _load_truth(args.dataset_root, flight)
        diagnostics = classify_viterbi_choices(
            replay,
            truth,
            flight=flight,
            max_time_delta_s=args.max_eval_time_delta_s,
            good_association_threshold_m=args.good_association_threshold_m,
        )
        if args.write_per_fold:
            diagnostics.to_csv(run_dir / "viterbi_choice_truth_classification.csv", index=False)
        all_diagnostics.append(diagnostics)
        fold_summary_rows.append(
            _summary_row(
                diagnostics,
                scope="fold",
                flight=flight,
                metrics_path=metrics_path,
            )
        )

    combined = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    fold_summary_rows.append(_summary_row(combined, scope="aggregate"))

    combined_path = output_dir / "viterbi_choice_truth_classification.csv"
    summary_path = output_dir / "viterbi_choice_truth_classification_summary.csv"
    combined.to_csv(combined_path, index=False)
    _write_csv(summary_path, fold_summary_rows)
    print(f"wrote per-row Viterbi truth classifications to {combined_path}")
    print(f"wrote Viterbi truth classification summary to {summary_path}")
    return 0


def classify_viterbi_choices(
    replay: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    flight: str,
    max_time_delta_s: float,
    good_association_threshold_m: float,
) -> pd.DataFrame:
    """Annotate Viterbi-selected radar rows with truth/gating classes."""

    _require_columns(
        replay,
        required=("time_s", "east_m", "north_m", "up_m", "association_replay_accepted"),
        frame_name="viterbi_selected_radar",
    )
    _require_columns(
        truth,
        required=("time_s", "east_m", "north_m", "up_m"),
        frame_name="truth",
    )
    annotated = replay.copy()
    annotated.insert(0, "flight", flight)
    if replay.empty:
        _add_empty_truth_columns(annotated)
        return annotated

    replay_times = pd.to_numeric(replay["time_s"], errors="coerce").to_numpy(dtype=float)
    replay_positions = replay[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    finite_truth = np.isfinite(truth_times) & np.isfinite(truth_positions).all(axis=1)
    truth_times = truth_times[finite_truth]
    truth_positions = truth_positions[finite_truth]
    if truth_times.size == 0:
        _add_empty_truth_columns(annotated)
        annotated["truth_match_within_time_gate"] = False
        annotated["association_close_to_truth"] = False
        accepted = _accepted_array(replay)
        annotated["replay_truth_classification"] = _classification_labels(
            close_to_truth=np.zeros(len(replay), dtype=bool),
            replay_accepted=accepted,
        )
        return annotated

    nearest_indices = nearest_time_indices(replay_times, truth_times)
    nearest_times = truth_times[nearest_indices]
    nearest_positions = truth_positions[nearest_indices]
    time_delta_s = np.abs(replay_times - nearest_times)
    residual = replay_positions - nearest_positions
    errors_2d = np.linalg.norm(residual[:, :2], axis=1)
    errors_3d = np.linalg.norm(residual, axis=1)
    finite_replay = np.isfinite(replay_times) & np.isfinite(replay_positions).all(axis=1)
    within_time = finite_replay & (time_delta_s <= float(max_time_delta_s))
    close_to_truth = within_time & (errors_3d <= float(good_association_threshold_m))
    accepted = _accepted_array(replay)

    annotated["nearest_truth_time_s"] = nearest_times
    annotated["nearest_truth_dt_s"] = time_delta_s
    annotated["nearest_truth_east_m"] = nearest_positions[:, 0]
    annotated["nearest_truth_north_m"] = nearest_positions[:, 1]
    annotated["nearest_truth_up_m"] = nearest_positions[:, 2]
    annotated["truth_error_2d_m"] = errors_2d
    annotated["truth_error_3d_m"] = errors_3d
    annotated["truth_match_within_time_gate"] = within_time
    annotated["association_close_to_truth"] = close_to_truth
    annotated["replay_truth_classification"] = _classification_labels(
        close_to_truth=close_to_truth,
        replay_accepted=accepted,
    )
    return annotated


def _classification_labels(
    *,
    close_to_truth: np.ndarray,
    replay_accepted: np.ndarray,
) -> list[str]:
    labels: list[str] = []
    for close, accepted in zip(close_to_truth, replay_accepted, strict=True):
        if close and accepted:
            labels.append(CLASS_GOOD_ASSOCIATION_GOOD_GATE)
        elif close and not accepted:
            labels.append(CLASS_GOOD_ASSOCIATION_BAD_GATE)
        elif not close and accepted:
            labels.append(CLASS_BAD_ASSOCIATION_GOOD_GATE)
        else:
            labels.append(CLASS_BAD_ASSOCIATION_BAD_GATE)
    return labels


def _summary_row(
    diagnostics: pd.DataFrame,
    *,
    scope: str,
    flight: str = "",
    metrics_path: Path | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "heldout_flight": flight,
        "metrics_path": "" if metrics_path is None else str(metrics_path),
        "viterbi_selected_radar_rows": int(len(diagnostics)),
    }
    if diagnostics.empty:
        row.update(_empty_count_rates(prefix=""))
        row.update(_empty_error_summary(prefix="truth_error_3d"))
        row.update(_empty_error_summary(prefix="rejected_truth_error_3d"))
        return row

    labels = diagnostics["replay_truth_classification"].astype(str)
    accepted = _accepted_array(diagnostics)
    rejected = ~accepted
    close_to_truth = diagnostics["association_close_to_truth"].astype(bool).to_numpy()
    row["viterbi_selected_radar_accepted_rows"] = int(np.count_nonzero(accepted))
    row["viterbi_selected_radar_rejected_rows"] = int(np.count_nonzero(rejected))
    row["viterbi_selected_radar_rejection_rate"] = _safe_rate(
        row["viterbi_selected_radar_rejected_rows"],
        len(diagnostics),
    )
    row["good_association_rows"] = int(np.count_nonzero(close_to_truth))
    row["bad_association_rows"] = int(len(diagnostics) - row["good_association_rows"])
    row["good_association_rate"] = _safe_rate(row["good_association_rows"], len(diagnostics))
    for label in CLASS_LABELS:
        count = int(np.count_nonzero(labels == label))
        row[f"{label}_count"] = count
        row[f"{label}_rate"] = _safe_rate(count, len(diagnostics))
    rejected_count = int(np.count_nonzero(rejected))
    row["good_association_bad_gate_rate_among_rejected"] = _safe_rate(
        int(row[f"{CLASS_GOOD_ASSOCIATION_BAD_GATE}_count"]),
        rejected_count,
    )
    row["bad_association_good_gate_rate_among_accepted"] = _safe_rate(
        int(row[f"{CLASS_BAD_ASSOCIATION_GOOD_GATE}_count"]),
        int(np.count_nonzero(accepted)),
    )
    row.update(_error_summary(diagnostics["truth_error_3d_m"], prefix="truth_error_3d"))
    row.update(
        _error_summary(
            diagnostics.loc[rejected, "truth_error_3d_m"],
            prefix="rejected_truth_error_3d",
        )
    )
    return row


def _error_summary(values: pd.Series | np.ndarray, *, prefix: str) -> dict[str, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return _empty_error_summary(prefix=prefix)
    return {
        f"{prefix}_count": float(array.size),
        f"{prefix}_mean_m": float(np.mean(array)),
        f"{prefix}_p50_m": float(np.percentile(array, 50)),
        f"{prefix}_p95_m": float(np.percentile(array, 95)),
        f"{prefix}_max_m": float(np.max(array)),
    }


def _empty_error_summary(*, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_count": 0.0,
        f"{prefix}_mean_m": float("nan"),
        f"{prefix}_p50_m": float("nan"),
        f"{prefix}_p95_m": float("nan"),
        f"{prefix}_max_m": float("nan"),
    }


def _empty_count_rates(*, prefix: str) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        f"{prefix}viterbi_selected_radar_accepted_rows": 0,
        f"{prefix}viterbi_selected_radar_rejected_rows": 0,
        f"{prefix}viterbi_selected_radar_rejection_rate": float("nan"),
        f"{prefix}good_association_rows": 0,
        f"{prefix}bad_association_rows": 0,
        f"{prefix}good_association_rate": float("nan"),
        f"{prefix}good_association_bad_gate_rate_among_rejected": float("nan"),
        f"{prefix}bad_association_good_gate_rate_among_accepted": float("nan"),
    }
    for label in CLASS_LABELS:
        row[f"{prefix}{label}_count"] = 0
        row[f"{prefix}{label}_rate"] = float("nan")
    return row


def _accepted_array(frame: pd.DataFrame) -> np.ndarray:
    if "association_replay_accepted" not in frame.columns:
        raise ValueError("viterbi_selected_radar.csv must contain association_replay_accepted")
    values = frame["association_replay_accepted"]
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)


def _add_empty_truth_columns(frame: pd.DataFrame) -> None:
    for column in (
        "nearest_truth_time_s",
        "nearest_truth_dt_s",
        "nearest_truth_east_m",
        "nearest_truth_north_m",
        "nearest_truth_up_m",
        "truth_error_2d_m",
        "truth_error_3d_m",
    ):
        frame[column] = np.nan


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_columns(frame: pd.DataFrame, *, required: Sequence[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing and not frame.empty:
        missing_text = ", ".join(missing)
        raise ValueError(f"{frame_name} is missing required columns: {missing_text}")


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
