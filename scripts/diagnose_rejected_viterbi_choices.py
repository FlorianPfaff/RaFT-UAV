"""Classify Viterbi-selected radar choices by truth error and Kalman replay outcome.

The tracklet-Viterbi runner writes two radar artifacts:

``selected_radar.csv``
    Kalman-accepted replay updates only.

``viterbi_selected_radar.csv``
    Every non-miss Viterbi-selected radar row, including rows rejected by
    Kalman replay.

This script compares ``viterbi_selected_radar.csv`` against ground truth and
classifies each Viterbi choice into association/gating categories, for example
``good_association_bad_gate`` when a radar row is close to truth but rejected by
Kalman replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

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

CLASS_COLUMNS = (
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
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--good-threshold-m", type=float, default=75.0)
    parser.add_argument("--output-prefix", default="rejected_viterbi_choices")
    args = parser.parse_args(argv)

    if args.max_time_delta_s <= 0.0:
        raise ValueError("--max-time-delta-s must be positive")
    if args.good_threshold_m <= 0.0:
        raise ValueError("--good-threshold-m must be positive")

    fold_summary_path = _default_fold_summary(args.summary_dir, args.fold_summary)
    fold_rows = _read_csv_rows(fold_summary_path)
    diagnostic_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for row in fold_rows:
        flight_name = str(row.get("heldout_flight", "")).strip()
        if not flight_name:
            raise ValueError(f"fold row in {fold_summary_path} is missing heldout_flight")
        metrics_path = Path(str(row.get("metrics_path", "")).strip())
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics file for {flight_name}: {metrics_path}")
        replay_path = metrics_path.parent / "viterbi_selected_radar.csv"
        if not replay_path.exists():
            raise FileNotFoundError(
                f"missing replay artifact for {flight_name}: {replay_path}. "
                "Run scripts/run_tracklet_viterbi_baseline.py with the replay artifact first."
            )

        replay = pd.read_csv(replay_path)
        truth = _load_truth(args.dataset_root, flight_name)
        diagnostics = classify_viterbi_choices(
            replay,
            truth,
            flight_name=flight_name,
            replay_path=replay_path,
            max_time_delta_s=args.max_time_delta_s,
            good_threshold_m=args.good_threshold_m,
        )
        diagnostic_frames.append(diagnostics)
        summary_rows.append(_summarize_classifications(flight_name, diagnostics))

    all_diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame()
    )
    aggregate = _aggregate_summary(all_diagnostics, args)

    diagnostics_path = args.summary_dir / f"{args.output_prefix}_diagnostics.csv"
    summary_path = args.summary_dir / f"{args.output_prefix}_summary.csv"
    aggregate_path = args.summary_dir / f"{args.output_prefix}_summary.json"
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    all_diagnostics.to_csv(diagnostics_path, index=False)
    _write_csv(summary_path, summary_rows)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(f"wrote row diagnostics to {diagnostics_path}")
    print(f"wrote per-flight summary to {summary_path}")
    print(f"wrote aggregate summary to {aggregate_path}")
    return 0


def classify_viterbi_choices(
    replay: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    flight_name: str,
    replay_path: Path | None = None,
    max_time_delta_s: float = 2.0,
    good_threshold_m: float = 75.0,
) -> pd.DataFrame:
    """Return row-level truth/gate classification for Viterbi-selected radar rows."""

    if replay.empty:
        return _empty_classification_frame(replay)
    _require_columns(replay, {"time_s", "east_m", "north_m", "up_m"}, "replay")
    _require_columns(truth, {"time_s", "east_m", "north_m", "up_m"}, "truth")

    replay_times = pd.to_numeric(replay["time_s"], errors="coerce").to_numpy(dtype=float)
    replay_xyz = replay[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    truth_xyz = truth[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_truth = np.isfinite(truth_times) & np.isfinite(truth_xyz).all(axis=1)
    if not finite_truth.any():
        raise ValueError("truth contains no finite trajectory rows")
    truth_times = truth_times[finite_truth]
    truth_xyz = truth_xyz[finite_truth]

    finite_replay = np.isfinite(replay_times) & np.isfinite(replay_xyz).all(axis=1)
    nearest_indices = np.zeros(len(replay), dtype=int)
    nearest_indices[finite_replay] = nearest_time_indices(
        replay_times[finite_replay],
        truth_times,
    )
    matched_truth_times = truth_times[nearest_indices]
    matched_truth_xyz = truth_xyz[nearest_indices]
    dt_s = replay_times - matched_truth_times
    residual = replay_xyz - matched_truth_xyz
    error_2d_m = np.linalg.norm(residual[:, :2], axis=1)
    error_3d_m = np.linalg.norm(residual, axis=1)
    truth_match_valid = finite_replay & (np.abs(dt_s) <= float(max_time_delta_s))
    association_is_good = truth_match_valid & (error_3d_m <= float(good_threshold_m))
    replay_accepted = _accepted_series(replay)
    classifications = [
        _classification_label(good, accepted)
        for good, accepted in zip(association_is_good, replay_accepted, strict=True)
    ]

    diagnostics = replay.copy()
    diagnostics.insert(0, "heldout_flight", flight_name)
    diagnostics.insert(1, "viterbi_choice_index", np.arange(len(diagnostics), dtype=int))
    diagnostics["viterbi_replay_path"] = "" if replay_path is None else str(replay_path)
    diagnostics["nearest_truth_time_s"] = matched_truth_times
    diagnostics["truth_time_delta_s"] = dt_s
    diagnostics["truth_match_valid"] = truth_match_valid
    diagnostics["truth_east_m"] = matched_truth_xyz[:, 0]
    diagnostics["truth_north_m"] = matched_truth_xyz[:, 1]
    diagnostics["truth_up_m"] = matched_truth_xyz[:, 2]
    diagnostics["viterbi_error_2d_m"] = error_2d_m
    diagnostics["viterbi_error_3d_m"] = error_3d_m
    diagnostics["good_association_threshold_m"] = float(good_threshold_m)
    diagnostics["association_is_good"] = association_is_good
    diagnostics["replay_accepted"] = replay_accepted
    diagnostics["rejected_choice_classification"] = classifications
    return diagnostics


def _classification_label(good_association: bool, replay_accepted: bool) -> str:
    if good_association and replay_accepted:
        return CLASS_GOOD_ASSOCIATION_GOOD_GATE
    if good_association and not replay_accepted:
        return CLASS_GOOD_ASSOCIATION_BAD_GATE
    if not good_association and replay_accepted:
        return CLASS_BAD_ASSOCIATION_GOOD_GATE
    return CLASS_BAD_ASSOCIATION_BAD_GATE


def _accepted_series(replay: pd.DataFrame) -> np.ndarray:
    if "association_replay_accepted" in replay.columns:
        return replay["association_replay_accepted"].map(_to_bool).to_numpy(dtype=bool)
    if "association_replay_update_action" in replay.columns:
        actions = replay["association_replay_update_action"].astype(str)
        rejected_actions = {"missed_detection", "rejected", "gated"}
        return ~actions.str.lower().isin(rejected_actions).to_numpy(dtype=bool)
    return np.zeros(len(replay), dtype=bool)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "accepted"}


def _summarize_classifications(flight_name: str, diagnostics: pd.DataFrame) -> dict[str, object]:
    counts = Counter(diagnostics["rejected_choice_classification"].astype(str))
    accepted = int(diagnostics["replay_accepted"].sum()) if len(diagnostics) else 0
    good = int(diagnostics["association_is_good"].sum()) if len(diagnostics) else 0
    row: dict[str, object] = {
        "heldout_flight": flight_name,
        "viterbi_selected_radar_rows": int(len(diagnostics)),
        "viterbi_selected_radar_accepted_rows": accepted,
        "viterbi_selected_radar_rejected_rows": int(len(diagnostics) - accepted),
        "viterbi_selected_radar_good_association_rows": good,
        "viterbi_selected_radar_bad_association_rows": int(len(diagnostics) - good),
        "viterbi_selected_radar_truth_match_valid_rows": int(
            diagnostics["truth_match_valid"].sum()
        )
        if len(diagnostics)
        else 0,
    }
    for label in CLASS_COLUMNS:
        row[label] = int(counts.get(label, 0))
        row[f"{label}_rate"] = (
            float(counts.get(label, 0) / len(diagnostics))
            if len(diagnostics)
            else float("nan")
        )
    row.update(
        _prefixed_summary(
            "viterbi_error_2d",
            _summarize_errors(diagnostics["viterbi_error_2d_m"]),
        )
    )
    row.update(
        _prefixed_summary(
            "viterbi_error_3d",
            _summarize_errors(diagnostics["viterbi_error_3d_m"]),
        )
    )
    return row


def _aggregate_summary(diagnostics: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    counts = (
        Counter(diagnostics["rejected_choice_classification"].astype(str))
        if len(diagnostics)
        else Counter()
    )
    accepted = int(diagnostics["replay_accepted"].sum()) if len(diagnostics) else 0
    good = int(diagnostics["association_is_good"].sum()) if len(diagnostics) else 0
    summary: dict[str, object] = {
        "summary_dir": str(args.summary_dir),
        "max_time_delta_s": args.max_time_delta_s,
        "good_threshold_m": args.good_threshold_m,
        "viterbi_selected_radar_rows": int(len(diagnostics)),
        "viterbi_selected_radar_accepted_rows": accepted,
        "viterbi_selected_radar_rejected_rows": int(len(diagnostics) - accepted),
        "viterbi_selected_radar_rejection_rate": (
            float((len(diagnostics) - accepted) / len(diagnostics))
            if len(diagnostics)
            else float("nan")
        ),
        "viterbi_selected_radar_good_association_rows": good,
        "viterbi_selected_radar_bad_association_rows": int(len(diagnostics) - good),
        "viterbi_selected_radar_good_association_rate": float(good / len(diagnostics))
        if len(diagnostics)
        else float("nan"),
    }
    for label in CLASS_COLUMNS:
        summary[label] = int(counts.get(label, 0))
        summary[f"{label}_rate"] = (
            float(counts.get(label, 0) / len(diagnostics))
            if len(diagnostics)
            else float("nan")
        )
    summary.update(
        _prefixed_summary(
            "viterbi_error_2d",
            _summarize_errors(diagnostics.get("viterbi_error_2d_m", [])),
        )
    )
    summary.update(
        _prefixed_summary(
            "viterbi_error_3d",
            _summarize_errors(diagnostics.get("viterbi_error_3d_m", [])),
        )
    )
    return summary


def _default_fold_summary(summary_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = (
        summary_dir / "fold_summary_with_viterbi_replay.csv",
        summary_dir / "fold_summary.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_truth(dataset_root: Path, flight_name: str) -> pd.DataFrame:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, _, _ = normalize_truth(read_truth(flight.truth_txt))
    return truth


def _summarize_errors(values: Iterable[float] | pd.Series) -> dict[str, float]:
    errors = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
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


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
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


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _empty_classification_frame(replay: pd.DataFrame) -> pd.DataFrame:
    out = replay.copy()
    for column in (
        "heldout_flight",
        "viterbi_choice_index",
        "viterbi_replay_path",
        "nearest_truth_time_s",
        "truth_time_delta_s",
        "truth_match_valid",
        "truth_east_m",
        "truth_north_m",
        "truth_up_m",
        "viterbi_error_2d_m",
        "viterbi_error_3d_m",
        "good_association_threshold_m",
        "association_is_good",
        "replay_accepted",
        "rejected_choice_classification",
    ):
        out[column] = []
    return out


if __name__ == "__main__":
    raise SystemExit(main())
