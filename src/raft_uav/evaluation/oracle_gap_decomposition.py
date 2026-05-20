"""Oracle-gap and confidence diagnostics for RaFT-UAV tracking runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OracleGapConfig:
    """Configuration for oracle-gap decomposition."""

    plausible_candidate_gate_m: float = 50.0
    truth_time_gate_s: float = 1.0
    estimate_time_gate_s: float = 2.0
    drift_error_gate_m: float = 150.0

    def __post_init__(self) -> None:
        for name in (
            "plausible_candidate_gate_m",
            "truth_time_gate_s",
            "estimate_time_gate_s",
            "drift_error_gate_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


def decompose_radar_oracle_gap(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    config: OracleGapConfig | None = None,
) -> pd.DataFrame:
    """Return one diagnostic row per radar frame.

    Ground truth is used only for post-run diagnostics. A row category separates
    candidate availability, association, gating/replay acceptance, and posterior
    drift after a seemingly correct association.
    """

    cfg = config or OracleGapConfig()
    if radar.empty:
        return pd.DataFrame(columns=_ORACLE_GAP_COLUMNS)
    if truth.empty:
        raise ValueError("truth must contain at least one row")

    selected_by_key = _selected_rows_by_key(selected_radar)
    estimate_times, estimate_positions = _time_position_arrays(estimates)
    rows: list[dict[str, object]] = []
    for frame in _radar_frame_groups(radar):
        key_type, key_value = _frame_key(frame)
        time_s = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
        truth_position = _nearest_position(
            truth,
            time_s=time_s,
            max_delta_s=float(cfg.truth_time_gate_s),
        )
        row: dict[str, object] = {
            "frame_key_type": key_type,
            "frame_key": key_value,
            "time_s": time_s,
            "candidate_count": int(len(frame)),
            "truth_available": truth_position is not None,
            "nearest_candidate_error_m": float("nan"),
            "nearest_candidate_track_id": "",
            "has_plausible_candidate": False,
            "selected_present": False,
            "selected_track_id": "",
            "selected_error_m": float("nan"),
            "selected_is_plausible": False,
            "selected_replay_accepted": "",
            "estimate_error_m": float("nan"),
            "category": "no_truth",
        }
        if truth_position is None:
            rows.append(row)
            continue

        positions = _positions(frame)
        if positions.size == 0:
            row["category"] = "empty_radar_frame"
            rows.append(row)
            continue
        candidate_errors = np.linalg.norm(positions - truth_position.reshape(1, 3), axis=1)
        nearest_index = int(np.argmin(candidate_errors))
        nearest_error = float(candidate_errors[nearest_index])
        nearest_row = frame.iloc[nearest_index]
        row.update(
            {
                "nearest_candidate_error_m": nearest_error,
                "nearest_candidate_track_id": _optional_track_id(nearest_row.get("track_id")),
                "has_plausible_candidate": bool(nearest_error <= cfg.plausible_candidate_gate_m),
            }
        )

        selected = selected_by_key.get((key_type, key_value))
        if selected is not None:
            selected_position = _row_position(selected)
            selected_error = (
                float(np.linalg.norm(selected_position - truth_position))
                if selected_position is not None
                else float("nan")
            )
            row.update(
                {
                    "selected_present": True,
                    "selected_track_id": _optional_track_id(selected.get("track_id")),
                    "selected_error_m": selected_error,
                    "selected_is_plausible": bool(
                        math.isfinite(selected_error)
                        and selected_error <= cfg.plausible_candidate_gate_m
                    ),
                    "selected_replay_accepted": _optional_bool(
                        selected.get("association_replay_accepted")
                    ),
                }
            )

        row["estimate_error_m"] = _nearest_estimate_error(
            estimate_times,
            estimate_positions,
            time_s=time_s,
            truth_position=truth_position,
            max_delta_s=float(cfg.estimate_time_gate_s),
        )
        row["category"] = _gap_category(row, cfg)
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=_ORACLE_GAP_COLUMNS)


def summarize_oracle_gap(frame_rows: pd.DataFrame) -> dict[str, object]:
    """Summarize oracle-gap rows into report-friendly counts and rates."""

    if frame_rows.empty:
        return {
            "radar_frame_count": 0,
            "truth_matched_frame_count": 0,
            "plausible_candidate_frame_count": 0,
        }
    categories = frame_rows["category"].value_counts(dropna=False).to_dict()
    truth_rows = frame_rows.loc[frame_rows["truth_available"].astype(bool)]
    plausible_rows = truth_rows.loc[truth_rows["has_plausible_candidate"].astype(bool)]
    correct_rows = plausible_rows.loc[plausible_rows["selected_is_plausible"].astype(bool)]
    out: dict[str, object] = {
        "radar_frame_count": int(len(frame_rows)),
        "truth_matched_frame_count": int(len(truth_rows)),
        "plausible_candidate_frame_count": int(len(plausible_rows)),
        "selected_plausible_frame_count": int(len(correct_rows)),
        "candidate_availability_rate": _safe_rate(len(plausible_rows), len(truth_rows)),
        "association_recall_given_candidate_rate": _safe_rate(len(correct_rows), len(plausible_rows)),
    }
    for category, count in sorted(categories.items()):
        out[f"category_{category}_count"] = int(count)
        out[f"category_{category}_rate"] = _safe_rate(count, len(frame_rows))
    for column, prefix in (
        ("nearest_candidate_error_m", "nearest_candidate"),
        ("selected_error_m", "selected"),
        ("estimate_error_m", "estimate"),
    ):
        out.update(_error_summary(prefix, pd.to_numeric(frame_rows[column], errors="coerce")))
    return out


def selected_track_stability_metrics(selected_radar: pd.DataFrame | None) -> dict[str, object]:
    """Return identity-stability metrics for selected radar rows."""

    if selected_radar is None or selected_radar.empty or "track_id" not in selected_radar.columns:
        return {
            "selected_radar_rows": 0,
            "track_switch_count": 0,
            "dominant_track_fraction": float("nan"),
            "selected_track_entropy": float("nan"),
        }
    sort_columns = [c for c in ("time_s", "frame_index") if c in selected_radar.columns]
    ordered = selected_radar.sort_values(sort_columns) if sort_columns else selected_radar
    track_ids = pd.to_numeric(ordered["track_id"], errors="coerce").dropna().astype(int)
    if track_ids.empty:
        return {
            "selected_radar_rows": int(len(ordered)),
            "track_switch_count": 0,
            "dominant_track_fraction": float("nan"),
            "selected_track_entropy": float("nan"),
        }
    values = track_ids.to_numpy(dtype=int)
    switches = int(np.count_nonzero(values[1:] != values[:-1])) if values.size > 1 else 0
    counts = track_ids.value_counts()
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0))))
    gaps = _time_gaps_s(ordered)
    return {
        "selected_radar_rows": int(len(ordered)),
        "finite_track_id_rows": int(values.size),
        "unique_selected_track_ids": int(counts.size),
        "track_switch_count": switches,
        "track_switch_rate": _safe_rate(switches, max(values.size - 1, 0)),
        "dominant_track_id": int(counts.index[0]),
        "dominant_track_fraction": float(counts.iloc[0] / counts.sum()),
        "selected_track_entropy": entropy,
        "selected_time_gap_p95_s": _percentile_or_nan(gaps, 95),
        "selected_time_gap_max_s": float(np.max(gaps)) if gaps.size else float("nan"),
    }


def confidence_diagnostics(
    estimates: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute lightweight confidence features from existing run artifacts."""

    if estimates.empty:
        return pd.DataFrame()
    out = estimates.copy()
    nis = pd.to_numeric(out.get("nis", pd.Series(np.nan, index=out.index)), errors="coerce")
    association_score = pd.to_numeric(
        out.get("association_score", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    hypothesis_count = pd.to_numeric(
        out.get("hypothesis_count", pd.Series(1.0, index=out.index)), errors="coerce"
    ).fillna(1.0)
    covariance_trace = _covariance_trace(out)
    out["confidence_nis_component"] = np.exp(
        -0.5 * np.clip(nis.fillna(0.0).to_numpy(dtype=float), 0.0, 100.0)
    )
    out["confidence_association_component"] = np.exp(
        -np.clip(association_score.fillna(0.0).to_numpy(dtype=float), 0.0, 100.0)
    )
    out["confidence_covariance_component"] = 1.0 / np.sqrt(
        1.0 + np.clip(covariance_trace, 0.0, np.inf) / 2500.0
    )
    out["confidence_hypothesis_component"] = 1.0 / np.sqrt(
        np.maximum(hypothesis_count.to_numpy(dtype=float), 1.0)
    )
    out["confidence_score"] = (
        out["confidence_nis_component"]
        * out["confidence_association_component"]
        * out["confidence_covariance_component"]
        * out["confidence_hypothesis_component"]
    )
    if selected_radar is not None and not selected_radar.empty:
        out = _merge_selected_context(out, selected_radar)
    return out


def write_oracle_gap_report(
    *,
    frame_rows: pd.DataFrame,
    output_csv: Path,
    output_json: Path | None = None,
    selected_radar: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Write frame-level and summary oracle-gap diagnostics."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_rows.to_csv(output_csv, index=False)
    summary = summarize_oracle_gap(frame_rows)
    summary.update(selected_track_stability_metrics(selected_radar))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


_ORACLE_GAP_COLUMNS = [
    "frame_key_type",
    "frame_key",
    "time_s",
    "candidate_count",
    "truth_available",
    "nearest_candidate_error_m",
    "nearest_candidate_track_id",
    "has_plausible_candidate",
    "selected_present",
    "selected_track_id",
    "selected_error_m",
    "selected_is_plausible",
    "selected_replay_accepted",
    "estimate_error_m",
    "category",
]


def _gap_category(row: dict[str, object], cfg: OracleGapConfig) -> str:
    if not bool(row["truth_available"]):
        return "no_truth"
    if not bool(row["has_plausible_candidate"]):
        return "no_plausible_candidate"
    if not bool(row["selected_present"]):
        return "plausible_candidate_not_selected"
    if row["selected_replay_accepted"] is False:
        return "selected_candidate_rejected_by_filter"
    if not bool(row["selected_is_plausible"]):
        return "wrong_candidate_selected"
    estimate_error = _finite_float_or_none(row.get("estimate_error_m"))
    if estimate_error is not None and estimate_error > float(cfg.drift_error_gate_m):
        return "filter_or_timing_drift_after_correct_selection"
    return "correct_candidate_selected"


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    sort_columns = [c for c in ("time_s", "frame_index", "track_id", "track_index") if c in radar]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True) if sort_columns else radar
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _selected_rows_by_key(selected: pd.DataFrame | None) -> dict[tuple[str, int | float], pd.Series]:
    if selected is None or selected.empty:
        return {}
    rows: dict[tuple[str, int | float], pd.Series] = {}
    for _, row in selected.iterrows():
        rows[_row_key(row)] = row
    return rows


def _frame_key(frame: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in frame.columns:
        values = pd.to_numeric(frame["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return "frame_index", int(values.iloc[0])
    time_s = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
    return "time_s", round(time_s, 9)


def _row_key(row: pd.Series) -> tuple[str, int | float]:
    frame_index = _finite_float_or_none(row.get("frame_index"))
    if frame_index is not None:
        return "frame_index", int(frame_index)
    time_s = _finite_float_or_none(row.get("time_s"))
    return "time_s", float("nan") if time_s is None else round(float(time_s), 9)


def _positions(frame: pd.DataFrame) -> np.ndarray:
    required = ["east_m", "north_m", "up_m"]
    if not all(column in frame.columns for column in required):
        return np.empty((0, 3), dtype=float)
    positions = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return positions[np.isfinite(positions).all(axis=1)]


def _row_position(row: pd.Series) -> np.ndarray | None:
    try:
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
    except (KeyError, TypeError, ValueError):
        return None
    return position if np.isfinite(position).all() else None


def _time_position_arrays(frame: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    if frame is None or frame.empty or "time_s" not in frame.columns:
        return np.empty(0), np.empty((0, 3))
    required = ["east_m", "north_m", "up_m"]
    if not all(column in frame.columns for column in required):
        return np.empty(0), np.empty((0, 3))
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    positions = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(times) & np.isfinite(positions).all(axis=1)
    return times[keep], positions[keep]


def _nearest_position(frame: pd.DataFrame, *, time_s: float, max_delta_s: float) -> np.ndarray | None:
    times, positions = _time_position_arrays(frame)
    if times.size == 0:
        return None
    index = int(np.argmin(np.abs(times - float(time_s))))
    if abs(float(times[index]) - float(time_s)) > float(max_delta_s):
        return None
    return positions[index]


def _nearest_estimate_error(
    estimate_times: np.ndarray,
    estimate_positions: np.ndarray,
    *,
    time_s: float,
    truth_position: np.ndarray,
    max_delta_s: float,
) -> float:
    if estimate_times.size == 0:
        return float("nan")
    index = int(np.argmin(np.abs(estimate_times - float(time_s))))
    if abs(float(estimate_times[index]) - float(time_s)) > float(max_delta_s):
        return float("nan")
    return float(np.linalg.norm(estimate_positions[index] - truth_position))


def _covariance_trace(frame: pd.DataFrame) -> np.ndarray:
    for column in ("covariance_trace_m2", "cov_trace_m2", "position_covariance_trace_m2"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(values), values, 0.0)
    for columns in (
        ("cov_ee", "cov_nn", "cov_uu"),
        ("variance_east_m2", "variance_north_m2", "variance_up_m2"),
        ("association_cov_ee", "association_cov_nn", "association_cov_uu"),
    ):
        if all(column in frame.columns for column in columns):
            values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            traces = np.nansum(values, axis=1)
            return np.where(np.isfinite(traces), traces, 0.0)
    return np.zeros(len(frame), dtype=float)


def _merge_selected_context(estimates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if "time_s" not in estimates.columns or "time_s" not in selected.columns:
        return estimates
    context_columns = [
        column
        for column in (
            "time_s",
            "track_id",
            "association_score",
            "association_nis",
            "association_weight_entropy",
            "association_hypothesis_count",
        )
        if column in selected.columns
    ]
    if len(context_columns) <= 1:
        return estimates
    context = selected[context_columns].copy().sort_values("time_s")
    left = estimates.copy().sort_values("time_s")
    return pd.merge_asof(left, context, on="time_s", direction="nearest", tolerance=0.25)


def _time_gaps_s(frame: pd.DataFrame) -> np.ndarray:
    if "time_s" not in frame.columns:
        return np.empty(0)
    times = pd.to_numeric(frame["time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return np.diff(np.sort(times)) if times.size >= 2 else np.empty(0)


def _error_summary(prefix: str, values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_error_count": 0,
            f"{prefix}_error_mean_m": float("nan"),
            f"{prefix}_error_rmse_m": float("nan"),
            f"{prefix}_error_p95_m": float("nan"),
        }
    return {
        f"{prefix}_error_count": int(array.size),
        f"{prefix}_error_mean_m": float(np.mean(array)),
        f"{prefix}_error_rmse_m": float(np.sqrt(np.mean(array**2))),
        f"{prefix}_error_p95_m": float(np.percentile(array, 95)),
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    return float("nan") if denominator <= 0.0 else float(numerator) / denominator


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _optional_track_id(value: object) -> object:
    number = _finite_float_or_none(value)
    return "" if number is None else int(number)


def _optional_bool(value: object) -> bool | str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return ""
    return bool(value)


def _finite_float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
