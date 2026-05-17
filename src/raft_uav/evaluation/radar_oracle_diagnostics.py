"""Radar nearest-candidate and timestamp-offset diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

PAPER_METRIC_COLUMNS = (
    "count",
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
)


def interpolate_truth_positions(
    truth: pd.DataFrame,
    query_times_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    query_times = np.asarray(list(query_times_s), dtype=float).reshape(-1)
    positions = np.full((query_times.size, 3), np.nan, dtype=float)
    valid = np.zeros(query_times.size, dtype=bool)
    if query_times.size == 0 or truth.empty:
        return positions, valid
    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(truth.columns):
        raise KeyError(f"truth is missing required columns: {sorted(required - set(truth.columns))}")
    ordered = truth.sort_values("time_s").reset_index(drop=True)
    truth_times = ordered["time_s"].to_numpy(dtype=float)
    truth_xyz = ordered[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    finite = np.isfinite(truth_times) & np.isfinite(truth_xyz).all(axis=1)
    truth_times = truth_times[finite]
    truth_xyz = truth_xyz[finite]
    if truth_times.size == 0:
        return positions, valid
    for idx, query_time in enumerate(query_times):
        if not np.isfinite(query_time):
            continue
        insertion = int(np.searchsorted(truth_times, query_time))
        if insertion < truth_times.size and np.isclose(truth_times[insertion], query_time):
            nearest_delta = 0.0
            interpolated = truth_xyz[insertion]
        elif insertion == 0 or insertion >= truth_times.size:
            continue
        else:
            left = insertion - 1
            right = insertion
            left_time = float(truth_times[left])
            right_time = float(truth_times[right])
            if right_time <= left_time:
                continue
            nearest_delta = min(abs(query_time - left_time), abs(right_time - query_time))
            alpha = (query_time - left_time) / (right_time - left_time)
            interpolated = truth_xyz[left] + alpha * (truth_xyz[right] - truth_xyz[left])
        if max_time_delta_s is not None and nearest_delta > float(max_time_delta_s):
            continue
        if np.isfinite(interpolated).all():
            positions[idx] = interpolated
            valid[idx] = True
    return positions, valid


def nearest_candidate_oracle(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    time_offset_s: float = 0.0,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    if radar.empty:
        return _empty_oracle_selection(radar)
    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(radar.columns):
        raise KeyError(f"radar is missing required columns: {sorted(required - set(radar.columns))}")
    rows: list[pd.Series] = []
    for frame in _radar_frame_groups(radar):
        frame_time = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
        truth_position, valid = interpolate_truth_positions(
            truth,
            [frame_time + float(time_offset_s)],
            max_time_delta_s=max_time_delta_s,
        )
        if not bool(valid[0]):
            continue
        xyz = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.any():
            continue
        errors_3d = np.full(len(frame), np.inf, dtype=float)
        errors_2d = np.full(len(frame), np.inf, dtype=float)
        residuals = xyz[finite] - truth_position[0]
        errors_3d[finite] = np.linalg.norm(residuals, axis=1)
        errors_2d[finite] = np.linalg.norm(residuals[:, :2], axis=1)
        best = int(np.argmin(errors_3d))
        selected = frame.iloc[best].copy()
        selected["oracle_time_offset_s"] = float(time_offset_s)
        selected["oracle_truth_time_s"] = frame_time + float(time_offset_s)
        selected["oracle_error_3d_m"] = float(errors_3d[best])
        selected["oracle_error_2d_m"] = float(errors_2d[best])
        selected["oracle_candidate_rows"] = int(len(frame))
        selected["association_mode"] = "oracle-nearest-candidate"
        rows.append(selected)
    if not rows:
        return _empty_oracle_selection(radar)
    selected = pd.DataFrame(rows)
    sort_columns = [c for c in ("time_s", "frame_index", "track_id", "track_index") if c in selected.columns]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def time_offset_sweep(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    frame_count = len(_radar_frame_groups(radar)) if not radar.empty else 0
    rows: list[dict[str, float]] = []
    for offset_s in offsets_s:
        selected = nearest_candidate_oracle(
            radar,
            truth,
            time_offset_s=float(offset_s),
            max_time_delta_s=max_time_delta_s,
        )
        row = summarize_oracle_selection(selected, frame_count=frame_count)
        row["time_offset_s"] = float(offset_s)
        rows.append(row)
    columns = ["time_offset_s", *PAPER_METRIC_COLUMNS]
    return pd.DataFrame.from_records(rows, columns=columns)


def best_time_offset(sweep: pd.DataFrame, *, metric: str = "mean_3d_error_m") -> float | None:
    if sweep.empty or metric not in sweep.columns:
        return None
    values = pd.to_numeric(sweep[metric], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    finite_indices = np.flatnonzero(finite)
    best = finite_indices[int(np.argmin(values[finite]))]
    return float(sweep.iloc[best]["time_offset_s"])


def summarize_oracle_selection(selected: pd.DataFrame, *, frame_count: int | None = None) -> dict[str, float]:
    if selected.empty:
        denominator = float(frame_count or 0)
        coverage = 0.0 if denominator > 0.0 else float("nan")
        empty = {column: float("nan") for column in PAPER_METRIC_COLUMNS if column not in {"count", "coverage"}}
        return {"count": 0.0, "coverage": coverage, **empty}
    e3 = pd.to_numeric(selected["oracle_error_3d_m"], errors="coerce").dropna().to_numpy(dtype=float)
    e2 = pd.to_numeric(selected["oracle_error_2d_m"], errors="coerce").dropna().to_numpy(dtype=float)
    e3 = e3[np.isfinite(e3)]
    e2 = e2[np.isfinite(e2)]
    count = int(e3.size)
    denominator = float(frame_count if frame_count is not None else count)
    coverage = float(count / denominator) if denominator > 0.0 else float("nan")
    return {"count": float(count), "coverage": coverage, **_stats(e3, "3d"), **_stats(e2, "2d")}


def _stats(errors: np.ndarray, suffix: str) -> dict[str, float]:
    if errors.size == 0:
        return {
            f"mean_{suffix}_error_m": float("nan"),
            f"std_{suffix}_error_m": float("nan"),
            f"rmse_{suffix}_error_m": float("nan"),
            f"p95_{suffix}_error_m": float("nan"),
            f"max_{suffix}_error_m": float("nan"),
        }
    return {
        f"mean_{suffix}_error_m": float(np.mean(errors)),
        f"std_{suffix}_error_m": float(np.std(errors)),
        f"rmse_{suffix}_error_m": float(np.sqrt(np.mean(errors**2))),
        f"p95_{suffix}_error_m": float(np.percentile(errors, 95)),
        f"max_{suffix}_error_m": float(np.max(errors)),
    }


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty:
        return []
    sort_columns = [c for c in ("time_s", "frame_index", "track_id", "track_index") if c in radar.columns]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _empty_oracle_selection(radar: pd.DataFrame) -> pd.DataFrame:
    selected = radar.iloc[0:0].copy()
    for column in (
        "oracle_time_offset_s",
        "oracle_truth_time_s",
        "oracle_error_3d_m",
        "oracle_error_2d_m",
        "oracle_candidate_rows",
        "association_mode",
    ):
        selected[column] = []
    return selected
