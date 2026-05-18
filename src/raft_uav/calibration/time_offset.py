"""LOFO-safe RF/radar timestamp-offset calibration helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import (
    best_time_offset,
    interpolate_truth_positions,
    time_offset_sweep,
)

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


@dataclass(frozen=True)
class TimeOffsetFitResult:
    """Best timestamp correction selected from an offset sweep."""

    source: str
    best_offset_s: float | None
    metric: str
    sweep: pd.DataFrame

    def summary(self) -> dict[str, float | str | None]:
        out: dict[str, float | str | None] = {
            "source": self.source,
            "metric": self.metric,
            "best_offset_s": self.best_offset_s,
        }
        if self.best_offset_s is None or self.sweep.empty:
            return out
        rows = self.sweep.loc[self.sweep["time_offset_s"] == float(self.best_offset_s)]
        if rows.empty:
            return out
        row = rows.iloc[0]
        for column in PAPER_METRIC_COLUMNS:
            if column in row.index:
                out[column] = _optional_float(row[column])
        return out


def make_offset_grid(min_s: float, max_s: float, step_s: float) -> np.ndarray:
    """Return an inclusive timestamp-offset grid."""

    if step_s <= 0.0:
        raise ValueError("step_s must be positive")
    if max_s < min_s:
        raise ValueError("max_s must be greater than or equal to min_s")
    count = int(np.floor((float(max_s) - float(min_s)) / float(step_s))) + 1
    offsets = float(min_s) + np.arange(count, dtype=float) * float(step_s)
    if offsets.size == 0 or offsets[-1] < float(max_s) - 1.0e-9:
        offsets = np.append(offsets, float(max_s))
    return np.round(offsets, decimals=9)


def apply_time_offset(
    frame: pd.DataFrame,
    offset_s: float | None,
    *,
    time_column: str = "time_s",
    copy_uncorrected: bool = True,
) -> pd.DataFrame:
    """Return a copy whose time column is shifted by ``offset_s`` seconds."""

    if time_column not in frame.columns:
        raise KeyError(f"frame is missing time column {time_column!r}")
    offset = 0.0 if offset_s is None else float(offset_s)
    out = frame.copy()
    raw_time = pd.to_numeric(out[time_column], errors="coerce")
    if copy_uncorrected:
        raw_column = f"{time_column}_uncorrected"
        if raw_column not in out.columns:
            out[raw_column] = raw_time
    out[time_column] = raw_time + offset
    out["time_offset_correction_s"] = offset
    return out


def fit_radar_time_offset(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
    metric: str = "mean_3d_error_m",
) -> TimeOffsetFitResult:
    """Fit a radar timestamp offset using training-flight oracle sweeps only."""

    sweep = aggregate_radar_time_offset_sweep(
        training_pairs,
        offsets_s,
        max_time_delta_s=max_time_delta_s,
    )
    return TimeOffsetFitResult(
        source="radar",
        best_offset_s=best_time_offset(sweep, metric=metric),
        metric=metric,
        sweep=sweep,
    )


def fit_measurement_time_offset(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    dimensions: int,
    max_time_delta_s: float | None = 2.0,
    metric: str | None = None,
) -> TimeOffsetFitResult:
    """Fit a point-measurement timestamp offset, for example for RF rows."""

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    selected_metric = metric or ("mean_2d_error_m" if dimensions == 2 else "mean_3d_error_m")
    sweep = aggregate_measurement_time_offset_sweep(
        training_pairs,
        offsets_s,
        dimensions=dimensions,
        max_time_delta_s=max_time_delta_s,
    )
    return TimeOffsetFitResult(
        source="measurement",
        best_offset_s=best_time_offset(sweep, metric=selected_metric),
        metric=selected_metric,
        sweep=sweep,
    )


def aggregate_radar_time_offset_sweep(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Aggregate radar nearest-candidate oracle sweeps over training flights."""

    offsets = list(float(offset) for offset in offsets_s)
    rows: list[dict[str, float]] = []
    for offset in offsets:
        selected: list[pd.DataFrame] = []
        frame_count = 0
        for radar, truth in training_pairs:
            if radar.empty or truth.empty:
                continue
            sweep = time_offset_sweep(
                radar,
                truth,
                [offset],
                max_time_delta_s=max_time_delta_s,
            )
            if sweep.empty:
                continue
            row = sweep.iloc[0].to_dict()
            frame_count += int(row.get("count", 0) / max(float(row.get("coverage", 1.0)), 1e-12))
            selected.append(_radar_selected_errors_from_summary(row))
        rows.append(_aggregate_error_frames(offset, selected, frame_count))
    return pd.DataFrame.from_records(rows, columns=["time_offset_s", *PAPER_METRIC_COLUMNS])


def aggregate_measurement_time_offset_sweep(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    dimensions: int,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Aggregate direct measurement-to-truth error sweeps over training flights."""

    rows: list[dict[str, float]] = []
    for offset in offsets_s:
        errors_2d: list[np.ndarray] = []
        errors_3d: list[np.ndarray] = []
        total_rows = 0
        for measurements, truth in training_pairs:
            if measurements.empty or truth.empty:
                continue
            required = {"time_s", "east_m", "north_m"}
            if dimensions == 3:
                required.add("up_m")
            if not required.issubset(measurements.columns):
                continue
            total_rows += len(measurements)
            query = pd.to_numeric(measurements["time_s"], errors="coerce").to_numpy(dtype=float)
            truth_xyz, valid = interpolate_truth_positions(
                truth,
                query + float(offset),
                max_time_delta_s=max_time_delta_s,
            )
            meas_cols = ["east_m", "north_m", "up_m"] if dimensions == 3 else ["east_m", "north_m"]
            meas = measurements[meas_cols].to_numpy(dtype=float)
            finite = valid & np.isfinite(meas).all(axis=1)
            if not finite.any():
                continue
            residual = meas[finite] - truth_xyz[finite, :dimensions]
            err = np.linalg.norm(residual, axis=1)
            if dimensions == 2:
                errors_2d.append(err)
            else:
                errors_3d.append(err)
                errors_2d.append(np.linalg.norm(residual[:, :2], axis=1))
        e2 = _concat(errors_2d)
        e3 = _concat(errors_3d) if dimensions == 3 else np.array([], dtype=float)
        row = {"time_offset_s": float(offset), "count": float(e2.size), "coverage": _coverage(e2.size, total_rows)}
        row.update(_stats(e3, "3d"))
        row.update(_stats(e2, "2d"))
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=["time_offset_s", *PAPER_METRIC_COLUMNS])


def _radar_selected_errors_from_summary(summary: dict[str, float]) -> pd.DataFrame:
    # The radar oracle diagnostic only exposes aggregate statistics here.  For
    # offset selection across flights, weighting by count and combining mean/RMSE
    # is sufficient and avoids reading every selected row again.
    return pd.DataFrame([summary])


def _aggregate_error_frames(offset: float, frames: list[pd.DataFrame], frame_count: int) -> dict[str, float]:
    if not frames:
        row = {"time_offset_s": float(offset), "count": 0.0, "coverage": 0.0 if frame_count else float("nan")}
        row.update(_stats(np.array([], dtype=float), "3d"))
        row.update(_stats(np.array([], dtype=float), "2d"))
        return row
    summary = pd.concat(frames, ignore_index=True)
    counts = pd.to_numeric(summary["count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = float(np.sum(counts))
    coverage_den = float(frame_count) if frame_count > 0 else total
    row = {"time_offset_s": float(offset), "count": total, "coverage": _coverage(total, coverage_den)}
    for dims in ("3d", "2d"):
        mean = _weighted(summary[f"mean_{dims}_error_m"], counts)
        rmse = np.sqrt(_weighted(np.square(pd.to_numeric(summary[f"rmse_{dims}_error_m"], errors="coerce")), counts))
        p95 = _weighted(summary[f"p95_{dims}_error_m"], counts)
        max_value = np.nanmax(pd.to_numeric(summary[f"max_{dims}_error_m"], errors="coerce").to_numpy(dtype=float))
        std = _weighted(summary[f"std_{dims}_error_m"], counts)
        row[f"mean_{dims}_error_m"] = float(mean)
        row[f"std_{dims}_error_m"] = float(std)
        row[f"rmse_{dims}_error_m"] = float(rmse)
        row[f"p95_{dims}_error_m"] = float(p95)
        row[f"max_{dims}_error_m"] = float(max_value)
    return row


def _stats(errors: np.ndarray, suffix: str) -> dict[str, float]:
    errors = np.asarray(errors, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
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


def _coverage(count: float | int, denominator: float | int) -> float:
    denominator = float(denominator)
    return float(count) / denominator if denominator > 0.0 else float("nan")


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _weighted(values: pd.Series | np.ndarray, weights: np.ndarray) -> float:
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(values_array[valid], weights=weights[valid]))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
