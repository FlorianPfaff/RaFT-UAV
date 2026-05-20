"""Temporal Fortem track-level features for radar association."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_track_level_features(radar: pd.DataFrame, *, window_frames: int = 10) -> pd.DataFrame:
    """Append causal track-level features to normalized radar rows."""

    if radar.empty or "track_id" not in radar.columns:
        return radar.copy()
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    sort_columns = [c for c in ("time_s", "frame_index", "track_index") if c in radar.columns]
    out = radar.sort_values(sort_columns).copy() if sort_columns else radar.copy()
    feature_frames: list[pd.DataFrame] = []
    for _, group in out.groupby("track_id", sort=False, dropna=False):
        feature_frames.append(_features_for_track(group.copy(), window_frames=window_frames))
    featured = pd.concat(feature_frames, ignore_index=False).sort_index()
    return featured.loc[radar.index].copy() if set(featured.index) == set(radar.index) else featured


def _features_for_track(group: pd.DataFrame, *, window_frames: int) -> pd.DataFrame:
    group = group.sort_values([c for c in ("time_s", "frame_index", "track_index") if c in group.columns])
    n = len(group)
    group["track_age_frames"] = np.arange(n, dtype=float)
    group["track_hit_streak_frames"] = _hit_streak(group)
    group["track_time_since_first_s"] = _time_since_first(group)
    group["track_frame_gap"] = _frame_gap(group)
    group["track_position_step_m"] = _position_step(group)
    group["track_speed_from_positions_mps"] = _speed_from_positions(group)
    group["track_range_rate_mps"] = _range_rate(group)
    if "cat_prob_uav" in group.columns:
        cat = pd.to_numeric(group["cat_prob_uav"], errors="coerce")
        group["track_catprob_mean_window"] = cat.rolling(window_frames, min_periods=1).mean().to_numpy(dtype=float)
        group["track_catprob_min_window"] = cat.rolling(window_frames, min_periods=1).min().to_numpy(dtype=float)
    else:
        group["track_catprob_mean_window"] = np.nan
        group["track_catprob_min_window"] = np.nan
    group["track_velocity_smoothness_mps"] = _velocity_smoothness(group, window_frames=window_frames)
    return group


def _hit_streak(group: pd.DataFrame) -> np.ndarray:
    if "frame_index" not in group.columns:
        return np.arange(1, len(group) + 1, dtype=float)
    frame_index = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
    streak = np.ones(len(group), dtype=float)
    for i in range(1, len(group)):
        if np.isfinite(frame_index[i]) and np.isfinite(frame_index[i - 1]) and frame_index[i] - frame_index[i - 1] <= 1.5:
            streak[i] = streak[i - 1] + 1.0
    return streak


def _time_since_first(group: pd.DataFrame) -> np.ndarray:
    times = pd.to_numeric(group.get("time_s", pd.Series(np.nan, index=group.index)), errors="coerce").to_numpy(dtype=float)
    if times.size == 0 or not np.isfinite(times[0]):
        return np.full(len(group), np.nan)
    return times - times[0]


def _frame_gap(group: pd.DataFrame) -> np.ndarray:
    if "frame_index" in group.columns:
        values = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
    else:
        values = pd.to_numeric(group.get("time_s", pd.Series(np.nan, index=group.index)), errors="coerce").to_numpy(dtype=float)
    gaps = np.r_[0.0, np.diff(values)]
    return np.where(np.isfinite(gaps), gaps, np.nan)


def _position_step(group: pd.DataFrame) -> np.ndarray:
    if not {"east_m", "north_m", "up_m"}.issubset(group.columns):
        return np.full(len(group), np.nan)
    positions = group[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    diffs = np.diff(positions, axis=0)
    steps = np.r_[0.0, np.linalg.norm(diffs, axis=1)]
    return np.where(np.isfinite(steps), steps, np.nan)


def _speed_from_positions(group: pd.DataFrame) -> np.ndarray:
    steps = _position_step(group)
    times = pd.to_numeric(group.get("time_s", pd.Series(np.nan, index=group.index)), errors="coerce").to_numpy(dtype=float)
    dt = np.r_[np.nan, np.diff(times)]
    speed = np.divide(steps, dt, out=np.full(len(group), np.nan), where=dt > 0.0)
    speed[0] = np.nan
    return speed


def _range_rate(group: pd.DataFrame) -> np.ndarray:
    if "range_m" in group.columns:
        ranges = pd.to_numeric(group["range_m"], errors="coerce").to_numpy(dtype=float)
    elif {"east_m", "north_m", "up_m"}.issubset(group.columns):
        ranges = np.linalg.norm(group[["east_m", "north_m", "up_m"]].to_numpy(dtype=float), axis=1)
    else:
        return np.full(len(group), np.nan)
    times = pd.to_numeric(group.get("time_s", pd.Series(np.nan, index=group.index)), errors="coerce").to_numpy(dtype=float)
    dt = np.r_[np.nan, np.diff(times)]
    dr = np.r_[np.nan, np.diff(ranges)]
    return np.divide(dr, dt, out=np.full(len(group), np.nan), where=dt > 0.0)


def _velocity_smoothness(group: pd.DataFrame, *, window_frames: int) -> np.ndarray:
    required = ["velocity_east_mps", "velocity_north_mps", "velocity_down_mps"]
    if not all(column in group.columns for column in required):
        return np.full(len(group), np.nan)
    velocity = np.column_stack(
        [
            pd.to_numeric(group["velocity_east_mps"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(group["velocity_north_mps"], errors="coerce").to_numpy(dtype=float),
            -pd.to_numeric(group["velocity_down_mps"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    diffs = np.r_[np.zeros((1, 3)), np.diff(velocity, axis=0)]
    norms = np.linalg.norm(diffs, axis=1)
    return pd.Series(norms).rolling(window_frames, min_periods=1).mean().to_numpy(dtype=float)
