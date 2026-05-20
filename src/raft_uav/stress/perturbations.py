"""Deterministic stress-test perturbations for normalized RF/radar frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for one stress-test perturbation set."""

    name: str
    radar_drop_rate: float = 0.0
    rf_drop_burst_rate: float = 0.0
    timestamp_jitter_std_s: float = 0.0
    false_tracks_per_frame: int = 0
    false_track_position_std_m: float = 150.0
    velocity_noise_std_mps: float = 0.0
    covariance_scale: float = 1.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PerturbationConfig":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__ if field in payload})


def perturb_radar(radar: pd.DataFrame, config: PerturbationConfig) -> pd.DataFrame:
    """Return a perturbed radar frame."""

    rng = np.random.default_rng(int(config.seed))
    out = radar.copy()
    out = drop_radar_frames(out, rate=float(config.radar_drop_rate), rng=rng)
    out = jitter_timestamps(out, std_s=float(config.timestamp_jitter_std_s), rng=rng)
    out = corrupt_velocity(out, std_mps=float(config.velocity_noise_std_mps), rng=rng)
    out = scale_covariance_columns(out, scale=float(config.covariance_scale))
    out = inject_false_tracks(
        out,
        false_tracks_per_frame=int(config.false_tracks_per_frame),
        position_std_m=float(config.false_track_position_std_m),
        rng=rng,
    )
    out["stress_config"] = config.name
    sort_columns = [c for c in ("time_s", "frame_index", "track_id") if c in out.columns]
    return out.sort_values(sort_columns) if sort_columns else out


def perturb_rf(rf: pd.DataFrame, config: PerturbationConfig) -> pd.DataFrame:
    """Return a perturbed RF frame."""

    rng = np.random.default_rng(int(config.seed) + 17)
    out = rf.copy()
    out = drop_rf_bursts(out, rate=float(config.rf_drop_burst_rate), rng=rng)
    out = jitter_timestamps(out, std_s=float(config.timestamp_jitter_std_s), rng=rng)
    out = scale_covariance_columns(out, scale=float(config.covariance_scale))
    out["stress_config"] = config.name
    return out.sort_values("time_s") if "time_s" in out.columns else out


def drop_radar_frames(frame: pd.DataFrame, *, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty or rate <= 0.0:
        return frame.copy()
    group_column = "frame_index" if "frame_index" in frame.columns else "time_s"
    groups = pd.Series(frame[group_column].dropna().unique())
    keep_groups = set(groups.loc[rng.random(len(groups)) >= min(max(rate, 0.0), 1.0)].tolist())
    return frame.loc[frame[group_column].isin(keep_groups)].copy()


def drop_rf_bursts(frame: pd.DataFrame, *, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    if frame.empty or rate <= 0.0 or "time_s" not in frame.columns:
        return frame.copy()
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    if times.size == 0:
        return frame.copy()
    bins = np.floor((times - np.nanmin(times)) / 5.0).astype(int)
    unique = np.unique(bins)
    dropped = set(unique[rng.random(unique.size) < min(max(rate, 0.0), 1.0)].tolist())
    return frame.loc[[int(bin_id) not in dropped for bin_id in bins]].copy()


def jitter_timestamps(frame: pd.DataFrame, *, std_s: float, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    if std_s <= 0.0 or "time_s" not in out.columns:
        return out
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce") + rng.normal(0.0, std_s, len(out))
    return out


def corrupt_velocity(frame: pd.DataFrame, *, std_mps: float, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    if std_mps <= 0.0:
        return out
    for column in ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") + rng.normal(0.0, std_mps, len(out))
    return out


def scale_covariance_columns(frame: pd.DataFrame, *, scale: float) -> pd.DataFrame:
    out = frame.copy()
    if scale == 1.0:
        return out
    for column in out.columns:
        if column.startswith("cov_") or column.startswith("association_cov_"):
            out[column] = pd.to_numeric(out[column], errors="coerce") * float(scale)
    return out


def inject_false_tracks(
    frame: pd.DataFrame,
    *,
    false_tracks_per_frame: int,
    position_std_m: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if frame.empty or false_tracks_per_frame <= 0 or not {"east_m", "north_m", "up_m"}.issubset(frame.columns):
        return frame.copy()
    group_column = "frame_index" if "frame_index" in frame.columns else "time_s"
    rows: list[pd.Series] = []
    next_track_id = _next_false_track_id(frame)
    for _, group in frame.groupby(group_column, sort=True):
        reference = group.iloc[0]
        center = group[["east_m", "north_m", "up_m"]].mean().to_numpy(dtype=float)
        for index in range(false_tracks_per_frame):
            row = reference.copy()
            position = center + rng.normal(0.0, position_std_m, 3)
            row["east_m"], row["north_m"], row["up_m"] = [float(value) for value in position]
            row["track_id"] = next_track_id + index
            row["cat_prob_uav"] = min(float(row.get("cat_prob_uav", 0.2)), 0.2)
            row["stress_false_track"] = True
            rows.append(row)
        next_track_id += false_tracks_per_frame
    if not rows:
        return frame.copy()
    out = pd.concat([frame.copy(), pd.DataFrame(rows)], ignore_index=True)
    if "stress_false_track" not in out.columns:
        out["stress_false_track"] = False
    out["stress_false_track"] = out["stress_false_track"].fillna(False).astype(bool)
    return out


def _next_false_track_id(frame: pd.DataFrame) -> int:
    if "track_id" not in frame.columns:
        return 10_000_000
    values = pd.to_numeric(frame["track_id"], errors="coerce").dropna()
    return 10_000_000 if values.empty else int(values.max()) + 1
