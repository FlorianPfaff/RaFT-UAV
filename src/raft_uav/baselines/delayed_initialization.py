"""Delayed and multi-hypothesis initialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InitialHypothesis:
    """One candidate initial 6D state."""

    time_s: float
    state: np.ndarray
    covariance: np.ndarray
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float).reshape(6)
        covariance = np.asarray(self.covariance, dtype=float).reshape(6, 6)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "score", float(self.score))


def build_delayed_initial_hypotheses(
    *,
    rf_measurements: Iterable[Any],
    radar: pd.DataFrame,
    window_s: float = 5.0,
    max_hypotheses: int = 8,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
) -> list[InitialHypothesis]:
    """Build initial-state candidates from the first short RF/radar window."""

    rf = list(rf_measurements)
    radar_window = _first_radar_window(radar, window_s=window_s)
    hypotheses: list[InitialHypothesis] = []
    for measurement in rf:
        vector = np.asarray(getattr(measurement, "vector", []), dtype=float).reshape(-1)
        if vector.size < 2:
            continue
        time_s = float(getattr(measurement, "time_s"))
        state = np.zeros(6)
        state[: min(vector.size, 3)] = vector[: min(vector.size, 3)]
        hypotheses.append(
            InitialHypothesis(
                time_s=time_s,
                state=state,
                covariance=_initial_covariance(initial_position_std_m, initial_velocity_std_mps),
                score=_rf_support_score(time_s, state[:3], radar_window),
                source="rf",
                metadata={"rf_dimension": int(vector.size)},
            )
        )
    for _, row in radar_window.iterrows():
        state = _radar_row_state(row, radar_window)
        if state is None:
            continue
        catprob = _optional_float(row.get("cat_prob_uav"))
        catprob_penalty = 0.0 if catprob is None else float(-np.log(np.clip(catprob, 1e-6, 1.0)))
        support = _track_support_score(row, radar_window)
        hypotheses.append(
            InitialHypothesis(
                time_s=float(row["time_s"]),
                state=state,
                covariance=_initial_covariance(initial_position_std_m, initial_velocity_std_mps),
                score=float(catprob_penalty + support),
                source="radar",
                metadata={
                    "track_id": _optional_int(row.get("track_id")),
                    "cat_prob_uav": catprob,
                    "support_score": support,
                },
            )
        )
    return sorted(hypotheses, key=lambda item: item.score)[: int(max_hypotheses)]


def best_initial_hypothesis(hypotheses: Iterable[InitialHypothesis]) -> InitialHypothesis | None:
    """Return the lowest-score initial hypothesis."""

    items = list(hypotheses)
    return min(items, key=lambda item: item.score) if items else None


def _first_radar_window(radar: pd.DataFrame, *, window_s: float) -> pd.DataFrame:
    if radar.empty or "time_s" not in radar.columns:
        return radar.iloc[0:0].copy()
    ordered = radar.sort_values("time_s").reset_index(drop=True)
    start = float(pd.to_numeric(ordered["time_s"], errors="coerce").min())
    return ordered.loc[ordered["time_s"] <= start + float(window_s)].copy()


def _radar_row_state(row: pd.Series, frame: pd.DataFrame) -> np.ndarray | None:
    try:
        state = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"]), 0, 0, 0], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(state[:3]).all():
        return None
    velocity = _velocity_from_row(row)
    if velocity is None:
        velocity = _velocity_from_track(row, frame)
    if velocity is not None:
        state[3:6] = velocity
    return state


def _velocity_from_row(row: pd.Series) -> np.ndarray | None:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row.index for column in required):
        return None
    try:
        velocity = np.array(
            [float(row["velocity_east_mps"]), float(row["velocity_north_mps"]), -float(row["velocity_down_mps"])],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _velocity_from_track(row: pd.Series, frame: pd.DataFrame) -> np.ndarray | None:
    track_id = _optional_int(row.get("track_id"))
    if track_id is None or "track_id" not in frame.columns:
        return None
    track = frame.loc[pd.to_numeric(frame["track_id"], errors="coerce") == track_id].sort_values("time_s")
    if len(track) < 2:
        return None
    positions = track[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    times = track["time_s"].to_numpy(dtype=float)
    dt = float(times[-1] - times[0])
    if dt <= 0.0:
        return None
    velocity = (positions[-1] - positions[0]) / dt
    return velocity if np.isfinite(velocity).all() else None


def _rf_support_score(time_s: float, position: np.ndarray, radar: pd.DataFrame) -> float:
    if radar.empty:
        return 1.0
    dt = np.abs(pd.to_numeric(radar["time_s"], errors="coerce").to_numpy(dtype=float) - float(time_s))
    nearby = radar.loc[dt <= 1.0]
    if nearby.empty:
        return 1.0
    positions = nearby[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    distances = np.linalg.norm(positions - position.reshape(1, 3), axis=1)
    return float(np.nanmin(distances) / 100.0)


def _track_support_score(row: pd.Series, radar: pd.DataFrame) -> float:
    track_id = _optional_int(row.get("track_id"))
    if track_id is None or "track_id" not in radar.columns:
        return 1.0
    count = int(np.count_nonzero(pd.to_numeric(radar["track_id"], errors="coerce") == track_id))
    return float(1.0 / max(count, 1))


def _initial_covariance(position_std_m: float, velocity_std_mps: float) -> np.ndarray:
    return np.diag([position_std_m**2] * 3 + [velocity_std_mps**2] * 3)


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)
