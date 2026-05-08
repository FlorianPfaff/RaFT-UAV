"""RTS, fixed-lag, and robust MAP smoothing for CV tracking records."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.record_helpers import copy_record, record_arrays, symmetrized
from raft_uav.baselines.robust_map import RobustMapSmootherConfig, robust_map_smooth_records

SMOOTHER_MODES = ("none", "rts", "fixed-lag", "robust-map", "fixed-lag-map")


def smooth_tracking_records(
    records: list[dict[str, object]],
    *,
    method: str,
    acceleration_std_mps2: float,
    lag_s: float | None = None,
    measurements: Iterable[TrackingMeasurement] | None = None,
    robust_map_config: RobustMapSmootherConfig | None = None,
) -> list[dict[str, object]]:
    """Return tracking records with smoothed state/covariance estimates.

    ``rts`` is a full offline Rauch--Tung--Striebel pass. ``fixed-lag`` applies
    the same backward recursion only over future records within ``lag_s``.
    ``robust-map`` and ``fixed-lag-map`` solve a robust constant-velocity factor
    graph over all records or a bounded future window, respectively.
    """

    if method not in SMOOTHER_MODES:
        raise ValueError(f"unknown smoother method {method!r}")
    if method == "none" or not records:
        return [copy_record(record) for record in records]
    if method in ("fixed-lag", "fixed-lag-map") and (lag_s is None or lag_s < 0.0):
        raise ValueError(f"{method} smoothing requires a nonnegative lag_s")
    if method in ("robust-map", "fixed-lag-map"):
        return robust_map_smooth_records(
            records,
            measurements=measurements,
            acceleration_std_mps2=acceleration_std_mps2,
            config=robust_map_config,
            lag_s=None if method == "robust-map" else float(lag_s),
        )

    times, filtered_states, filtered_covariances = record_arrays(records)
    if method == "rts":
        smoothed_states, smoothed_covariances = _rts_smooth(
            times,
            filtered_states,
            filtered_covariances,
            acceleration_std_mps2=acceleration_std_mps2,
            start_index=0,
            end_index=len(records) - 1,
        )
    else:
        smoothed_states, smoothed_covariances = _fixed_lag_smooth(
            times,
            filtered_states,
            filtered_covariances,
            acceleration_std_mps2=acceleration_std_mps2,
            lag_s=float(lag_s),
        )

    out: list[dict[str, object]] = []
    for idx, record in enumerate(records):
        item = copy_record(record)
        item["filtered_state"] = filtered_states[idx].copy()
        item["filtered_covariance"] = filtered_covariances[idx].copy()
        item["state"] = smoothed_states[idx].copy()
        item["covariance"] = smoothed_covariances[idx].copy()
        item["smoother_method"] = method
        item["smoother_lag_s"] = None if method == "rts" else float(lag_s)
        out.append(item)
    return out


def _fixed_lag_smooth(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    *,
    acceleration_std_mps2: float,
    lag_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for start_index, time_s in enumerate(times):
        end_index = int(np.searchsorted(times, time_s + lag_s, side="right") - 1)
        if end_index <= start_index:
            continue
        states, covariances = _rts_smooth(
            times,
            filtered_states,
            filtered_covariances,
            acceleration_std_mps2=acceleration_std_mps2,
            start_index=start_index,
            end_index=end_index,
        )
        smoothed_states[start_index] = states[start_index]
        smoothed_covariances[start_index] = covariances[start_index]
    return smoothed_states, smoothed_covariances


def _rts_smooth(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    *,
    acceleration_std_mps2: float,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for idx in range(end_index - 1, start_index - 1, -1):
        transition, predicted_state, predicted_covariance = _predict_from_record(
            times,
            filtered_states,
            filtered_covariances,
            idx,
            acceleration_std_mps2=acceleration_std_mps2,
        )
        gain = _smoothing_gain(filtered_covariances[idx], transition, predicted_covariance)
        smoothed_states[idx] = filtered_states[idx] + gain @ (
            smoothed_states[idx + 1] - predicted_state
        )
        smoothed_covariances[idx] = symmetrized(
            filtered_covariances[idx]
            + gain @ (smoothed_covariances[idx + 1] - predicted_covariance) @ gain.T
        )
    return smoothed_states, smoothed_covariances


def _predict_from_record(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    index: int,
    *,
    acceleration_std_mps2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt_s = float(times[index + 1] - times[index])
    if dt_s < -1e-9:
        raise ValueError("smoothing records must be sorted by time")
    transition = constant_velocity_matrix(max(0.0, dt_s))
    process_noise = white_acceleration_process_noise(max(0.0, dt_s), acceleration_std_mps2)
    predicted_state = transition @ filtered_states[index]
    predicted_covariance = symmetrized(
        transition @ filtered_covariances[index] @ transition.T + process_noise
    )
    return transition, predicted_state, predicted_covariance


def _smoothing_gain(
    filtered_covariance: np.ndarray,
    transition: np.ndarray,
    predicted_covariance: np.ndarray,
) -> np.ndarray:
    right = filtered_covariance @ transition.T
    try:
        return np.linalg.solve(predicted_covariance.T, right.T).T
    except np.linalg.LinAlgError:
        return right @ np.linalg.pinv(predicted_covariance)
