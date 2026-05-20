"""Runtime wiring for range-dependent radar covariance.

This module is intentionally small and conservative: it preserves the existing
public API and teaches the existing radar-association helpers to consume
candidate-specific covariance columns.  The active covariance model is controlled
by environment variables and defaults to ``range-angle`` when ``range_m`` and ENU
position columns are present.  Radar association now consumes those covariance
columns natively; the runtime hook only annotates radar rows at ingestion points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_covariance import (
    RadarCovarianceConfig,
    append_radar_covariance_columns,
    candidate_radar_covariances,
    fixed_radar_covariance,
    row_radar_covariance,
)

_INSTALLED = False
_ORIGINAL_EVENTS: Any = None
_ORIGINAL_RADAR_MEASUREMENTS_TO_ENU: Any = None


def install() -> None:
    """Install candidate-specific radar covariance support once."""

    global _INSTALLED, _ORIGINAL_EVENTS, _ORIGINAL_RADAR_MEASUREMENTS_TO_ENU
    if _INSTALLED:
        return

    from raft_uav.baselines import radar_association as association
    from raft_uav.io import aerpaw

    _ORIGINAL_EVENTS = association._events
    _ORIGINAL_RADAR_MEASUREMENTS_TO_ENU = aerpaw.radar_measurements_to_enu

    association._events = _events_with_covariance
    aerpaw.radar_measurements_to_enu = _radar_measurements_to_enu_with_candidate_covariance

    _INSTALLED = True


def _events_with_covariance(
    rf_measurements: list[Any], radar: pd.DataFrame
) -> list[dict[str, object]]:
    assert _ORIGINAL_EVENTS is not None
    return _ORIGINAL_EVENTS(rf_measurements, _annotate_radar(radar))


def _annotate_radar(radar: pd.DataFrame) -> pd.DataFrame:
    return append_radar_covariance_columns(radar, RadarCovarianceConfig.from_environment())


def _nis_scored_candidates_with_candidate_covariance(
    candidates: pd.DataFrame,
    tracker: Any,
    covariance: np.ndarray,
) -> pd.DataFrame:
    from raft_uav.baselines.kalman import measurement_matrix

    if candidates.empty:
        return candidates.iloc[0:0].copy()
    observation = measurement_matrix(3)
    state_position = observation @ tracker.state
    predicted_covariance = observation @ tracker.covariance_matrix @ observation.T
    measurement_covariances = candidate_radar_covariances(candidates, covariance)
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = vectors - state_position

    nis = np.empty(len(candidates), dtype=float)
    for index, residual in enumerate(residuals):
        innovation_covariance = predicted_covariance + measurement_covariances[index]
        try:
            precision = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(innovation_covariance)
        nis[index] = float(residual @ precision @ residual)

    scored = candidates.copy()
    scored["association_nis"] = nis
    scored["association_candidate_rows"] = int(len(candidates))
    scored["association_cov_trace_m2"] = np.trace(measurement_covariances, axis1=1, axis2=2)
    return scored


def _pda_mixture_candidate_with_candidate_covariance(
    candidates: pd.DataFrame,
    *,
    base_covariance: np.ndarray,
    nis_temperature: float,
    catprob_exponent: float,
) -> pd.Series:
    from raft_uav.baselines import radar_association as association

    weights = association._pda_weights(
        candidates,
        nis_temperature=nis_temperature,
        catprob_exponent=catprob_exponent,
    )
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    mean = weights @ vectors
    residuals = vectors - mean
    spread = residuals.T @ (residuals * weights[:, None])
    measurement_covariances = candidate_radar_covariances(candidates, base_covariance)
    mean_measurement_covariance = np.einsum("i,ijk->jk", weights, measurement_covariances)
    covariance = mean_measurement_covariance + spread

    best_index = int(np.argmax(weights))
    selected = candidates.iloc[best_index].copy()
    selected["east_m"] = float(mean[0])
    selected["north_m"] = float(mean[1])
    selected["up_m"] = float(mean[2])
    selected["association_mode"] = "pda-mixture"
    selected["association_action"] = "pda_mixture"
    selected["association_candidate_rows"] = int(len(candidates))
    selected["association_nis"] = float(
        weights @ candidates["association_nis"].to_numpy(dtype=float)
    )
    selected["association_score"] = float(-np.log(float(np.max(weights))))
    selected["association_weight_max"] = float(np.max(weights))
    selected["association_weight_entropy"] = association._weight_entropy(weights)
    selected["association_effective_candidates"] = float(1.0 / np.sum(weights**2))
    selected["association_best_track_id"] = association._optional_track_id(selected)
    selected["association_position_spread_trace_m2"] = float(np.trace(spread))
    selected["association_cov_ee"] = float(covariance[0, 0])
    selected["association_cov_nn"] = float(covariance[1, 1])
    selected["association_cov_uu"] = float(covariance[2, 2])
    selected["association_cov_en"] = float(covariance[0, 1])
    selected["association_cov_eu"] = float(covariance[0, 2])
    selected["association_cov_nu"] = float(covariance[1, 2])
    return selected


def _radar_measurements_to_enu_with_candidate_covariance(
    radar: pd.DataFrame,
    projector: Any | None = None,
    truth_origin_time: Any | None = None,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
    include_velocity: bool = False,
    clock_offset_s: float = -4.0 * 60.0 * 60.0,
) -> list[Any]:
    from raft_uav.baselines.kalman import TrackingMeasurement
    from raft_uav.io import aerpaw

    frame = radar
    if "east_m" not in frame.columns:
        if projector is None or truth_origin_time is None:
            raise ValueError("raw radar rows require projector and truth_origin_time")
        frame = aerpaw.normalize_radar(
            frame,
            projector,
            truth_origin_time,
            clock_offset_s=clock_offset_s,
        )

    fallback_position_covariance = fixed_radar_covariance(default_xy_std_m, default_z_std_m)
    measurements: list[TrackingMeasurement] = []
    for _, row in frame.iterrows():
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
        position_covariance = row_radar_covariance(row, fallback_position_covariance)
        velocity = aerpaw._radar_velocity_vector_enu(row) if include_velocity else None
        if velocity is None:
            vector = position
            covariance = position_covariance
        else:
            vector = np.concatenate([position, velocity])
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:3, :3] = position_covariance
            covariance[3:, 3:] = np.diag([default_velocity_std_mps**2] * 3)
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements
