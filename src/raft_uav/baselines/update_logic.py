"""Shared linear-update gating utilities for RaFT-UAV baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pyrecest.filters._linear_gaussian import (
    huber_covariance_scale as _pyrecest_huber_covariance_scale,
    normalized_innovation_squared as _pyrecest_normalized_innovation_squared,
    student_t_covariance_scale as _pyrecest_student_t_covariance_scale,
)

ROBUST_UPDATE_MODES = ("nis-inflate", "student-t", "huber")
DEFAULT_STUDENT_T_DOF = 4.0
DEFAULT_HUBER_THRESHOLD = 2.0


class TrackingMeasurementLike(Protocol):
    """Protocol for measurement objects used by baseline trackers."""

    source: str
    vector: np.ndarray


@dataclass(frozen=True)
class LinearUpdatePlan:
    """Precomputed gating and covariance-inflation decision for one update."""

    vector: np.ndarray
    covariance: np.ndarray
    observation: np.ndarray
    residual: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    threshold: float | None
    covariance_scale: float
    update_action: str
    accepted: bool
    inflation_alpha: float


def _backend_scalar_to_float(value) -> float:
    """Convert a backend scalar to a Python float."""

    return float(np.asarray(value, dtype=float))


def normalized_innovation_squared(
    residual: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    """Return the squared Mahalanobis innovation distance via PyRecEst."""

    return _backend_scalar_to_float(
        _pyrecest_normalized_innovation_squared(residual, innovation_covariance)
    )


def student_t_covariance_scale(
    nis: float,
    measurement_dim: int,
    degrees_of_freedom: float = DEFAULT_STUDENT_T_DOF,
) -> float:
    """Return the PyRecEst Student-t robust covariance inflation factor."""

    return _backend_scalar_to_float(
        _pyrecest_student_t_covariance_scale(
            nis,
            measurement_dim,
            dof=degrees_of_freedom,
        )
    )


def huber_covariance_scale(
    nis: float,
    threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> float:
    """Return the PyRecEst multivariate Huber covariance inflation factor."""

    return _backend_scalar_to_float(
        _pyrecest_huber_covariance_scale(nis, huber_threshold=threshold)
    )


def robust_update_covariance_scale(
    robust_update: str | None,
    *,
    nis: float,
    measurement_dim: int,
    gate_threshold: float | None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> tuple[float, str | None]:
    """Return covariance scale and diagnostic action for a robust update mode."""

    if robust_update is None:
        return 1.0, None
    if robust_update == "nis-inflate":
        if gate_threshold is None or float(nis) <= float(gate_threshold):
            return 1.0, None
        scale = max(1.0, float((float(nis) / float(gate_threshold)) ** inflation_alpha))
        return scale, "inflated"
    if robust_update == "student-t":
        scale = student_t_covariance_scale(nis, measurement_dim, student_t_dof)
        return scale, "student_t" if scale > 1.0 else None
    if robust_update == "huber":
        scale = huber_covariance_scale(nis, huber_threshold)
        return scale, "huber" if scale > 1.0 else None
    raise ValueError(f"unknown robust update mode {robust_update!r}")


def plan_linear_measurement_update(
    *,
    mean: np.ndarray,
    covariance_matrix: np.ndarray,
    measurement_vector: np.ndarray,
    measurement_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    gate_threshold: float | None = None,
    robust_update: str | None = None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> LinearUpdatePlan:
    """Prepare shared NIS gating/inflation quantities for a linear update."""

    alpha = float(inflation_alpha)
    if alpha <= 0.0:
        raise ValueError("inflation_alpha must be positive")

    vector = np.asarray(measurement_vector, dtype=float).reshape(-1)
    covariance = np.asarray(measurement_covariance, dtype=float)
    observation = np.asarray(observation_matrix, dtype=float)
    posterior_mean = np.asarray(mean, dtype=float)
    posterior_covariance = np.asarray(covariance_matrix, dtype=float)

    residual = vector - observation @ posterior_mean
    innovation_covariance = observation @ posterior_covariance @ observation.T + covariance
    nis = normalized_innovation_squared(residual, innovation_covariance)
    threshold = None if gate_threshold is None else float(gate_threshold)
    covariance_scale = 1.0
    update_action = "updated"
    accepted = True

    if threshold is not None and nis > threshold and robust_update is None:
        accepted = False
        update_action = "rejected"
    else:
        covariance_scale, robust_action = robust_update_covariance_scale(
            robust_update,
            nis=nis,
            measurement_dim=vector.size,
            gate_threshold=threshold,
            inflation_alpha=alpha,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
        )
        if covariance_scale > 1.0:
            covariance = covariance * covariance_scale
            innovation_covariance = observation @ posterior_covariance @ observation.T + covariance
        if robust_action is not None:
            update_action = robust_action

    return LinearUpdatePlan(
        vector=vector,
        covariance=covariance,
        observation=observation,
        residual=residual,
        innovation_covariance=innovation_covariance,
        nis=float(nis),
        threshold=threshold,
        covariance_scale=float(covariance_scale),
        update_action=update_action,
        accepted=bool(accepted),
        inflation_alpha=alpha,
    )


def gate_threshold_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    probability_to_threshold,
) -> float | None:
    """Resolve a source-specific NIS threshold for one measurement."""

    if gate_thresholds_by_source and measurement.source in gate_thresholds_by_source:
        threshold = gate_thresholds_by_source[measurement.source]
        return None if threshold is None else float(threshold)
    if gate_probabilities_by_source and measurement.source in gate_probabilities_by_source:
        return probability_to_threshold(
            gate_probabilities_by_source[measurement.source],
            measurement.vector.size,
        )
    return None


def robust_update_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    robust_update_by_source: Mapping[str, str | None] | None,
) -> str | None:
    """Resolve a source-specific robust update mode for one measurement."""

    if robust_update_by_source and measurement.source in robust_update_by_source:
        return robust_update_by_source[measurement.source]
    return None


def inflation_alpha_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    inflation_alpha_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific NIS-inflation exponent for one measurement."""

    if inflation_alpha_by_source and measurement.source in inflation_alpha_by_source:
        return float(inflation_alpha_by_source[measurement.source])
    return 1.0


def student_t_dof_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    student_t_dof_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific Student-t degrees-of-freedom value."""

    if student_t_dof_by_source and measurement.source in student_t_dof_by_source:
        return float(student_t_dof_by_source[measurement.source])
    return DEFAULT_STUDENT_T_DOF


def huber_threshold_for_measurement(
    measurement: TrackingMeasurementLike,
    *,
    huber_threshold_by_source: Mapping[str, float] | None,
) -> float:
    """Resolve a source-specific Huber innovation-radius threshold."""

    if huber_threshold_by_source and measurement.source in huber_threshold_by_source:
        return float(huber_threshold_by_source[measurement.source])
    return DEFAULT_HUBER_THRESHOLD


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)
