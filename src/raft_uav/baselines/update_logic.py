"""Shared linear-update gating utilities for RaFT-UAV baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


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


def normalized_innovation_squared(residual: np.ndarray, innovation_covariance: np.ndarray) -> float:
    """Return the squared Mahalanobis innovation distance."""

    residual = np.asarray(residual, dtype=float).reshape(-1)
    covariance = np.asarray(innovation_covariance, dtype=float)
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    return float(residual @ solved)


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

    if threshold is not None and nis > threshold:
        if robust_update == "nis-inflate":
            covariance_scale = max(1.0, float((nis / threshold) ** alpha))
            covariance = covariance * covariance_scale
            innovation_covariance = observation @ posterior_covariance @ observation.T + covariance
            update_action = "inflated"
        elif robust_update is None:
            accepted = False
            update_action = "rejected"
        else:
            raise ValueError(f"unknown robust update mode {robust_update!r}")

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


def symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (matrix + matrix.T)
