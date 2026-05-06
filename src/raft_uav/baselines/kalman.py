"""Asynchronous constant-velocity Kalman fusion baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import chi2

try:  # Keep the reproduced baseline on PyRecEst when the dependency is installed.
    from pyrecest.filters import KalmanFilter
except ImportError:  # pragma: no cover - exercised only in lightweight local smoke tests.
    KalmanFilter = None


@dataclass(frozen=True)
class TrackingMeasurement:
    """Position-like measurement in a local East-North-Up frame."""

    time_s: float
    vector: np.ndarray
    covariance: np.ndarray
    source: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        if vector.size not in (2, 3):
            raise ValueError("measurement vector must have 2 or 3 elements")
        if covariance.shape != (vector.size, vector.size):
            raise ValueError("measurement covariance must match vector dimension")
        if not np.isfinite(vector).all():
            raise ValueError("measurement vector must be finite")
        if not np.isfinite(covariance).all():
            raise ValueError("measurement covariance must be finite")
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True)
class TrackingUpdateDiagnostics:
    """Innovation and gating diagnostics for one measurement update."""

    time_s: float
    source: str
    measurement_dim: int
    accepted: bool
    update_action: str
    nis: float
    gate_threshold: float | None
    covariance_scale: float
    inflation_alpha: float | None
    residual_norm_m: float

    def to_record(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly representation."""

        return asdict(self)


def constant_velocity_matrix(dt_s: float) -> np.ndarray:
    """Return the 6D constant-velocity transition matrix for ENU position/velocity."""

    dt = float(dt_s)
    matrix = np.eye(6)
    matrix[0, 3] = dt
    matrix[1, 4] = dt
    matrix[2, 5] = dt
    return matrix


def white_acceleration_process_noise(dt_s: float, acceleration_std: float) -> np.ndarray:
    """Return continuous white-acceleration process noise discretized for 3D CV."""

    dt = float(dt_s)
    q = float(acceleration_std) ** 2
    block = q * np.array(
        [
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ]
    )
    covariance = np.zeros((6, 6))
    for pos_idx, vel_idx in ((0, 3), (1, 4), (2, 5)):
        covariance[np.ix_([pos_idx, vel_idx], [pos_idx, vel_idx])] = block
    return covariance


def measurement_matrix(measurement_dim: int) -> np.ndarray:
    """Return a matrix mapping the 6D state to 2D or 3D position observations."""

    if measurement_dim == 2:
        matrix = np.zeros((2, 6))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        return matrix
    if measurement_dim == 3:
        matrix = np.zeros((3, 6))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        matrix[2, 2] = 1.0
        return matrix
    raise ValueError("measurement_dim must be 2 or 3")


def normalized_innovation_squared(residual: np.ndarray, innovation_covariance: np.ndarray) -> float:
    """Return the squared Mahalanobis innovation distance."""

    residual = np.asarray(residual, dtype=float).reshape(-1)
    covariance = np.asarray(innovation_covariance, dtype=float)
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    return float(residual @ solved)


def gate_threshold_from_probability(
    probability: float | None, measurement_dim: int
) -> float | None:
    """Convert a chi-square gate probability to an NIS threshold.

    ``probability=None`` disables gating for that source. For example, a 0.99
    gate gives thresholds of about 9.21 in 2D and 11.34 in 3D.
    """

    if probability is None:
        return None
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("gate probability must be in (0, 1), or None to disable gating")
    if measurement_dim not in (2, 3):
        raise ValueError("measurement_dim must be 2 or 3")
    return float(chi2.ppf(probability, df=measurement_dim))


class AsyncConstantVelocityKalmanTracker:
    """Asynchronous constant-velocity Kalman tracker with optional NIS gating.

    PyRecEst is still used for the posterior mean when available. The covariance
    is also maintained explicitly so that innovation covariance, NIS values, and
    rejected-update behavior are available for diagnostics.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_time_s: float,
        initial_position_std_m: float = 50.0,
        initial_velocity_std_mps: float = 15.0,
        acceleration_std_mps2: float = 4.0,
    ) -> None:
        position = np.asarray(initial_position, dtype=float).reshape(-1)
        if position.size == 2:
            position = np.array([position[0], position[1], 0.0])
        if position.size != 3:
            raise ValueError("initial_position must contain 2 or 3 elements")

        self.mean = np.zeros(6)
        self.mean[:3] = position
        self.covariance = np.diag(
            [
                initial_position_std_m**2,
                initial_position_std_m**2,
                initial_position_std_m**2,
                initial_velocity_std_mps**2,
                initial_velocity_std_mps**2,
                initial_velocity_std_mps**2,
            ]
        )
        self.filter: Any | None = None
        if KalmanFilter is not None:
            self.filter = KalmanFilter((self.mean.copy(), self.covariance.copy()))
        self.current_time_s = float(initial_time_s)
        self.acceleration_std_mps2 = float(acceleration_std_mps2)

    @property
    def state(self) -> np.ndarray:
        """Return the current posterior mean."""

        return self.mean.copy()

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the current posterior covariance."""

        return self.covariance.copy()

    def predict_to(self, time_s: float) -> None:
        """Predict to an absolute timestamp."""

        dt_s = float(time_s) - self.current_time_s
        if dt_s < -1e-9:
            raise ValueError("measurements must be processed in chronological order")
        if dt_s > 0.0:
            transition = constant_velocity_matrix(dt_s)
            process_noise = white_acceleration_process_noise(dt_s, self.acceleration_std_mps2)
            if self.filter is not None:
                self.filter.predict_linear(transition, process_noise)
                self.mean = np.asarray(self.filter.get_point_estimate(), dtype=float)
            else:
                self.mean = transition @ self.mean
            self.covariance = transition @ self.covariance @ transition.T + process_noise
            self.covariance = _symmetrized(self.covariance)
            self.current_time_s = float(time_s)

    def update(
        self,
        measurement: TrackingMeasurement,
        gate_threshold: float | None = None,
        robust_update: str | None = None,
        inflation_alpha: float = 1.0,
    ) -> TrackingUpdateDiagnostics:
        """Predict to and conditionally update from one RF or radar measurement.

        If ``gate_threshold`` is provided and ``robust_update`` is ``None``, the
        update is skipped when the normalized innovation squared exceeds that
        threshold. If ``robust_update="nis-inflate"``, the update is kept but
        its measurement covariance is inflated by ``(nis / gate_threshold) **
        inflation_alpha`` when the threshold is exceeded.
        """

        self.predict_to(measurement.time_s)
        inflation_alpha = float(inflation_alpha)
        if inflation_alpha <= 0.0:
            raise ValueError("inflation_alpha must be positive")
        vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
        covariance = np.asarray(measurement.covariance, dtype=float)
        observation = measurement_matrix(vector.size)

        residual = vector - observation @ self.mean
        innovation_covariance = observation @ self.covariance @ observation.T + covariance
        nis = normalized_innovation_squared(residual, innovation_covariance)
        threshold = None if gate_threshold is None else float(gate_threshold)
        covariance_scale = 1.0
        update_action = "updated"
        accepted = True

        if threshold is not None and nis > threshold:
            if robust_update == "nis-inflate":
                covariance_scale = max(1.0, float((nis / threshold) ** inflation_alpha))
                covariance = covariance * covariance_scale
                innovation_covariance = observation @ self.covariance @ observation.T + covariance
                update_action = "inflated"
            elif robust_update is None:
                accepted = False
                update_action = "rejected"
            else:
                raise ValueError(f"unknown robust update mode {robust_update!r}")

        if accepted:
            self._linear_update(observation, residual, innovation_covariance, covariance, vector)

        return TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=vector.size,
            accepted=bool(accepted),
            update_action=update_action,
            nis=float(nis),
            gate_threshold=threshold,
            covariance_scale=float(covariance_scale),
            inflation_alpha=inflation_alpha if robust_update == "nis-inflate" else None,
            residual_norm_m=float(np.linalg.norm(residual)),
        )

    def _linear_update(
        self,
        observation: np.ndarray,
        residual: np.ndarray,
        innovation_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        vector: np.ndarray,
    ) -> None:
        """Apply a numerically stable Joseph-form linear Kalman update."""

        gain = np.linalg.solve(
            innovation_covariance.T,
            (self.covariance @ observation.T).T,
        ).T
        updated_mean = self.mean + gain @ residual
        identity = np.eye(self.covariance.shape[0])
        update_matrix = identity - gain @ observation
        updated_covariance = (
            update_matrix @ self.covariance @ update_matrix.T
            + gain @ measurement_covariance @ gain.T
        )

        if self.filter is not None:
            self.filter.update_linear(vector, observation, measurement_covariance)
            self.mean = np.asarray(self.filter.get_point_estimate(), dtype=float)
        else:
            self.mean = updated_mean
        self.covariance = _symmetrized(updated_covariance)


def run_async_cv_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Run the asynchronous CV Kalman baseline and return posterior records.

    ``gate_probabilities_by_source`` maps source names such as ``"rf"`` or
    ``"radar"`` to chi-square gate probabilities. ``None`` or a missing source
    disables NIS thresholding. ``gate_thresholds_by_source`` can be used for
    deterministic tests or manually tuned thresholds and takes precedence over
    probabilities. ``robust_update_by_source={"rf": "nis-inflate"}`` keeps
    high-NIS updates and inflates their measurement covariance instead of
    rejecting them. ``inflation_alpha_by_source`` controls the source-specific
    exponent used by ``nis-inflate``.
    """

    ordered = sorted(measurements, key=lambda item: item.time_s)
    if not ordered:
        return []

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=ordered[0].vector,
        initial_time_s=ordered[0].time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )

    records: list[dict[str, object]] = []
    for measurement in ordered:
        gate_threshold = _gate_threshold_for_measurement(
            measurement,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
        )
        robust_update = _robust_update_for_measurement(
            measurement,
            robust_update_by_source=robust_update_by_source,
        )
        inflation_alpha = _inflation_alpha_for_measurement(
            measurement,
            inflation_alpha_by_source=inflation_alpha_by_source,
        )
        diagnostics = tracker.update(
            measurement,
            gate_threshold=gate_threshold,
            robust_update=robust_update,
            inflation_alpha=inflation_alpha,
        )
        records.append(
            {
                "time_s": measurement.time_s,
                "source": measurement.source,
                "state": tracker.state.copy(),
                "covariance": tracker.covariance_matrix.copy(),
                **diagnostics.to_record(),
            }
        )
    return records


def _gate_threshold_for_measurement(
    measurement: TrackingMeasurement,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
) -> float | None:
    if gate_thresholds_by_source and measurement.source in gate_thresholds_by_source:
        threshold = gate_thresholds_by_source[measurement.source]
        return None if threshold is None else float(threshold)
    if gate_probabilities_by_source and measurement.source in gate_probabilities_by_source:
        return gate_threshold_from_probability(
            gate_probabilities_by_source[measurement.source],
            measurement.vector.size,
        )
    return None


def _robust_update_for_measurement(
    measurement: TrackingMeasurement,
    *,
    robust_update_by_source: Mapping[str, str | None] | None,
) -> str | None:
    if robust_update_by_source and measurement.source in robust_update_by_source:
        return robust_update_by_source[measurement.source]
    return None


def _inflation_alpha_for_measurement(
    measurement: TrackingMeasurement,
    *,
    inflation_alpha_by_source: Mapping[str, float] | None,
) -> float:
    if inflation_alpha_by_source and measurement.source in inflation_alpha_by_source:
        return float(inflation_alpha_by_source[measurement.source])
    return 1.0


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
