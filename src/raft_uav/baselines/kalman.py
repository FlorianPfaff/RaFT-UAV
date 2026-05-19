"""Asynchronous constant-velocity Kalman fusion baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
from pyrecest.filters import KalmanFilter
from scipy.stats import chi2

from raft_uav.baselines.update_logic import (
    gate_threshold_for_measurement,
    huber_covariance_scale,
    huber_threshold_for_measurement,
    inflation_alpha_for_measurement,
    max_residual_norm_for_measurement,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_for_measurement,
    student_t_covariance_scale,
    student_t_dof_for_measurement,
    symmetrized,
)
from raft_uav.calibration.nis_covariance import scale_covariance_for_calibrated_source

__all__ = [
    "AsyncConstantVelocityKalmanTracker",
    "TrackingMeasurement",
    "TrackingUpdateDiagnostics",
    "constant_velocity_matrix",
    "gate_threshold_from_probability",
    "huber_covariance_scale",
    "measurement_matrix",
    "normalized_innovation_squared",
    "run_async_cv_baseline",
    "student_t_covariance_scale",
    "white_acceleration_process_noise",
]


@dataclass(frozen=True)
class TrackingMeasurement:
    """Linear position or position-plus-velocity measurement in local ENU coordinates."""

    time_s: float
    vector: np.ndarray
    covariance: np.ndarray
    source: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        source = str(self.source)
        if vector.size not in (2, 3, 6):
            raise ValueError("measurement vector must have 2, 3, or 6 elements")
        if covariance.shape != (vector.size, vector.size):
            raise ValueError("measurement covariance must match vector dimension")
        if not np.isfinite(vector).all():
            raise ValueError("measurement vector must be finite")
        if not np.isfinite(covariance).all():
            raise ValueError("measurement covariance must be finite")
        covariance = scale_covariance_for_calibrated_source(source, vector.size, covariance)
        if covariance.shape != (vector.size, vector.size):
            raise ValueError("calibrated measurement covariance must match vector dimension")
        if not np.isfinite(covariance).all():
            raise ValueError("calibrated measurement covariance must be finite")
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source", source)


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
    safety_gate_threshold: float | None
    residual_gate_threshold_m: float | None
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
    """Return discretized continuous white-acceleration noise for 3D CV.

    ``acceleration_std`` is the square-root spectral density used by the
    continuous-time white-acceleration model. For one position/velocity axis,
    the discretized covariance is ``q * [[dt**3 / 3, dt**2 / 2], [dt**2 / 2,
    dt]]`` with ``q = acceleration_std**2``.
    """

    dt = float(dt_s)
    q = float(acceleration_std) ** 2
    block = q * np.array(
        [
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ]
    )
    covariance = np.zeros((6, 6))
    for pos_idx, vel_idx in ((0, 3), (1, 4), (2, 5)):
        covariance[np.ix_([pos_idx, vel_idx], [pos_idx, vel_idx])] = block
    return covariance


def measurement_matrix(measurement_dim: int) -> np.ndarray:
    """Return the matrix mapping the 6D state to a supported observation vector."""

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
    if measurement_dim == 6:
        return np.eye(6)
    raise ValueError("measurement_dim must be 2, 3, or 6")


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
    if measurement_dim not in (2, 3, 6):
        raise ValueError("measurement_dim must be 2, 3, or 6")
    return float(chi2.ppf(probability, df=measurement_dim))


class AsyncConstantVelocityKalmanTracker:
    """Asynchronous constant-velocity Kalman tracker with optional NIS gating.

    PyRecEst owns the linear Gaussian prediction, measurement update,
    robust-update policy, and update diagnostics. RaFT-UAV keeps only the
    application-specific constant-velocity model, source-specific options, and
    CSV/plot-facing diagnostics schema.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_time_s: float,
        initial_position_std_m: float = 50.0,
        initial_velocity_std_mps: float = 15.0,
        acceleration_std_mps2: float = 4.0,
    ) -> None:
        initial = np.asarray(initial_position, dtype=float).reshape(-1)
        initial_velocity: np.ndarray | None = None
        if initial.size == 2:
            position = np.array([initial[0], initial[1], 0.0])
        elif initial.size == 3:
            position = initial
        elif initial.size == 6:
            position = initial[:3]
            initial_velocity = initial[3:6]
        else:
            raise ValueError("initial_position must contain 2, 3, or 6 elements")

        self.mean = np.zeros(6)
        self.mean[:3] = position
        if initial_velocity is not None:
            self.mean[3:6] = initial_velocity
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
        self.filter = KalmanFilter((self.mean.copy(), self.covariance.copy()))
        self.current_time_s = float(initial_time_s)
        self.acceleration_std_mps2 = float(acceleration_std_mps2)
        self._sync_from_filter()

    @property
    def state(self) -> np.ndarray:
        """Return the current posterior mean."""

        return self.mean.copy()

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the current posterior covariance."""

        return self.covariance.copy()

    def _sync_from_filter(self) -> None:
        """Mirror the PyRecEst filter state into NumPy arrays used downstream."""

        self.mean = np.asarray(self.filter.get_point_estimate(), dtype=float).reshape(6)
        self.covariance = symmetrized(
            np.asarray(self.filter.filter_state.C, dtype=float).reshape(6, 6)
        )

    def predict_to(self, time_s: float) -> None:
        """Predict to an absolute timestamp."""

        dt_s = float(time_s) - self.current_time_s
        if dt_s < -1e-9:
            raise ValueError("measurements must be processed in chronological order")
        if dt_s > 0.0:
            transition = constant_velocity_matrix(dt_s)
            process_noise = white_acceleration_process_noise(dt_s, self.acceleration_std_mps2)
            self.filter.predict_linear(transition, process_noise)
            self._sync_from_filter()
            self.current_time_s = float(time_s)

    def update(
        self,
        measurement: TrackingMeasurement,
        gate_threshold: float | None = None,
        safety_gate_threshold: float | None = None,
        max_residual_norm: float | None = None,
        robust_update: str | None = None,
        inflation_alpha: float = 1.0,
        student_t_dof: float = 4.0,
        huber_threshold: float = 2.0,
    ) -> TrackingUpdateDiagnostics:
        """Predict to and conditionally update from one RF or radar measurement."""

        self.predict_to(measurement.time_s)
        observation = measurement_matrix(measurement.vector.size)
        plan = plan_linear_measurement_update(
            mean=self.mean,
            covariance_matrix=self.covariance,
            measurement_vector=measurement.vector,
            measurement_covariance=measurement.covariance,
            observation_matrix=observation,
            robust_update=robust_update,
            gate_threshold=gate_threshold,
            safety_gate_threshold=safety_gate_threshold,
            max_residual_norm=max_residual_norm,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
            inflation_alpha=inflation_alpha,
        )

        if plan.accepted:
            self.filter.update_linear(plan.vector, plan.observation, plan.covariance)
            self._sync_from_filter()

        return TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=measurement.vector.size,
            accepted=plan.accepted,
            update_action=plan.update_action,
            nis=plan.nis,
            gate_threshold=plan.threshold,
            safety_gate_threshold=plan.safety_threshold,
            residual_gate_threshold_m=plan.residual_threshold,
            covariance_scale=plan.covariance_scale,
            inflation_alpha=float(inflation_alpha) if robust_update == "nis-inflate" else None,
            residual_norm_m=plan.residual_norm,
        )


def run_async_cv_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    student_t_dof_by_source: Mapping[str, float] | None = None,
    huber_threshold_by_source: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Run the asynchronous CV Kalman baseline and return posterior records."""

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
        diagnostics = tracker.update(
            measurement,
            gate_threshold=gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
                probability_to_threshold=gate_threshold_from_probability,
            ),
            safety_gate_threshold=gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=safety_gate_probabilities_by_source,
                gate_thresholds_by_source=safety_gate_thresholds_by_source,
                probability_to_threshold=gate_threshold_from_probability,
            ),
            max_residual_norm=max_residual_norm_for_measurement(
                measurement,
                max_residual_norms_by_source=max_residual_norms_by_source,
            ),
            robust_update=robust_update_for_measurement(
                measurement,
                robust_update_by_source=robust_update_by_source,
            ),
            inflation_alpha=inflation_alpha_for_measurement(
                measurement,
                inflation_alpha_by_source=inflation_alpha_by_source,
            ),
            student_t_dof=student_t_dof_for_measurement(
                measurement,
                student_t_dof_by_source=student_t_dof_by_source,
            ),
            huber_threshold=huber_threshold_for_measurement(
                measurement,
                huber_threshold_by_source=huber_threshold_by_source,
            ),
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


_symmetrized = symmetrized
