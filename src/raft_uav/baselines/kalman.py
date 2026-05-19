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
    robust_update_covariance_scale,
    student_t_covariance_scale,
    student_t_dof_for_measurement,
    symmetrized,
)

__all__ = [
    "AsyncConstantVelocityKalmanTracker",
    "RadarPolarMeasurement",
    "bootstrap_tracking_diagnostics",
    "TrackingMeasurement",
    "TrackingUpdateDiagnostics",
    "constant_velocity_matrix",
    "enu_position_to_radar_polar",
    "gate_threshold_from_probability",
    "huber_covariance_scale",
    "measurement_matrix",
    "radar_polar_observation_and_jacobian",
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
        if vector.size not in (2, 3, 6):
            raise ValueError("measurement vector must have 2, 3, or 6 elements")
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
class RadarPolarMeasurement:
    """Native radar measurement in range/azimuth/elevation coordinates.

    The coordinate convention is ENU with azimuth measured by ``atan2(east,
    north)`` and elevation measured by ``atan2(up, horizontal_range)``.  A
    three-element vector contains ``[range_m, azimuth_rad, elevation_rad]``.
    A four-element vector additionally contains radial velocity in m/s.
    """

    time_s: float
    vector: np.ndarray
    covariance: np.ndarray
    source: str = "radar"
    origin_enu_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        origin = np.asarray(self.origin_enu_m, dtype=float).reshape(-1)
        if vector.size not in (3, 4):
            raise ValueError("radar polar vector must have 3 or 4 elements")
        if covariance.shape != (vector.size, vector.size):
            raise ValueError("radar polar covariance must match vector dimension")
        if origin.size != 3 or not np.isfinite(origin).all():
            raise ValueError("radar polar origin must contain three finite ENU values")
        if not np.isfinite(vector).all() or not np.isfinite(covariance).all():
            raise ValueError("radar polar measurement and covariance must be finite")
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "origin_enu_m", tuple(float(value) for value in origin))

    @property
    def origin_vector(self) -> np.ndarray:
        """Return the radar origin as an ENU vector."""

        return np.asarray(self.origin_enu_m, dtype=float)


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


def bootstrap_tracking_diagnostics(measurement: TrackingMeasurement) -> TrackingUpdateDiagnostics:
    """Return diagnostics for a measurement used only as the initial prior.

    Baseline runners initialize the tracker state from the first available
    measurement.  That bootstrap measurement must not also be replayed as a
    Kalman update at the same timestamp, otherwise the first observation is
    counted twice and the initial posterior covariance is spuriously tightened.
    """

    return TrackingUpdateDiagnostics(
        time_s=float(measurement.time_s),
        source=measurement.source,
        measurement_dim=measurement.vector.size,
        accepted=True,
        update_action="bootstrap",
        nis=float("nan"),
        gate_threshold=None,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )


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


def enu_position_to_radar_polar(
    position_enu_m: np.ndarray,
    origin_enu_m: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_enu_mps: np.ndarray | None = None,
) -> np.ndarray:
    """Convert an ENU position, and optionally velocity, to radar polar coordinates."""

    state = np.zeros(6, dtype=float)
    state[:3] = np.asarray(position_enu_m, dtype=float).reshape(3)
    include_range_rate = velocity_enu_mps is not None
    if velocity_enu_mps is not None:
        state[3:6] = np.asarray(velocity_enu_mps, dtype=float).reshape(3)
    vector, _ = radar_polar_observation_and_jacobian(
        state,
        origin_enu_m,
        include_range_rate=include_range_rate,
    )
    return vector


def radar_polar_observation_and_jacobian(
    state: np.ndarray,
    origin_enu_m: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    include_range_rate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native radar-polar observation and EKF Jacobian for a CV state."""

    state_vector = np.asarray(state, dtype=float).reshape(6)
    origin = np.asarray(origin_enu_m, dtype=float).reshape(3)
    position = state_vector[:3] - origin
    velocity = state_vector[3:6]
    east, north, up = position

    horizontal2 = float(east * east + north * north)
    horizontal = float(np.sqrt(max(horizontal2, 1.0e-18)))
    range2 = float(horizontal2 + up * up)
    range_m = float(np.sqrt(max(range2, 1.0e-18)))

    azimuth_rad = float(np.arctan2(east, north))
    elevation_rad = float(np.arctan2(up, horizontal))
    dim = 4 if include_range_rate else 3
    observation = np.zeros((dim, 6), dtype=float)
    predicted = np.zeros(dim, dtype=float)
    predicted[:3] = [range_m, azimuth_rad, elevation_rad]

    observation[0, :3] = position / range_m
    observation[1, 0] = north / max(horizontal2, 1.0e-18)
    observation[1, 1] = -east / max(horizontal2, 1.0e-18)
    observation[2, 0] = -up * east / (range_m * range_m * horizontal)
    observation[2, 1] = -up * north / (range_m * range_m * horizontal)
    observation[2, 2] = horizontal / (range_m * range_m)

    if include_range_rate:
        radial_velocity_mps = float(position @ velocity / range_m)
        predicted[3] = radial_velocity_mps
        observation[3, :3] = velocity / range_m - position * radial_velocity_mps / (
            range_m * range_m
        )
        observation[3, 3:6] = position / range_m

    return predicted, observation


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angular residual into [-pi, pi)."""

    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)


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
    if measurement_dim not in (2, 3, 4, 6):
        raise ValueError("measurement_dim must be 2, 3, 4, or 6")
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

    def _update_polar_radar(
        self,
        measurement: RadarPolarMeasurement,
        *,
        gate_threshold: float | None,
        safety_gate_threshold: float | None,
        max_residual_norm: float | None,
        robust_update: str | None,
        inflation_alpha: float,
        student_t_dof: float,
        huber_threshold: float,
    ) -> TrackingUpdateDiagnostics:
        """Condition the CV state on a nonlinear native radar-polar measurement."""

        alpha = float(inflation_alpha)
        if alpha <= 0.0:
            raise ValueError("inflation_alpha must be positive")
        vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
        covariance = np.asarray(measurement.covariance, dtype=float).copy()
        predicted, observation = radar_polar_observation_and_jacobian(
            self.mean,
            measurement.origin_vector,
            include_range_rate=vector.size == 4,
        )
        residual = vector - predicted
        residual[1] = _wrap_angle(residual[1])
        residual[2] = _wrap_angle(residual[2])
        innovation_covariance = observation @ self.covariance @ observation.T + covariance
        nis = normalized_innovation_squared(residual, innovation_covariance)
        residual_norm = float(np.linalg.norm(residual))
        threshold = None if gate_threshold is None else float(gate_threshold)
        safety_threshold = None if safety_gate_threshold is None else float(safety_gate_threshold)
        residual_threshold = None if max_residual_norm is None else float(max_residual_norm)
        if residual_threshold is not None and residual_threshold <= 0.0:
            raise ValueError("max_residual_norm must be positive or None")

        covariance_scale = 1.0
        update_action = "updated"
        accepted = True
        residual_over_threshold = (
            residual_threshold is not None and residual_norm > residual_threshold
        )
        safety_over_threshold = safety_threshold is not None and nis > safety_threshold
        reject_by_residual = residual_over_threshold and (
            safety_threshold is None or safety_over_threshold
        )

        if reject_by_residual:
            accepted = False
            update_action = "missed_detection"
        elif safety_over_threshold:
            accepted = False
            update_action = "missed_detection"
        elif threshold is not None and nis > threshold and robust_update is None:
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
                innovation_covariance = observation @ self.covariance @ observation.T + covariance
            if robust_action is not None:
                update_action = robust_action

        if accepted:
            prior_covariance = self.covariance.copy()
            try:
                kalman_gain = np.linalg.solve(
                    innovation_covariance,
                    observation @ prior_covariance,
                ).T
            except np.linalg.LinAlgError:
                kalman_gain = prior_covariance @ observation.T @ np.linalg.pinv(
                    innovation_covariance
                )
            self.mean = self.mean + kalman_gain @ residual
            identity = np.eye(6)
            joseph = identity - kalman_gain @ observation
            self.covariance = symmetrized(
                joseph @ prior_covariance @ joseph.T + kalman_gain @ covariance @ kalman_gain.T
            )
            self.filter = KalmanFilter((self.mean.copy(), self.covariance.copy()))
            self._sync_from_filter()

        return TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=vector.size,
            accepted=bool(accepted),
            update_action=update_action,
            nis=float(nis),
            gate_threshold=threshold,
            safety_gate_threshold=safety_threshold,
            residual_gate_threshold_m=residual_threshold,
            covariance_scale=float(covariance_scale),
            inflation_alpha=alpha if robust_update == "nis-inflate" else None,
            residual_norm_m=residual_norm,
        )

    def update(
        self,
        measurement: TrackingMeasurement | RadarPolarMeasurement,
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
        if isinstance(measurement, RadarPolarMeasurement):
            return self._update_polar_radar(
                measurement,
                gate_threshold=gate_threshold,
                safety_gate_threshold=safety_gate_threshold,
                max_residual_norm=max_residual_norm,
                robust_update=robust_update,
                inflation_alpha=inflation_alpha,
                student_t_dof=student_t_dof,
                huber_threshold=huber_threshold,
            )

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

    first = ordered[0]
    bootstrap_diagnostics = bootstrap_tracking_diagnostics(first)
    records: list[dict[str, object]] = [
        {
            "time_s": first.time_s,
            "source": first.source,
            "state": tracker.state.copy(),
            "covariance": tracker.covariance_matrix.copy(),
            **bootstrap_diagnostics.to_record(),
        }
    ]
    for measurement in ordered[1:]:
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
