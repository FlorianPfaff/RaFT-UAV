"""Asynchronous constant-velocity Kalman fusion baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import chi2

from raft_uav.baselines.update_logic import (
    gate_threshold_for_measurement,
    huber_threshold_for_measurement,
    inflation_alpha_for_measurement,
    plan_linear_measurement_update,
    robust_update_for_measurement,
    student_t_dof_for_measurement,
    symmetrized,
)

try:  # Keep the reproduced baseline on PyRecEst when the dependency is installed.
    from pyrecest.filters import KalmanFilter
except ImportError:  # pragma: no cover - exercised only in lightweight local smoke tests.
    KalmanFilter = None


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


def _raft_update_action(action: object) -> str:
    """Normalize PyRecEst diagnostic labels to RaFT-UAV's public labels."""

    action_string = str(action)
    if action_string == "huberized":
        return "huber"
    return action_string


class AsyncConstantVelocityKalmanTracker:
    """Asynchronous constant-velocity Kalman tracker with optional NIS gating.

    PyRecEst is used for accepted linear updates and update diagnostics when
    available. RaFT-UAV still owns the application-level reject/robust-scale
    policy so rejected measurements remain observable in diagnostics.
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
            self.covariance = symmetrized(self.covariance)
            self.current_time_s = float(time_s)

    def update(
        self,
        measurement: TrackingMeasurement,
        gate_threshold: float | None = None,
        robust_update: str | None = None,
        inflation_alpha: float = 1.0,
        student_t_dof: float = 4.0,
        huber_threshold: float = 2.0,
    ) -> TrackingUpdateDiagnostics:
        """Predict to and conditionally update from one RF or radar measurement."""

        self.predict_to(measurement.time_s)
        observation = measurement_matrix(measurement.vector.size)

        if self.filter is not None and hasattr(self.filter, "update_linear_robust"):
            update_diagnostics = self.filter.update_linear_robust(
                measurement.vector,
                observation,
                measurement.covariance,
                robust_update=robust_update,
                gate_threshold=gate_threshold,
                student_t_dof=student_t_dof,
                huber_threshold=huber_threshold,
                inflation_alpha=inflation_alpha,
                return_diagnostics=True,
            )
            self.mean = np.asarray(self.filter.get_point_estimate(), dtype=float)
            self.covariance = symmetrized(
                np.asarray(self.filter.filter_state.C, dtype=float).reshape(6, 6)
            )

            diagnostic_residual = np.asarray(
                update_diagnostics["residual"],
                dtype=float,
            ).reshape(-1)
            diagnostic_nis = float(np.asarray(update_diagnostics["nis"], dtype=float))
            diagnostic_scale = float(np.asarray(update_diagnostics["scale"], dtype=float))
            diagnostic_action = _raft_update_action(
                update_diagnostics.get("action", "updated")
            )
            diagnostic_accepted = bool(update_diagnostics.get("accepted", True))

            return TrackingUpdateDiagnostics(
                time_s=float(measurement.time_s),
                source=measurement.source,
                measurement_dim=measurement.vector.size,
                accepted=diagnostic_accepted,
                update_action=diagnostic_action,
                nis=diagnostic_nis,
                gate_threshold=None if gate_threshold is None else float(gate_threshold),
                covariance_scale=diagnostic_scale,
                inflation_alpha=float(inflation_alpha)
                if robust_update == "nis-inflate"
                else None,
                residual_norm_m=float(np.linalg.norm(diagnostic_residual)),
            )

        # Fallback for local smoke tests or older PyRecEst installations. The project
        # dependency points at PyRecEst main, where ``update_linear_robust`` is
        # available, so the branch above is the normal path.
        plan = plan_linear_measurement_update(
            mean=self.mean,
            covariance_matrix=self.covariance,
            measurement_vector=measurement.vector,
            measurement_covariance=measurement.covariance,
            observation_matrix=observation,
            gate_threshold=gate_threshold,
            robust_update=robust_update,
            inflation_alpha=inflation_alpha,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
        )

        update_diagnostics: Mapping[str, object] | None = None
        if plan.accepted:
            update_diagnostics = self._linear_update(
                plan.observation,
                plan.residual,
                plan.innovation_covariance,
                plan.covariance,
                plan.vector,
                nominal_measurement_covariance=measurement.covariance,
                covariance_scale=plan.covariance_scale,
                update_action=plan.update_action,
            )

        diagnostic_residual = plan.residual
        diagnostic_nis = plan.nis
        diagnostic_scale = plan.covariance_scale
        diagnostic_action = plan.update_action
        if update_diagnostics is not None:
            diagnostic_residual = np.asarray(
                update_diagnostics["residual"],
                dtype=float,
            ).reshape(-1)
            diagnostic_nis = float(np.asarray(update_diagnostics["nis"], dtype=float))
            diagnostic_scale = float(update_diagnostics["scale"])
            diagnostic_action = str(update_diagnostics["action"])

        return TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=plan.vector.size,
            accepted=plan.accepted,
            update_action=diagnostic_action,
            nis=diagnostic_nis,
            gate_threshold=plan.threshold,
            covariance_scale=diagnostic_scale,
            inflation_alpha=plan.inflation_alpha if robust_update == "nis-inflate" else None,
            residual_norm_m=float(np.linalg.norm(diagnostic_residual)),
        )

    def _linear_update(
        self,
        observation: np.ndarray,
        residual: np.ndarray,
        innovation_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        vector: np.ndarray,
        *,
        nominal_measurement_covariance: np.ndarray | None = None,
        covariance_scale: float = 1.0,
        update_action: str = "updated",
    ) -> Mapping[str, object] | None:
        """Apply an accepted linear Kalman update and return PyRecEst diagnostics."""

        if self.filter is not None:
            base_covariance = (
                measurement_covariance
                if nominal_measurement_covariance is None
                else nominal_measurement_covariance
            )
            try:
                diagnostics = self.filter.update_linear(
                    vector,
                    observation,
                    base_covariance,
                    return_diagnostics=True,
                    scale=covariance_scale,
                    action=update_action,
                )
            except TypeError:  # pragma: no cover - compatibility with older PyRecEst.
                self.filter.update_linear(vector, observation, measurement_covariance)
                diagnostics = None
            self.mean = np.asarray(self.filter.get_point_estimate(), dtype=float)
            self.covariance = symmetrized(
                np.asarray(self.filter.filter_state.C, dtype=float).reshape(6, 6)
            )
            return diagnostics

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

        self.mean = updated_mean
        self.covariance = symmetrized(updated_covariance)
        return None


def run_async_cv_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
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
