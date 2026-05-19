"""Asynchronous interacting-multiple-model fusion tracker for RaFT-UAV.

The tracker keeps the same public interface as ``AsyncConstantVelocityKalmanTracker``
so that radar-association code can use it through a tracker-factory hook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pyrecest.filters import InteractingMultipleModelFilter, KalmanFilter

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
    bootstrap_tracking_diagnostics,
    constant_velocity_matrix,
    gate_threshold_from_probability,
    measurement_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.update_logic import (
    gate_threshold_for_measurement,
    huber_threshold_for_measurement,
    inflation_alpha_for_measurement,
    max_residual_norm_for_measurement,
    plan_linear_measurement_update,
    robust_update_for_measurement,
    student_t_dof_for_measurement,
    symmetrized,
)


@dataclass(frozen=True)
class IMMMode:
    """One motion mode in the 6D ENU IMM state space.

    The state remains ``[east, north, up, v_east, v_north, v_up]`` for every mode,
    which matches PyRecEst's current IMM requirement that all subfilters have the
    same state dimension.
    """

    name: str
    acceleration_std_mps2: float
    turn_rate_radps: float = 0.0

    def transition_matrix(self, dt_s: float) -> np.ndarray:
        """Return the 6D transition matrix for this mode."""

        if abs(self.turn_rate_radps) < 1.0e-9:
            return constant_velocity_matrix(dt_s)
        return fixed_turn_rate_matrix(dt_s, self.turn_rate_radps)

    def process_noise(self, dt_s: float) -> np.ndarray:
        """Return the 6D process-noise covariance for this mode."""

        return white_acceleration_process_noise(dt_s, self.acceleration_std_mps2)


def default_imm_modes(base_acceleration_std_mps2: float = 4.0) -> tuple[IMMMode, ...]:
    """Return a compact UAV motion-mode bank.

    The numbers are intentionally conservative starting values. They should be
    tuned with leave-flight-out validation rather than optimized on the test flight.
    """

    base = float(base_acceleration_std_mps2)
    return (
        IMMMode("cv-smooth", max(0.5, 0.4 * base), 0.0),
        IMMMode("cv-nominal", base, 0.0),
        IMMMode("cv-maneuver", 2.5 * base, 0.0),
        IMMMode("ct-left-6dps", base, np.deg2rad(6.0)),
        IMMMode("ct-right-6dps", base, -np.deg2rad(6.0)),
    )


def fixed_turn_rate_matrix(dt_s: float, turn_rate_radps: float) -> np.ndarray:
    """Return a 6D fixed-turn-rate transition matrix.

    The horizontal dynamics assume

    ``d vx / dt = -omega * vy`` and ``d vy / dt = omega * vx``.

    For a fixed ``omega`` this is linear in the 6D state, so PyRecEst's linear IMM
    can be used without switching to UKF/EKF prediction.
    """

    dt = float(dt_s)
    omega = float(turn_rate_radps)
    if abs(omega) < 1.0e-9:
        return constant_velocity_matrix(dt)

    angle = omega * dt
    sin_a = float(np.sin(angle))
    cos_a = float(np.cos(angle))

    matrix = np.eye(6)
    matrix[0, 3] = sin_a / omega
    matrix[0, 4] = (cos_a - 1.0) / omega
    matrix[1, 3] = (1.0 - cos_a) / omega
    matrix[1, 4] = sin_a / omega
    matrix[2, 5] = dt
    matrix[3, 3] = cos_a
    matrix[3, 4] = -sin_a
    matrix[4, 3] = sin_a
    matrix[4, 4] = cos_a
    return matrix


def uniform_ctmc_transition_matrix(
    n_modes: int,
    dt_s: float,
    mode_switch_time_constant_s: float,
) -> np.ndarray:
    """Return a row-stochastic, dt-dependent Markov transition matrix.

    This is the closed form of a continuous-time Markov chain with a uniform
    stationary distribution. At ``dt=0`` it becomes identity; at larger ``dt`` it
    gradually allows mode changes.
    """

    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be positive")
    tau = float(mode_switch_time_constant_s)
    if tau <= 0.0:
        raise ValueError("mode_switch_time_constant_s must be positive")
    dt = max(0.0, float(dt_s))
    persistence = float(np.exp(-dt / tau))
    matrix = np.full((n, n), (1.0 - persistence) / n)
    matrix[np.diag_indices(n)] += persistence
    return matrix


class AsyncInteractingMultipleModelTracker:
    """Asynchronous RF/radar fusion tracker using PyRecEst's IMM.

    The interface mirrors ``AsyncConstantVelocityKalmanTracker``: ``predict_to``,
    ``update``, ``state``, and ``covariance_matrix``. This makes it easy to use as
    a drop-in tracker in the existing radar-association loop.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_time_s: float,
        initial_position_std_m: float = 50.0,
        initial_velocity_std_mps: float = 15.0,
        acceleration_std_mps2: float = 4.0,
        modes: Sequence[IMMMode] | None = None,
        initial_mode_probabilities: Sequence[float] | None = None,
        mode_switch_time_constant_s: float = 20.0,
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

        self.modes = tuple(modes or default_imm_modes(acceleration_std_mps2))
        if not self.modes:
            raise ValueError("modes must contain at least one IMMMode")
        self.mode_names = tuple(mode.name for mode in self.modes)
        self.mode_switch_time_constant_s = float(mode_switch_time_constant_s)

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

        filter_bank = [
            KalmanFilter((self.mean.copy(), self.covariance.copy())) for _ in self.modes
        ]
        transition_matrix = np.eye(len(self.modes))
        mode_probabilities = None
        if initial_mode_probabilities is not None:
            mode_probabilities = np.asarray(initial_mode_probabilities, dtype=float)
        self.filter = InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=transition_matrix,
            mode_probabilities=mode_probabilities,
        )
        self.current_time_s = float(initial_time_s)
        self._sync_combined_state()

    @property
    def state(self) -> np.ndarray:
        """Return the current moment-matched posterior mean."""

        return self.mean.copy()

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Return the current moment-matched posterior covariance."""

        return self.covariance.copy()

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Return IMM mode probabilities in ``self.mode_names`` order."""

        return np.asarray(self.filter.mode_probabilities, dtype=float).copy()

    @property
    def mode_probability_map(self) -> dict[str, float]:
        """Return mode probabilities as a JSON/CSV-friendly mapping."""

        return {
            name: float(probability)
            for name, probability in zip(self.mode_names, self.mode_probabilities)
        }

    @property
    def most_likely_mode_name(self) -> str:
        """Return the name of the currently most probable IMM motion mode."""

        return self.mode_names[int(np.argmax(self.mode_probabilities))]

    def predict_to(self, time_s: float) -> None:
        """Predict to an absolute timestamp."""

        dt_s = float(time_s) - self.current_time_s
        if dt_s < -1.0e-9:
            raise ValueError("measurements must be processed in chronological order")
        if dt_s <= 0.0:
            return

        self.filter.transition_matrix = uniform_ctmc_transition_matrix(
            len(self.modes),
            dt_s=dt_s,
            mode_switch_time_constant_s=self.mode_switch_time_constant_s,
        )
        system_matrices = [mode.transition_matrix(dt_s) for mode in self.modes]
        process_noises = [mode.process_noise(dt_s) for mode in self.modes]
        self.filter.predict_linear(system_matrices, process_noises)
        self.current_time_s = float(time_s)
        self._sync_combined_state()

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
        plan = plan_linear_measurement_update(
            mean=self.mean,
            covariance_matrix=self.covariance,
            measurement_vector=measurement.vector,
            measurement_covariance=measurement.covariance,
            observation_matrix=measurement_matrix(measurement.vector.size),
            gate_threshold=gate_threshold,
            safety_gate_threshold=safety_gate_threshold,
            max_residual_norm=max_residual_norm,
            robust_update=robust_update,
            inflation_alpha=inflation_alpha,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
        )

        if plan.accepted:
            self.filter.update_linear(plan.vector, plan.observation, plan.covariance)
            self._sync_combined_state()

        return TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=plan.vector.size,
            accepted=plan.accepted,
            update_action=plan.update_action,
            nis=plan.nis,
            gate_threshold=plan.threshold,
            safety_gate_threshold=plan.safety_threshold,
            residual_gate_threshold_m=plan.residual_threshold,
            covariance_scale=plan.covariance_scale,
            inflation_alpha=plan.inflation_alpha if robust_update == "nis-inflate" else None,
            residual_norm_m=plan.residual_norm,
        )

    def _sync_combined_state(self) -> None:
        combined = self.filter.combined_filter_state
        self.mean = np.asarray(combined.mu, dtype=float).reshape(6)
        self.covariance = symmetrized(np.asarray(combined.C, dtype=float).reshape(6, 6))


def run_async_imm_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    modes: Sequence[IMMMode] | None = None,
    student_t_dof_by_source: Mapping[str, float] | None = None,
    huber_threshold_by_source: Mapping[str, float] | None = None,
    mode_switch_time_constant_s: float = 20.0,
) -> list[dict[str, object]]:
    """Run the asynchronous IMM baseline and return posterior records."""

    ordered = sorted(measurements, key=lambda item: item.time_s)
    if not ordered:
        return []

    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=ordered[0].vector,
        initial_time_s=ordered[0].time_s,
        acceleration_std_mps2=acceleration_std_mps2,
        modes=modes,
        mode_switch_time_constant_s=mode_switch_time_constant_s,
    )

    first = ordered[0]
    bootstrap_diagnostics = bootstrap_tracking_diagnostics(first)
    records: list[dict[str, object]] = [
        {
            "time_s": first.time_s,
            "source": first.source,
            "state": tracker.state.copy(),
            "covariance": tracker.covariance_matrix.copy(),
            "mode_names": tracker.mode_names,
            "mode_probabilities": tracker.mode_probabilities.copy(),
            "mode_probability_map": tracker.mode_probability_map,
            "most_likely_mode": tracker.most_likely_mode_name,
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
                "mode_names": tracker.mode_names,
                "mode_probabilities": tracker.mode_probabilities.copy(),
                "mode_probability_map": tracker.mode_probability_map,
                "most_likely_mode": tracker.most_likely_mode_name,
                **diagnostics.to_record(),
            }
        )
    return records


_symmetrized = symmetrized
