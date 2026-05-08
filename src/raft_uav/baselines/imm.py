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
    constant_velocity_matrix,
    gate_threshold_from_probability,
    measurement_matrix,
    normalized_innovation_squared,
    white_acceleration_process_noise,
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
        position = np.asarray(initial_position, dtype=float).reshape(-1)
        if position.size == 2:
            position = np.array([position[0], position[1], 0.0])
        if position.size != 3:
            raise ValueError("initial_position must contain 2 or 3 elements")

        self.modes = tuple(modes or default_imm_modes(acceleration_std_mps2))
        if not self.modes:
            raise ValueError("modes must contain at least one IMMMode")
        self.mode_names = tuple(mode.name for mode in self.modes)
        self.mode_switch_time_constant_s = float(mode_switch_time_constant_s)

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
        robust_update: str | None = None,
        inflation_alpha: float = 1.0,
    ) -> TrackingUpdateDiagnostics:
        """Predict to and conditionally update from one RF or radar measurement."""

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
                update_action = "inflated"
            elif robust_update is None:
                accepted = False
                update_action = "rejected"
            else:
                raise ValueError(f"unknown robust update mode {robust_update!r}")

        if accepted:
            self.filter.update_linear(vector, observation, covariance)
            self._sync_combined_state()

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

    def _sync_combined_state(self) -> None:
        combined = self.filter.combined_filter_state
        self.mean = np.asarray(combined.mu, dtype=float).reshape(6)
        self.covariance = _symmetrized(np.asarray(combined.C, dtype=float).reshape(6, 6))


def run_async_imm_baseline(
    measurements: Iterable[TrackingMeasurement],
    acceleration_std_mps2: float = 4.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    modes: Sequence[IMMMode] | None = None,
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

    records: list[dict[str, object]] = []
    for measurement in ordered:
        diagnostics = tracker.update(
            measurement,
            gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
            ),
            robust_update=_robust_update_for_measurement(
                measurement,
                robust_update_by_source=robust_update_by_source,
            ),
            inflation_alpha=_inflation_alpha_for_measurement(
                measurement,
                inflation_alpha_by_source=inflation_alpha_by_source,
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
