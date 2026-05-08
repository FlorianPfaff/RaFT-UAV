"""Robust factor-graph/MAP smoothing for asynchronous CV tracking records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    constant_velocity_matrix,
    measurement_matrix,
    white_acceleration_process_noise,
)

ROBUST_MAP_LOSSES = ("linear", "soft_l1", "huber", "cauchy", "arctan")


@dataclass(frozen=True)
class RobustMapSmootherConfig:
    """Configuration for robust factor-graph/MAP smoothing."""

    loss: str = "huber"
    loss_scale: float = 3.0
    max_iterations: int = 50
    relative_tolerance: float = 1.0e-5
    measurement_time_tolerance_s: float = 1.0e-6
    process_position_floor_m: float = 0.25
    process_velocity_floor_mps: float = 0.25
    accepted_measurements_only: bool = False

    def __post_init__(self) -> None:
        if self.loss not in ROBUST_MAP_LOSSES:
            raise ValueError(f"loss must be one of {ROBUST_MAP_LOSSES}")
        if self.loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if self.measurement_time_tolerance_s < 0.0:
            raise ValueError("measurement_time_tolerance_s must be nonnegative")
        if self.process_position_floor_m < 0.0:
            raise ValueError("process_position_floor_m must be nonnegative")
        if self.process_velocity_floor_mps < 0.0:
            raise ValueError("process_velocity_floor_mps must be nonnegative")


@dataclass(frozen=True)
class RobustMapResult:
    states: np.ndarray
    covariances: np.ndarray
    matched_measurements: int
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    message: str


@dataclass(frozen=True)
class _MeasurementFactor:
    index: int
    vector: np.ndarray
    covariance: np.ndarray
    source: str


def robust_map_smooth_records(
    records: list[dict[str, object]],
    *,
    measurements: Iterable[TrackingMeasurement] | None,
    acceleration_std_mps2: float,
    config: RobustMapSmootherConfig | None = None,
    lag_s: float | None = None,
) -> list[dict[str, object]]:
    """Return records smoothed by a robust constant-velocity factor graph.

    The graph contains one 6D state per posterior record, a prior on the first
    state, constant-velocity process factors, and robust RF/radar measurement
    factors. When the original measurements are not supplied, posterior
    pseudo-measurements are used so the mode remains usable from existing CLI
    call sites.
    """

    cfg = config or RobustMapSmootherConfig()
    if not records:
        return []
    if lag_s is not None and lag_s < 0.0:
        raise ValueError("lag_s must be nonnegative")

    out = [_copy_record(record) for record in records]
    times, filtered_states, filtered_covariances = _record_arrays(out)
    if measurements is None:
        factors = _record_pseudo_measurement_factors(
            out,
            filtered_states,
            filtered_covariances,
            accepted_only=cfg.accepted_measurements_only,
        )
    else:
        factors = _matched_measurement_factors(
            out,
            measurements,
            times,
            time_tolerance_s=cfg.measurement_time_tolerance_s,
            accepted_only=cfg.accepted_measurements_only,
        )
    if not factors:
        raise ValueError("robust-map smoothing matched no measurement factors")

    if lag_s is None:
        result = _solve_window(
            times,
            filtered_states,
            filtered_covariances,
            factors,
            start_index=0,
            end_index=len(out) - 1,
            acceleration_std_mps2=acceleration_std_mps2,
            config=cfg,
        )
        for idx, record in enumerate(out):
            _write_result_to_record(
                record,
                filtered_states[idx],
                filtered_covariances[idx],
                result.states[idx],
                result.covariances[idx],
                method="robust-map",
                lag_s=None,
                result=result,
            )
        return out

    for start_index, time_s in enumerate(times):
        end_index = int(np.searchsorted(times, time_s + lag_s, side="right") - 1)
        if end_index <= start_index:
            _write_result_to_record(
                out[start_index],
                filtered_states[start_index],
                filtered_covariances[start_index],
                filtered_states[start_index],
                filtered_covariances[start_index],
                method="fixed-lag-map",
                lag_s=lag_s,
                result=None,
            )
            continue
        result = _solve_window(
            times,
            filtered_states,
            filtered_covariances,
            factors,
            start_index=start_index,
            end_index=end_index,
            acceleration_std_mps2=acceleration_std_mps2,
            config=cfg,
        )
        _write_result_to_record(
            out[start_index],
            filtered_states[start_index],
            filtered_covariances[start_index],
            result.states[0],
            result.covariances[0],
            method="fixed-lag-map",
            lag_s=lag_s,
            result=result,
        )
    return out


def _solve_window(
    times: np.ndarray,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    measurement_factors: list[_MeasurementFactor],
    *,
    start_index: int,
    end_index: int,
    acceleration_std_mps2: float,
    config: RobustMapSmootherConfig,
) -> RobustMapResult:
    window = slice(start_index, end_index + 1)
    local_times = times[window]
    x0 = filtered_states[window].copy()
    covariances = filtered_covariances[window].copy()
    local_factors = [
        _MeasurementFactor(
            index=factor.index - start_index,
            vector=factor.vector,
            covariance=factor.covariance,
            source=factor.source,
        )
        for factor in measurement_factors
        if start_index <= factor.index <= end_index
    ]
    if not local_factors:
        local_factors = _record_pseudo_measurement_factors(
            [
                {"measurement_dim": 3, "accepted": True, "source": "posterior"}
                for _ in range(x0.shape[0])
            ],
            x0,
            covariances,
            accepted_only=False,
        )

    x = x0.copy()
    previous_norm = max(1.0, float(np.linalg.norm(x)))
    initial_cost = _objective_cost(
        x,
        x0,
        covariances,
        local_times,
        local_factors,
        acceleration_std_mps2=acceleration_std_mps2,
        config=config,
    )
    final_cost = initial_cost
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):
        a_matrix, b_vector = _linearized_system(
            x,
            x0,
            covariances,
            local_times,
            local_factors,
            acceleration_std_mps2=acceleration_std_mps2,
            config=config,
        )
        solution = lsmr(a_matrix, b_vector, atol=1.0e-10, btol=1.0e-10, maxiter=2000)[0]
        new_x = solution.reshape(x.shape)
        delta = float(np.linalg.norm(new_x - x))
        x = new_x
        final_cost = _objective_cost(
            x,
            x0,
            covariances,
            local_times,
            local_factors,
            acceleration_std_mps2=acceleration_std_mps2,
            config=config,
        )
        if delta <= config.relative_tolerance * previous_norm:
            success = True
            message = "converged"
            break
        previous_norm = max(1.0, float(np.linalg.norm(x)))

    return RobustMapResult(
        states=x,
        covariances=covariances,
        matched_measurements=len(local_factors),
        initial_cost=float(initial_cost),
        final_cost=float(final_cost),
        iterations=int(iterations),
        success=bool(success),
        message=message,
    )


def _linearized_system(
    states: np.ndarray,
    reference_states: np.ndarray,
    reference_covariances: np.ndarray,
    times: np.ndarray,
    measurement_factors: list[_MeasurementFactor],
    *,
    acceleration_std_mps2: float,
    config: RobustMapSmootherConfig,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0
    n_states = states.shape[0]

    row = _append_factor(
        rows,
        cols,
        data,
        rhs,
        row,
        [(0, np.eye(6))],
        _whiten_covariance(reference_covariances[0]) @ reference_states[0],
        _whiten_covariance(reference_covariances[0]),
    )

    for idx in range(n_states - 1):
        dt_s = max(0.0, float(times[idx + 1] - times[idx]))
        transition = constant_velocity_matrix(dt_s)
        process_cov = _regularized_process_noise(
            dt_s,
            acceleration_std_mps2,
            position_floor_m=config.process_position_floor_m,
            velocity_floor_mps=config.process_velocity_floor_mps,
        )
        white = _whiten_covariance(process_cov)
        row = _append_factor(
            rows,
            cols,
            data,
            rhs,
            row,
            [(idx + 1, np.eye(6)), (idx, -transition)],
            np.zeros(6),
            white,
        )

    for factor in measurement_factors:
        observation = measurement_matrix(factor.vector.size)
        white = _whiten_covariance(factor.covariance)
        residual = white @ (factor.vector - observation @ states[factor.index])
        weights = _robust_weights(residual, config)
        weighted_white = weights[:, None] * white
        row = _append_factor(
            rows,
            cols,
            data,
            rhs,
            row,
            [(factor.index, observation)],
            weighted_white @ factor.vector,
            weighted_white,
        )

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(row, 6 * n_states)).tocsr()
    return matrix, np.asarray(rhs, dtype=float)


def _append_factor(
    rows: list[int],
    cols: list[int],
    data: list[float],
    rhs: list[float],
    row: int,
    blocks: list[tuple[int, np.ndarray]],
    target: np.ndarray,
    white: np.ndarray,
) -> int:
    for local_row in range(white.shape[0]):
        for state_index, block in blocks:
            weighted = white[local_row] @ block
            for dim, value in enumerate(weighted):
                if value != 0.0:
                    rows.append(row + local_row)
                    cols.append(6 * state_index + dim)
                    data.append(float(value))
        rhs.append(float(target[local_row]))
    return row + white.shape[0]


def _objective_cost(
    states: np.ndarray,
    reference_states: np.ndarray,
    reference_covariances: np.ndarray,
    times: np.ndarray,
    measurement_factors: list[_MeasurementFactor],
    *,
    acceleration_std_mps2: float,
    config: RobustMapSmootherConfig,
) -> float:
    cost = float(
        0.5
        * np.sum(
            (_whiten_covariance(reference_covariances[0]) @ (states[0] - reference_states[0]))
            ** 2
        )
    )
    for idx in range(states.shape[0] - 1):
        dt_s = max(0.0, float(times[idx + 1] - times[idx]))
        residual = states[idx + 1] - constant_velocity_matrix(dt_s) @ states[idx]
        process_cov = _regularized_process_noise(
            dt_s,
            acceleration_std_mps2,
            position_floor_m=config.process_position_floor_m,
            velocity_floor_mps=config.process_velocity_floor_mps,
        )
        cost += float(0.5 * np.sum((_whiten_covariance(process_cov) @ residual) ** 2))
    for factor in measurement_factors:
        observation = measurement_matrix(factor.vector.size)
        residual = _whiten_covariance(factor.covariance) @ (
            factor.vector - observation @ states[factor.index]
        )
        cost += float(np.sum(_robust_component_cost(residual, config)))
    return cost


def _robust_weights(residual: np.ndarray, config: RobustMapSmootherConfig) -> np.ndarray:
    if config.loss == "linear":
        return np.ones_like(residual)
    abs_residual = np.abs(residual)
    c = config.loss_scale
    scaled_sq = (residual / c) ** 2
    eps = 1.0e-12
    if config.loss == "huber":
        return np.sqrt(np.where(abs_residual <= c, 1.0, c / np.maximum(abs_residual, eps)))
    if config.loss == "soft_l1":
        return np.power(1.0 + scaled_sq, -0.25)
    if config.loss == "cauchy":
        return np.sqrt(1.0 / (1.0 + scaled_sq))
    if config.loss == "arctan":
        return np.sqrt(1.0 / (1.0 + scaled_sq**2))
    raise ValueError(f"unknown robust loss {config.loss!r}")


def _robust_component_cost(residual: np.ndarray, config: RobustMapSmootherConfig) -> np.ndarray:
    if config.loss == "linear":
        return 0.5 * residual**2
    c = config.loss_scale
    abs_residual = np.abs(residual)
    scaled_sq = (residual / c) ** 2
    if config.loss == "huber":
        return np.where(abs_residual <= c, 0.5 * residual**2, c * (abs_residual - 0.5 * c))
    if config.loss == "soft_l1":
        return c**2 * (np.sqrt(1.0 + scaled_sq) - 1.0)
    if config.loss == "cauchy":
        return 0.5 * c**2 * np.log1p(scaled_sq)
    if config.loss == "arctan":
        return 0.5 * c**2 * np.arctan(scaled_sq)
    raise ValueError(f"unknown robust loss {config.loss!r}")


def _record_pseudo_measurement_factors(
    records: list[dict[str, object]],
    states: np.ndarray,
    covariances: np.ndarray,
    *,
    accepted_only: bool,
) -> list[_MeasurementFactor]:
    factors: list[_MeasurementFactor] = []
    for idx, record in enumerate(records):
        if accepted_only and not bool(record.get("accepted", True)):
            continue
        dimension = int(record.get("measurement_dim", 3) or 3)
        if dimension not in (2, 3, 6):
            dimension = 3
        observation = measurement_matrix(dimension)
        covariance = _symmetrized(observation @ covariances[idx] @ observation.T)
        covariance = covariance + np.eye(dimension) * 1.0e-6
        factors.append(
            _MeasurementFactor(
                index=int(idx),
                vector=observation @ states[idx],
                covariance=covariance,
                source=str(record.get("source", "posterior")),
            )
        )
    return factors


def _matched_measurement_factors(
    records: list[dict[str, object]],
    measurements: Iterable[TrackingMeasurement] | None,
    times: np.ndarray,
    *,
    time_tolerance_s: float,
    accepted_only: bool,
) -> list[_MeasurementFactor]:
    if measurements is None:
        return []
    used_record_indices: set[int] = set()
    factors: list[_MeasurementFactor] = []
    ordered = sorted(
        measurements,
        key=lambda item: (float(item.time_s), str(item.source), int(item.vector.size)),
    )
    for measurement in ordered:
        candidate_indices = _candidate_record_indices(
            times,
            float(measurement.time_s),
            tolerance_s=time_tolerance_s,
        )
        candidate_indices = [idx for idx in candidate_indices if idx not in used_record_indices]
        source_matches = [
            idx for idx in candidate_indices if str(records[idx].get("source")) == measurement.source
        ]
        if source_matches:
            candidate_indices = source_matches
        if not candidate_indices:
            continue
        best_index = min(candidate_indices, key=lambda idx: abs(times[idx] - measurement.time_s))
        used_record_indices.add(best_index)
        if accepted_only and not bool(records[best_index].get("accepted", True)):
            continue
        factors.append(
            _MeasurementFactor(
                index=int(best_index),
                vector=np.asarray(measurement.vector, dtype=float).reshape(-1),
                covariance=np.asarray(measurement.covariance, dtype=float),
                source=measurement.source,
            )
        )
    return factors


def _candidate_record_indices(
    times: np.ndarray, time_s: float, *, tolerance_s: float
) -> list[int]:
    start = int(np.searchsorted(times, time_s - tolerance_s, side="left"))
    end = int(np.searchsorted(times, time_s + tolerance_s, side="right"))
    return list(range(start, end))


def _regularized_process_noise(
    dt_s: float,
    acceleration_std_mps2: float,
    *,
    position_floor_m: float,
    velocity_floor_mps: float,
) -> np.ndarray:
    covariance = white_acceleration_process_noise(dt_s, acceleration_std_mps2)
    covariance = covariance + np.diag(
        [
            position_floor_m**2,
            position_floor_m**2,
            position_floor_m**2,
            velocity_floor_mps**2,
            velocity_floor_mps**2,
            velocity_floor_mps**2,
        ]
    )
    return _symmetrized(covariance)


def _whiten_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = _symmetrized(np.asarray(covariance, dtype=float))
    jitter = 1.0e-9
    identity = np.eye(covariance.shape[0])
    for _ in range(8):
        try:
            return np.linalg.inv(np.linalg.cholesky(covariance + jitter * identity))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, jitter)
    return np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _write_result_to_record(
    record: dict[str, object],
    filtered_state: np.ndarray,
    filtered_covariance: np.ndarray,
    state: np.ndarray,
    covariance: np.ndarray,
    *,
    method: str,
    lag_s: float | None,
    result: RobustMapResult | None,
) -> None:
    record["filtered_state"] = filtered_state.copy()
    record["filtered_covariance"] = filtered_covariance.copy()
    record["state"] = state.copy()
    record["covariance"] = covariance.copy()
    record["smoother_method"] = method
    record["smoother_lag_s"] = lag_s
    record["map_covariance_source"] = "filtered"
    if result is None:
        record["map_success"] = True
        record["map_iterations"] = 0
        record["map_initial_cost"] = 0.0
        record["map_final_cost"] = 0.0
        record["map_matched_measurements"] = 0
        record["map_message"] = "window contains no future state"
        return
    record["map_success"] = result.success
    record["map_iterations"] = result.iterations
    record["map_initial_cost"] = result.initial_cost
    record["map_final_cost"] = result.final_cost
    record["map_matched_measurements"] = result.matched_measurements
    record["map_message"] = result.message


def _record_arrays(
    records: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(record["time_s"]) for record in records], dtype=float)
    states = np.stack([np.asarray(record["state"], dtype=float).reshape(6) for record in records])
    covariances = np.stack(
        [np.asarray(record["covariance"], dtype=float).reshape(6, 6) for record in records]
    )
    return times, states, covariances


def _copy_record(record: dict[str, object]) -> dict[str, object]:
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in record.items()}


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
