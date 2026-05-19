"""Radar row association for the asynchronous CV baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyrecest.filters import KalmanFilter
from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
    constant_velocity_matrix,
    gate_threshold_from_probability,
    measurement_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.update_logic import (
    max_residual_norm_for_measurement,
    plan_linear_measurement_update,
)

RADAR_ASSOCIATION_MODES = (
    "oracle-nearest-truth",
    "prediction-nis",
    "track-continuity",
    "geometry-score",
    "pda-mixture",
    "track-bank",
    "stable-segments",
)


@dataclass(frozen=True)
class _TrackSegment:
    frame: pd.DataFrame
    track_id: int
    start_time_s: float
    end_time_s: float
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    frames: int
    mean_catprob: float

    @property
    def score(self) -> float:
        return float(self.frames) * max(self.mean_catprob, 0.0)


def run_async_cv_baseline_with_radar_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    association: str,
    truth: pd.DataFrame | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    track_switch_nis_ratio: float = 0.5,
    candidate_catprob_threshold: float | None = 0.5,
    geometry_velocity_std_mps: float = 12.0,
    geometry_velocity_weight: float = 0.25,
    geometry_switch_penalty: float = 4.0,
    geometry_catprob_weight: float = 2.0,
    pda_nis_temperature: float = 1.0,
    pda_catprob_exponent: float = 1.0,
    track_bank_max_hypotheses: int = 16,
    track_bank_max_assignments: int = 16,
    track_bank_max_candidates: int = 16,
    track_bank_gate_probability: float = 0.9999999,
    track_bank_detection_probability: float = 0.999,
    track_bank_clutter_intensity: float = 1.0e-12,
    track_bank_prune_log_weight_delta: float = 80.0,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    stable_segment_range_gate_m: float | None = 800.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion while selecting at most one radar row per radar frame.

    ``oracle-nearest-truth`` uses ground truth and is only a diagnostic upper
    bound. ``prediction-nis`` picks the radar candidate with the lowest
    normalized innovation squared against the current predicted state.
    ``track-continuity`` prefers the current Fortem track ID and switches only
    when another candidate has a substantially lower NIS. ``geometry-score``
    is an online score that augments NIS with radar velocity consistency,
    track switching, and UAV class-probability penalties. ``pda-mixture``
    keeps a single Kalman update but forms it from a probability-weighted
    candidate mixture and adds candidate spread to the radar covariance.
    ``track-bank`` uses PyRecEst's track-oriented MHT to keep multiple
    single-target association hypotheses alive across radar frames.
    ``stable-segments`` preselects stitched high-confidence Fortem track
    segments and skips all other radar frames.
    """

    if association not in RADAR_ASSOCIATION_MODES:
        raise ValueError(f"unknown radar association mode {association!r}")
    if association == "oracle-nearest-truth" and truth is None:
        raise ValueError("oracle-nearest-truth association requires normalized truth")
    if track_switch_nis_ratio <= 0.0:
        raise ValueError("track_switch_nis_ratio must be positive")
    if geometry_velocity_std_mps <= 0.0:
        raise ValueError("geometry_velocity_std_mps must be positive")
    for name, value in {
        "geometry_velocity_weight": geometry_velocity_weight,
        "geometry_switch_penalty": geometry_switch_penalty,
        "geometry_catprob_weight": geometry_catprob_weight,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if pda_nis_temperature <= 0.0:
        raise ValueError("pda_nis_temperature must be positive")
    if pda_catprob_exponent < 0.0:
        raise ValueError("pda_catprob_exponent must be nonnegative")
    if track_bank_max_hypotheses < 1:
        raise ValueError("track_bank_max_hypotheses must be positive")
    if track_bank_max_assignments < 1:
        raise ValueError("track_bank_max_assignments must be positive")
    if track_bank_max_candidates < 1:
        raise ValueError("track_bank_max_candidates must be positive")
    if not 0.0 < track_bank_gate_probability < 1.0:
        raise ValueError("track_bank_gate_probability must be in (0, 1)")
    if not 0.0 < track_bank_detection_probability < 1.0:
        raise ValueError("track_bank_detection_probability must be in (0, 1)")
    if track_bank_clutter_intensity <= 0.0:
        raise ValueError("track_bank_clutter_intensity must be positive")
    if track_bank_prune_log_weight_delta <= 0.0:
        raise ValueError("track_bank_prune_log_weight_delta must be positive")
    if stable_segment_min_frames < 1:
        raise ValueError("stable_segment_min_frames must be positive")
    if stable_segment_max_transition_speed_mps <= 0.0:
        raise ValueError("stable_segment_max_transition_speed_mps must be positive")
    if stable_segment_range_gate_m is not None and stable_segment_range_gate_m <= 0.0:
        raise ValueError("stable_segment_range_gate_m must be positive or None")

    covariance = np.diag([float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2])
    if association == "track-bank":
        return _run_mht_track_bank(
            rf_measurements=list(rf_measurements),
            radar=radar,
            covariance=covariance,
            acceleration_std_mps2=acceleration_std_mps2,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            max_global_hypotheses=track_bank_max_hypotheses,
            max_assignments_per_hypothesis=track_bank_max_assignments,
            max_candidates_per_track=track_bank_max_candidates,
            gate_probability=track_bank_gate_probability,
            detection_probability=track_bank_detection_probability,
            clutter_intensity=track_bank_clutter_intensity,
            prune_log_weight_delta=track_bank_prune_log_weight_delta,
        )

    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _empty_selected_radar(radar)

    stable_anchor_by_key: dict[object, pd.Series] | None = None
    if association == "stable-segments":
        stable_anchors = _select_stable_radar_segments(
            radar,
            range_gate_m=stable_segment_range_gate_m,
            catprob_threshold=candidate_catprob_threshold,
            min_segment_frames=stable_segment_min_frames,
            max_transition_speed_mps=stable_segment_max_transition_speed_mps,
        )
        stable_anchor_by_key = {
            _radar_row_key(row): row for _, row in stable_anchors.iterrows()
        }

    start_index = 0
    initial_measurement = None
    initial_events = enumerate(events) if association == "stable-segments" else [(0, events[0])]
    for index, event in initial_events:
        initial_measurement = _initial_measurement(
            event,
            association=association,
            covariance=covariance,
            stable_anchor_by_key=stable_anchor_by_key,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        if initial_measurement is not None:
            start_index = int(index)
            break
    if initial_measurement is None:
        return [], _empty_selected_radar(radar)

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    current_track_id: int | None = None

    for event in events[start_index:]:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(
                measurement,
                gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=gate_probabilities_by_source,
                    gate_thresholds_by_source=gate_thresholds_by_source,
                ),
                safety_gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=safety_gate_probabilities_by_source,
                    gate_thresholds_by_source=safety_gate_thresholds_by_source,
                ),
                max_residual_norm=_max_residual_norm_for_measurement(
                    measurement,
                    max_residual_norms_by_source=max_residual_norms_by_source,
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
            records.append(_record(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        time_s = float(event["time_s"])
        tracker.predict_to(time_s)
        selected = _select_radar_candidate(
            candidates,
            association=association,
            tracker=tracker,
            covariance=covariance,
            truth=truth,
            current_track_id=current_track_id,
            track_switch_nis_ratio=track_switch_nis_ratio,
            candidate_catprob_threshold=candidate_catprob_threshold,
            geometry_velocity_std_mps=geometry_velocity_std_mps,
            geometry_velocity_weight=geometry_velocity_weight,
            geometry_switch_penalty=geometry_switch_penalty,
            geometry_catprob_weight=geometry_catprob_weight,
            pda_nis_temperature=pda_nis_temperature,
            pda_catprob_exponent=pda_catprob_exponent,
            stable_anchor_by_key=stable_anchor_by_key,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        if selected is None:
            continue

        measurement = _radar_row_to_measurement(selected, covariance)
        diagnostics = tracker.update(
            measurement,
            gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
            ),
            safety_gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=safety_gate_probabilities_by_source,
                gate_thresholds_by_source=safety_gate_thresholds_by_source,
            ),
            max_residual_norm=_max_residual_norm_for_measurement(
                measurement,
                max_residual_norms_by_source=max_residual_norms_by_source,
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
        if diagnostics.accepted:
            current_track_id = _optional_track_id(selected)
            selected_rows.append(selected)
        records.append(
            _record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode=association,
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


def _events(
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {"time_s": measurement.time_s, "priority": 0, "kind": "rf", "measurement": measurement}
        for measurement in rf_measurements
    ]
    for group in _radar_frame_groups(radar):
        events.append(
            {
                "time_s": float(group["time_s"].median()),
                "priority": 1,
                "kind": "radar",
                "candidates": group,
            }
        )
    return sorted(events, key=lambda item: (float(item["time_s"]), int(item["priority"])))


def _run_mht_track_bank(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
    covariance: np.ndarray,
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    candidate_catprob_threshold: float | None,
    max_global_hypotheses: int,
    max_assignments_per_hypothesis: int,
    max_candidates_per_track: int,
    gate_probability: float,
    detection_probability: float,
    clutter_intensity: float,
    prune_log_weight_delta: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    events = _events(rf_measurements, radar)
    if not events:
        return [], _empty_selected_radar(radar)

    initial_measurement = _initial_measurement(
        events[0],
        association="track-bank",
        covariance=covariance,
        stable_anchor_by_key=None,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial_measurement is None:
        return [], _empty_selected_radar(radar)

    tracker = _initial_mht_tracker(
        initial_measurement,
        max_global_hypotheses=max_global_hypotheses,
        max_assignments_per_hypothesis=max_assignments_per_hypothesis,
        max_candidates_per_track=max_candidates_per_track,
        gate_probability=gate_probability,
        detection_probability=detection_probability,
        clutter_intensity=clutter_intensity,
        prune_log_weight_delta=prune_log_weight_delta,
    )
    current_time_s = float(initial_measurement.time_s)
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []

    for event in events:
        time_s = float(event["time_s"])
        _predict_mht_to(
            tracker,
            current_time_s=current_time_s,
            target_time_s=time_s,
            acceleration_std_mps2=acceleration_std_mps2,
        )
        current_time_s = time_s

        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = _deterministic_update_mht_hypotheses(
                tracker,
                measurement,
                gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=gate_probabilities_by_source,
                    gate_thresholds_by_source=gate_thresholds_by_source,
                ),
                safety_gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=safety_gate_probabilities_by_source,
                    gate_thresholds_by_source=safety_gate_thresholds_by_source,
                ),
                max_residual_norm=_max_residual_norm_for_measurement(
                    measurement,
                    max_residual_norms_by_source=max_residual_norms_by_source,
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
                _mht_record(
                    measurement=measurement,
                    tracker=tracker,
                    diagnostics=diagnostics,
                    association_mode="track-bank",
                )
            )
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
        if candidates.empty:
            continue

        pre_update_nis = _nis_scored_candidates(
            candidates,
            _tracker_from_best_mht_hypothesis(tracker, current_time_s, acceleration_std_mps2),
            covariance,
        )
        measurements = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float).T
        covariances = np.repeat(covariance[:, :, None], measurements.shape[1], axis=2)
        tracker.update_linear(measurements, measurement_matrix(3), covariances)

        selected = _selected_row_from_best_mht_assignment(
            candidates,
            pre_update_nis,
            tracker,
        )
        if selected is not None:
            selected_rows.append(selected)

        measurement = _mht_radar_measurement(time_s=time_s, selected=selected, tracker=tracker)
        records.append(
            _mht_record(
                measurement=measurement,
                tracker=tracker,
                diagnostics=_mht_radar_diagnostics(time_s, selected),
                track_id=None if selected is None else _optional_track_id(selected),
                association_nis=None if selected is None else _optional_float(selected.get("association_nis")),
                association_score=None
                if selected is None
                else _optional_float(selected.get("association_score")),
                association_mode="track-bank",
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


class _MHTStateView:
    def __init__(self, filter_obj: KalmanFilter) -> None:
        self._filter = filter_obj

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self._filter.get_point_estimate(), dtype=float).reshape(6)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(self._filter.filter_state.C, dtype=float).reshape(6, 6)


def _initial_mht_tracker(
    initial_measurement: TrackingMeasurement,
    *,
    max_global_hypotheses: int,
    max_assignments_per_hypothesis: int,
    max_candidates_per_track: int,
    gate_probability: float,
    detection_probability: float,
    clutter_intensity: float,
    prune_log_weight_delta: float,
) -> MultiHypothesisTracker:
    position = np.asarray(initial_measurement.vector, dtype=float).reshape(-1)
    if position.size == 2:
        position = np.array([position[0], position[1], 0.0])
    state = np.zeros(6)
    state[:3] = position
    state_covariance = np.diag([50.0**2, 50.0**2, 50.0**2, 15.0**2, 15.0**2, 15.0**2])
    return MultiHypothesisTracker(
        initial_prior=[KalmanFilter((state, state_covariance))],
        association_param={
            "gating_probability": float(gate_probability),
            "detection_probability": float(detection_probability),
            "clutter_intensity": float(clutter_intensity),
            "max_global_hypotheses": int(max_global_hypotheses),
            "max_hypotheses_per_global_hypothesis": int(max_assignments_per_hypothesis),
            "max_measurements_per_track": int(max_candidates_per_track),
            "prune_log_weight_delta": float(prune_log_weight_delta),
        },
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )


def _predict_mht_to(
    tracker: MultiHypothesisTracker,
    *,
    current_time_s: float,
    target_time_s: float,
    acceleration_std_mps2: float,
) -> None:
    dt_s = float(target_time_s) - float(current_time_s)
    if dt_s < -1e-9:
        raise ValueError("measurements must be processed in chronological order")
    if dt_s <= 0.0:
        return
    tracker.predict_linear(
        constant_velocity_matrix(dt_s),
        white_acceleration_process_noise(dt_s, acceleration_std_mps2),
    )


def _deterministic_update_mht_hypotheses(
    tracker: MultiHypothesisTracker,
    measurement: TrackingMeasurement,
    *,
    gate_threshold: float | None,
    safety_gate_threshold: float | None,
    max_residual_norm: float | None,
    robust_update: str | None,
    inflation_alpha: float,
) -> TrackingUpdateDiagnostics:
    best_index = tracker.get_best_hypothesis_index()
    best_diagnostics: TrackingUpdateDiagnostics | None = None
    for hypothesis_index, filter_bank in enumerate(tracker.global_hypotheses):
        filter_obj = filter_bank[0]
        diagnostics = _update_filter_linear(
            filter_obj,
            measurement,
            gate_threshold=gate_threshold,
            safety_gate_threshold=safety_gate_threshold,
            max_residual_norm=max_residual_norm,
            robust_update=robust_update,
            inflation_alpha=inflation_alpha,
        )
        if hypothesis_index == best_index:
            best_diagnostics = diagnostics
    if best_diagnostics is None:
        raise RuntimeError("MHT track bank has no best hypothesis")
    return best_diagnostics


def _update_filter_linear(
    filter_obj: KalmanFilter,
    measurement: TrackingMeasurement,
    *,
    gate_threshold: float | None,
    safety_gate_threshold: float | None,
    max_residual_norm: float | None,
    robust_update: str | None,
    inflation_alpha: float,
) -> TrackingUpdateDiagnostics:
    state = np.asarray(filter_obj.get_point_estimate(), dtype=float).reshape(6)
    state_covariance = np.asarray(filter_obj.filter_state.C, dtype=float).reshape(6, 6)
    vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
    covariance = np.asarray(measurement.covariance, dtype=float)
    observation = measurement_matrix(vector.size)
    plan = plan_linear_measurement_update(
        mean=state,
        covariance_matrix=state_covariance,
        measurement_vector=vector,
        measurement_covariance=covariance,
        observation_matrix=observation,
        gate_threshold=gate_threshold,
        safety_gate_threshold=safety_gate_threshold,
        max_residual_norm=max_residual_norm,
        robust_update=robust_update,
        inflation_alpha=inflation_alpha,
    )

    if plan.accepted:
        filter_obj.update_linear(plan.vector, plan.observation, plan.covariance)

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
        inflation_alpha=float(inflation_alpha) if robust_update == "nis-inflate" else None,
        residual_norm_m=plan.residual_norm,
    )


def _selected_row_from_best_mht_assignment(
    candidates: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    tracker: MultiHypothesisTracker,
) -> pd.Series | None:
    best_index = tracker.get_best_hypothesis_index()
    history = tracker.global_hypothesis_histories[best_index]
    if not history:
        return None
    assignment = history[-1]
    if not assignment or int(assignment[0]) < 0:
        return None
    measurement_index = int(assignment[0])
    selected = candidates.iloc[measurement_index].copy()
    scored = scored_candidates.iloc[measurement_index]
    selected["association_mode"] = "track-bank"
    selected["association_action"] = "mht_assigned"
    selected["association_nis"] = float(scored["association_nis"])
    selected["association_score"] = _mht_best_negative_log_weight(tracker)
    selected["association_candidate_rows"] = int(len(candidates))
    selected["association_hypothesis_count"] = int(tracker.get_number_of_global_hypotheses())
    selected["association_best_weight"] = _mht_best_weight(tracker)
    selected["association_weight_margin"] = _mht_weight_margin(tracker)
    return selected


def _mht_radar_measurement(
    *,
    time_s: float,
    selected: pd.Series | None,
    tracker: MultiHypothesisTracker,
) -> TrackingMeasurement:
    best_filter = tracker.get_best_hypothesis()[0]
    state = np.asarray(best_filter.get_point_estimate(), dtype=float).reshape(6)
    covariance = np.asarray(best_filter.filter_state.C, dtype=float).reshape(6, 6)
    vector = state[:3] if selected is None else selected[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return TrackingMeasurement(
        time_s=time_s,
        vector=vector,
        covariance=covariance[:3, :3],
        source="radar",
    )


def _mht_radar_diagnostics(
    time_s: float,
    selected: pd.Series | None,
) -> TrackingUpdateDiagnostics:
    return TrackingUpdateDiagnostics(
        time_s=float(time_s),
        source="radar",
        measurement_dim=3,
        accepted=selected is not None,
        update_action="mht_assigned" if selected is not None else "mht_missed",
        nis=float("nan") if selected is None else float(selected["association_nis"]),
        gate_threshold=None,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )


def _mht_record(
    measurement: TrackingMeasurement,
    tracker: MultiHypothesisTracker,
    diagnostics: TrackingUpdateDiagnostics,
    *,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    best_filter = tracker.get_best_hypothesis()[0]
    state = np.asarray(best_filter.get_point_estimate(), dtype=float).reshape(6)
    covariance = np.asarray(best_filter.filter_state.C, dtype=float).reshape(6, 6)
    record = {
        "time_s": float(measurement.time_s),
        "source": measurement.source,
        "state": state.copy(),
        "covariance": covariance.copy(),
        "hypothesis_count": int(tracker.get_number_of_global_hypotheses()),
        "best_hypothesis_weight": _mht_best_weight(tracker),
        "hypothesis_weight_margin": _mht_weight_margin(tracker),
        "hypotheses": _mht_hypothesis_snapshot(tracker, float(measurement.time_s)),
        **diagnostics.to_record(),
    }
    if track_id is not None:
        record["track_id"] = track_id
    if association_nis is not None:
        record["association_nis"] = association_nis
    if association_score is not None:
        record["association_score"] = association_score
    if association_mode is not None:
        record["association_mode"] = association_mode
    return record


def _mht_hypothesis_snapshot(
    tracker: MultiHypothesisTracker,
    time_s: float,
) -> list[dict[str, float | int]]:
    weights = tracker.get_global_hypothesis_weights()
    order = np.argsort(-weights)
    rows: list[dict[str, float | int]] = []
    for rank, hypothesis_index in enumerate(order):
        filter_obj = tracker.global_hypotheses[int(hypothesis_index)][0]
        state = np.asarray(filter_obj.get_point_estimate(), dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(time_s),
                "rank": int(rank),
                "hypothesis_index": int(hypothesis_index),
                "weight": float(weights[int(hypothesis_index)]),
                "east_m": float(state[0]),
                "north_m": float(state[1]),
                "up_m": float(state[2]),
                "v_east_mps": float(state[3]),
                "v_north_mps": float(state[4]),
                "v_up_mps": float(state[5]),
            }
        )
    return rows


def _mht_best_weight(tracker: MultiHypothesisTracker) -> float:
    weights = tracker.get_global_hypothesis_weights()
    return float(np.max(weights)) if len(weights) else float("nan")


def _mht_weight_margin(tracker: MultiHypothesisTracker) -> float:
    weights = np.sort(tracker.get_global_hypothesis_weights())[::-1]
    if len(weights) < 2:
        return float(weights[0]) if len(weights) else float("nan")
    return float(weights[0] - weights[1])


def _mht_best_negative_log_weight(tracker: MultiHypothesisTracker) -> float:
    weight = max(_mht_best_weight(tracker), 1e-300)
    return float(-np.log(weight))


def _tracker_from_best_mht_hypothesis(
    tracker: MultiHypothesisTracker,
    current_time_s: float,
    acceleration_std_mps2: float,
) -> _MHTStateView:
    del current_time_s, acceleration_std_mps2
    return _MHTStateView(tracker.get_best_hypothesis()[0])


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _select_stable_radar_segments(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None,
    catprob_threshold: float | None,
    min_segment_frames: int,
    max_transition_speed_mps: float,
) -> pd.DataFrame:
    """Select stitched high-confidence Fortem track segments for sparse updates."""

    pool = _range_candidate_pool(radar, range_gate_m)
    pool = _catprob_candidate_pool(pool, catprob_threshold)
    if pool.empty or "track_id" not in pool.columns:
        return _empty_selected_radar(radar)

    segments = _stable_track_segments(pool, min_segment_frames=min_segment_frames)
    if not segments:
        return _empty_selected_radar(radar)
    selected_segments = _stitch_segments(
        segments,
        max_transition_speed_mps=max_transition_speed_mps,
    )
    if not selected_segments:
        return _empty_selected_radar(radar)

    selected = pd.concat([segment.frame for segment in selected_segments], ignore_index=True)
    selected["association_mode"] = "stable-segments"
    selected["association_action"] = "stable_segment_anchor"
    selected["association_segment_count"] = int(len(selected_segments))
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _stable_track_segments(
    radar: pd.DataFrame,
    *,
    min_segment_frames: int,
) -> list[_TrackSegment]:
    segments: list[_TrackSegment] = []
    for track_id, track_rows in radar.groupby("track_id", sort=True):
        ordered = track_rows.sort_values(
            ["frame_index" if "frame_index" in track_rows.columns else "time_s", "time_s"]
        )
        frame_values = (
            pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(dtype=float)
            if "frame_index" in ordered.columns
            else ordered["time_s"].to_numpy(dtype=float)
        )
        splits = np.r_[
            0,
            np.where(np.diff(frame_values) > _segment_gap_threshold(frame_values))[0] + 1,
            len(ordered),
        ]
        for start, end in zip(splits[:-1], splits[1:]):
            frame = ordered.iloc[int(start) : int(end)].copy()
            if len(frame) < int(min_segment_frames):
                continue
            positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
            times = frame["time_s"].to_numpy(dtype=float)
            catprob = (
                pd.to_numeric(frame["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
                if "cat_prob_uav" in frame.columns
                else np.ones(len(frame), dtype=float)
            )
            mean_catprob = float(np.nanmean(catprob))
            if not np.isfinite(mean_catprob):
                mean_catprob = 0.0
            segments.append(
                _TrackSegment(
                    frame=frame,
                    track_id=int(track_id),
                    start_time_s=float(times[0]),
                    end_time_s=float(times[-1]),
                    start_position_m=positions[0],
                    end_position_m=positions[-1],
                    frames=int(len(frame)),
                    mean_catprob=mean_catprob,
                )
            )
    return sorted(segments, key=lambda item: (item.start_time_s, -item.score))


def _segment_gap_threshold(frame_values: np.ndarray) -> float:
    values = np.sort(np.asarray(frame_values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("inf")
    diffs = np.diff(values)
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if _integer_like(values):
        return 1.5
    return 1.5 * float(np.median(positive))


def _stitch_segments(
    segments: list[_TrackSegment],
    *,
    max_transition_speed_mps: float,
) -> list[_TrackSegment]:
    ordered = sorted(segments, key=lambda item: (item.start_time_s, item.end_time_s))
    best_paths: list[list[_TrackSegment]] = []
    best_scores: list[float] = []
    for segment in ordered:
        best_path = [segment]
        best_score = segment.score
        for index, previous in enumerate(ordered[: len(best_paths)]):
            if not _segments_can_follow(
                previous,
                segment,
                max_transition_speed_mps=max_transition_speed_mps,
            ):
                continue
            score = best_scores[index] + segment.score
            if score > best_score:
                best_score = score
                best_path = [*best_paths[index], segment]
        best_paths.append(best_path)
        best_scores.append(best_score)
    if not best_paths:
        return []
    return best_paths[int(np.argmax(best_scores))]


def _segments_can_follow(
    previous: _TrackSegment,
    current: _TrackSegment,
    *,
    max_transition_speed_mps: float,
) -> bool:
    if current.start_time_s <= previous.end_time_s:
        return False
    dt_s = current.start_time_s - previous.end_time_s
    if dt_s <= 0.0:
        return False
    distance_m = float(np.linalg.norm(current.start_position_m - previous.end_position_m))
    return distance_m / dt_s <= float(max_transition_speed_mps)


def _range_candidate_pool(candidates: pd.DataFrame, range_gate_m: float | None) -> pd.DataFrame:
    if candidates.empty or range_gate_m is None:
        return candidates
    ranges = _candidate_ranges_m(candidates)
    pool = candidates.loc[np.isfinite(ranges) & (ranges <= float(range_gate_m))].copy()
    pool["association_range_gate_m"] = float(range_gate_m)
    return pool


def _candidate_ranges_m(candidates: pd.DataFrame) -> np.ndarray:
    if "range_m" in candidates.columns:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(ranges).any():
            return ranges
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return np.linalg.norm(positions, axis=1)


def _radar_event_key(event: dict[str, object]) -> object:
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if "frame_index" in candidates.columns:
        values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return ("frame_index", int(values.iloc[0]))
    time_s = event.get("time_s")
    if time_s is None:
        time_s = float(candidates["time_s"].median())
    return ("time_s", round(float(time_s), 9))


def _radar_row_key(row: pd.Series) -> object:
    if "frame_index" in row.index and np.isfinite(float(row["frame_index"])):
        return ("frame_index", int(row["frame_index"]))
    return ("time_s", round(float(row["time_s"]), 9))


def _integer_like(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.allclose(finite, np.round(finite)))


def _initial_measurement(
    event: dict[str, object],
    *,
    association: str,
    covariance: np.ndarray,
    stable_anchor_by_key: dict[object, pd.Series] | None = None,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> TrackingMeasurement | None:
    if event["kind"] == "rf":
        measurement = event["measurement"]
        assert isinstance(measurement, TrackingMeasurement)
        return measurement
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if association == "oracle-nearest-truth":
        selected = _oracle_nearest_truth(
            candidates,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
    elif association == "stable-segments":
        selected = (
            None
            if stable_anchor_by_key is None
            else stable_anchor_by_key.get(_radar_event_key(event))
        )
    else:
        selected = _highest_catprob(candidates)
    if selected is None:
        return None
    return _radar_row_to_measurement(selected, covariance)


def _select_radar_candidate(
    candidates: pd.DataFrame,
    *,
    association: str,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
    truth: pd.DataFrame | None,
    current_track_id: int | None,
    track_switch_nis_ratio: float,
    candidate_catprob_threshold: float | None,
    geometry_velocity_std_mps: float,
    geometry_velocity_weight: float,
    geometry_switch_penalty: float,
    geometry_catprob_weight: float,
    pda_nis_temperature: float,
    pda_catprob_exponent: float,
    stable_anchor_by_key: dict[object, pd.Series] | None = None,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.Series | None:
    if candidates.empty:
        return None
    if association == "oracle-nearest-truth":
        return _oracle_nearest_truth(
            candidates,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
    if association == "stable-segments":
        if stable_anchor_by_key is None:
            return None
        selected = stable_anchor_by_key.get(_radar_event_key({"candidates": candidates}))
        if selected is None:
            return None
        selected = selected.copy()
        selected["association_mode"] = "stable-segments"
        selected["association_action"] = "stable_segment_update"
        selected["association_candidate_rows"] = int(len(candidates))
        return selected

    candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    if candidates.empty:
        return None
    scored = _nis_scored_candidates(candidates, tracker, covariance)
    if scored.empty:
        return None
    if association == "geometry-score":
        geometry_scored = _geometry_scored_candidates(
            scored,
            tracker=tracker,
            current_track_id=current_track_id,
            velocity_std_mps=geometry_velocity_std_mps,
            velocity_weight=geometry_velocity_weight,
            switch_penalty=geometry_switch_penalty,
            catprob_weight=geometry_catprob_weight,
        )
        best = geometry_scored.loc[geometry_scored["association_score"].idxmin()].copy()
        best["association_action"] = "geometry_score"
        return best
    if association == "pda-mixture":
        return _pda_mixture_candidate(
            scored,
            base_covariance=covariance,
            nis_temperature=pda_nis_temperature,
            catprob_exponent=pda_catprob_exponent,
        )
    best = scored.loc[scored["association_nis"].idxmin()].copy()
    if association == "prediction-nis" or current_track_id is None:
        return best
    if association != "track-continuity":
        raise ValueError(f"unknown radar association mode {association!r}")

    current = scored.loc[scored["track_id"] == current_track_id]
    if current.empty:
        return best
    current_best = current.loc[current["association_nis"].idxmin()].copy()
    if int(best["track_id"]) == current_track_id:
        return best
    if float(best["association_nis"]) < float(current_best["association_nis"]) * float(
        track_switch_nis_ratio
    ):
        return best
    current_best["association_action"] = "kept_track"
    return current_best


def _oracle_nearest_truth(
    candidates: pd.DataFrame,
    *,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.Series | None:
    if truth is None or truth.empty:
        return None
    truth_xyz = _nearest_truth_position(
        truth,
        time_s=float(candidates["time_s"].median()),
        max_delta_s=float(truth_time_gate_s),
    )
    if truth_xyz is None:
        return None
    candidate_xyz = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
    best_position = int(np.argmin(errors))
    if float(errors[best_position]) > float(truth_gate_m):
        return None
    selected = candidates.iloc[best_position].copy()
    selected["association_nis"] = float(errors[best_position])
    selected["association_truth_error_m"] = float(errors[best_position])
    selected["association_mode"] = "oracle-nearest-truth"
    selected["association_candidate_rows"] = int(len(candidates))
    return selected


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    candidate_catprob_threshold: float | None,
) -> pd.DataFrame:
    if candidate_catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    threshold = float(candidate_catprob_threshold)
    keep = catprob >= threshold
    pool = candidates.loc[keep].copy()
    pool["association_catprob_threshold"] = threshold
    pool["association_catprob_fallback"] = False
    return pool


def _highest_catprob(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    selected: pd.Series
    if "cat_prob_uav" in candidates.columns:
        scores = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(-np.inf)
        selected = candidates.loc[scores.idxmax()].copy()
    else:
        selected = candidates.iloc[0].copy()
    selected["association_mode"] = "bootstrap-catprob"
    selected["association_candidate_rows"] = int(len(candidates))
    return selected


def _nis_scored_candidates(
    candidates: pd.DataFrame,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.iloc[0:0].copy()
    observation = measurement_matrix(3)
    state_position = observation @ tracker.state
    innovation_covariance = observation @ tracker.covariance_matrix @ observation.T + covariance
    try:
        precision = np.linalg.inv(innovation_covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(innovation_covariance)
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = vectors - state_position
    scored = candidates.copy()
    scored["association_nis"] = np.einsum("ij,jk,ik->i", residuals, precision, residuals)
    scored["association_candidate_rows"] = int(len(candidates))
    return scored


def _geometry_scored_candidates(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncConstantVelocityKalmanTracker,
    current_track_id: int | None,
    velocity_std_mps: float,
    velocity_weight: float,
    switch_penalty: float,
    catprob_weight: float,
) -> pd.DataFrame:
    scored = candidates.copy()
    velocity_nis = _candidate_velocity_nis(scored, tracker.state[3:6], velocity_std_mps)
    scored["association_velocity_nis"] = velocity_nis
    scored["association_velocity_penalty"] = float(velocity_weight) * velocity_nis
    scored["association_switch_penalty"] = _track_switch_penalty(
        scored,
        current_track_id=current_track_id,
        switch_penalty=switch_penalty,
    )
    scored["association_catprob_penalty"] = _catprob_penalty(scored, catprob_weight)
    scored["association_score"] = (
        scored["association_nis"]
        + scored["association_velocity_penalty"]
        + scored["association_switch_penalty"]
        + scored["association_catprob_penalty"]
    )
    return scored


def _candidate_velocity_nis(
    candidates: pd.DataFrame,
    predicted_velocity_enu_mps: np.ndarray,
    velocity_std_mps: float,
) -> np.ndarray:
    required = {"velocity_east_mps", "velocity_north_mps", "velocity_down_mps"}
    if not required.issubset(candidates.columns):
        return np.zeros(len(candidates), dtype=float)
    velocities = np.column_stack(
        [
            pd.to_numeric(candidates["velocity_east_mps"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(candidates["velocity_north_mps"], errors="coerce").to_numpy(dtype=float),
            -pd.to_numeric(candidates["velocity_down_mps"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    finite = np.isfinite(velocities).all(axis=1)
    residuals = velocities - np.asarray(predicted_velocity_enu_mps, dtype=float).reshape(3)
    velocity_nis = np.sum((residuals / float(velocity_std_mps)) ** 2, axis=1)
    return np.where(finite, velocity_nis, 0.0)


def _track_switch_penalty(
    candidates: pd.DataFrame,
    *,
    current_track_id: int | None,
    switch_penalty: float,
) -> np.ndarray:
    if current_track_id is None or "track_id" not in candidates.columns:
        return np.zeros(len(candidates), dtype=float)
    track_ids = pd.to_numeric(candidates["track_id"], errors="coerce").to_numpy(dtype=float)
    switches = np.zeros(len(candidates), dtype=bool)
    finite = np.isfinite(track_ids)
    switches[finite] = track_ids[finite].astype(int) != int(current_track_id)
    return np.where(switches, float(switch_penalty), 0.0)


def _catprob_penalty(candidates: pd.DataFrame, catprob_weight: float) -> np.ndarray:
    if "cat_prob_uav" not in candidates.columns:
        return np.zeros(len(candidates), dtype=float)
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(1.0)
    catprob = np.clip(catprob.to_numpy(dtype=float), 0.0, 1.0)
    return float(catprob_weight) * (1.0 - catprob) ** 2


def _pda_mixture_candidate(
    candidates: pd.DataFrame,
    *,
    base_covariance: np.ndarray,
    nis_temperature: float,
    catprob_exponent: float,
) -> pd.Series:
    weights = _pda_weights(
        candidates,
        nis_temperature=nis_temperature,
        catprob_exponent=catprob_exponent,
    )
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    mean = weights @ vectors
    residuals = vectors - mean
    spread = residuals.T @ (residuals * weights[:, None])
    covariance = np.asarray(base_covariance, dtype=float) + spread

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
    selected["association_weight_entropy"] = _weight_entropy(weights)
    selected["association_effective_candidates"] = float(1.0 / np.sum(weights**2))
    selected["association_best_track_id"] = _optional_track_id(selected)
    selected["association_position_spread_trace_m2"] = float(np.trace(spread))
    selected["association_cov_ee"] = float(covariance[0, 0])
    selected["association_cov_nn"] = float(covariance[1, 1])
    selected["association_cov_uu"] = float(covariance[2, 2])
    selected["association_cov_en"] = float(covariance[0, 1])
    selected["association_cov_eu"] = float(covariance[0, 2])
    selected["association_cov_nu"] = float(covariance[1, 2])
    return selected


def _pda_weights(
    candidates: pd.DataFrame,
    *,
    nis_temperature: float,
    catprob_exponent: float,
) -> np.ndarray:
    nis = pd.to_numeric(candidates["association_nis"], errors="coerce").to_numpy(dtype=float)
    log_weights = -0.5 * nis / float(nis_temperature)
    if "cat_prob_uav" in candidates.columns and catprob_exponent > 0.0:
        catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(1e-3)
        catprob = np.clip(catprob.to_numpy(dtype=float), 1e-3, 1.0)
        log_weights = log_weights + float(catprob_exponent) * np.log(catprob)
    log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        return np.full(len(candidates), 1.0 / len(candidates))
    weights = np.exp(log_weights - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(candidates), 1.0 / len(candidates))
    return weights / total


def _weight_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _radar_row_to_measurement(row: pd.Series, covariance: np.ndarray) -> TrackingMeasurement:
    row_covariance = _row_covariance(row)
    return TrackingMeasurement(
        time_s=float(row["time_s"]),
        vector=np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]),
        covariance=covariance if row_covariance is None else row_covariance,
        source="radar",
    )


def _row_covariance(row: pd.Series) -> np.ndarray | None:
    columns = [
        "association_cov_ee",
        "association_cov_nn",
        "association_cov_uu",
        "association_cov_en",
        "association_cov_eu",
        "association_cov_nu",
    ]
    if not all(column in row for column in columns):
        return None
    values = [float(row[column]) for column in columns]
    if not np.isfinite(values).all():
        return None
    ee, nn, uu, en, eu, nu = values
    return np.array(
        [
            [ee, en, eu],
            [en, nn, nu],
            [eu, nu, uu],
        ],
        dtype=float,
    )


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> np.ndarray | None:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    if truth_times.size == 0:
        return None
    insertion = np.searchsorted(truth_times, float(time_s))
    right = int(np.clip(insertion, 0, truth_times.size - 1))
    left = int(np.clip(insertion - 1, 0, truth_times.size - 1))
    nearest = right if abs(truth_times[right] - time_s) < abs(truth_times[left] - time_s) else left
    if abs(truth_times[nearest] - time_s) > max_delta_s:
        return None
    return truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[nearest]


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


def _max_residual_norm_for_measurement(
    measurement: TrackingMeasurement,
    *,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> float | None:
    return max_residual_norm_for_measurement(
        measurement,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )


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


def _record(
    measurement: TrackingMeasurement,
    tracker: AsyncConstantVelocityKalmanTracker,
    diagnostics: Any,
    *,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": measurement.time_s,
        "source": measurement.source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        **diagnostics.to_record(),
    }
    if track_id is not None:
        record["track_id"] = track_id
    if association_nis is not None:
        record["association_nis"] = association_nis
    if association_score is not None:
        record["association_score"] = association_score
    if association_mode is not None:
        record["association_mode"] = association_mode
    return record


def _selected_rows_frame(radar: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    if not rows:
        return _empty_selected_radar(radar)
    selected = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _empty_selected_radar(radar: pd.DataFrame) -> pd.DataFrame:
    selected = radar.iloc[0:0].copy()
    for column in (
        "association_mode",
        "association_nis",
        "association_score",
        "association_candidate_rows",
        "association_effective_candidates",
        "association_weight_max",
        "association_position_spread_trace_m2",
    ):
        selected[column] = []
    return selected


def _optional_track_id(row: pd.Series) -> int | None:
    value = row.get("track_id")
    if value is None or not np.isfinite(float(value)):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
