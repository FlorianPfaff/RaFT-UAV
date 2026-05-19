"""Stateful learned radar association with Fortem track-ID persistence.

The per-frame learned association runner scores radar candidates independently.
This module keeps a small beam of single-target association hypotheses over
radar frames.  Each hypothesis has its own Kalman state and a discrete current
Fortem ``track_id`` mode.  Candidate likelihoods still come from the learned
logistic model, but the beam adds explicit penalties for switching or losing
track IDs and keeps missed-detection branches alive when all candidates look
bad.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatefulAssociationConfig:
    """Configuration for stateful learned radar association."""

    max_hypotheses: int = 16
    max_candidates_per_hypothesis: int = 6
    missed_detection_cost: float = 4.0
    consecutive_miss_cost: float = 0.5
    track_switch_cost: float = 3.0
    missing_track_id_cost: float = 1.0
    min_candidate_probability: float = 1.0e-9
    allow_missed_detection: bool = True
    lag_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive")
        if self.max_candidates_per_hypothesis < 1:
            raise ValueError("max_candidates_per_hypothesis must be positive")
        for name in (
            "missed_detection_cost",
            "consecutive_miss_cost",
            "track_switch_cost",
            "missing_track_id_cost",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < float(self.min_candidate_probability) <= 1.0:
            raise ValueError("min_candidate_probability must be in (0, 1]")
        if self.lag_s is not None and float(self.lag_s) < 0.0:
            raise ValueError("lag_s must be nonnegative or None")


@dataclass(frozen=True)
class RadarDecisionNode:
    """Persistent linked-list node for one radar-frame decision."""

    parent: "RadarDecisionNode | None"
    event_key: tuple[str, int | float]
    time_s: float
    selected: pd.Series | None


@dataclass
class _BeamHypothesis:
    tracker: Any
    log_cost: float
    current_track_id: int | None
    decision: RadarDecisionNode | None
    missed_frames: int = 0


def run_async_cv_baseline_with_stateful_learned_radar_association(
    *,
    rf_measurements: Iterable[Any],
    radar: pd.DataFrame,
    model: Any | str | Path,
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
    candidate_catprob_threshold: float | None = 0.5,
    config: StatefulAssociationConfig | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion with stateful learned radar association.

    Ground truth is not used here.  The learned model supplies per-candidate
    probabilities.  The beam carries multiple tracker states over time and adds
    explicit Fortem track-ID persistence costs before finally replaying the best
    selected radar path to produce the same record schema as the other baseline
    runners.
    """

    from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
    from raft_uav.baselines.learned_radar_likelihood import (
        LearnedRadarAssociationModel,
        score_radar_candidates_with_learned_likelihood,
    )
    from raft_uav.baselines.radar_association import (
        _catprob_candidate_pool,
        _empty_selected_radar,
        _events,
        _gate_threshold_for_measurement,
        _initial_observation,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _nis_scored_candidates,
        _radar_row_to_measurement,
        _record,
        _robust_update_for_measurement,
        _selected_rows_frame,
    )

    cfg = config or StatefulAssociationConfig()
    learned_model = (
        model
        if isinstance(model, LearnedRadarAssociationModel)
        else LearnedRadarAssociationModel.load(model)
    )
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    rf_measurements = list(rf_measurements)
    events = _events(rf_measurements, radar)
    if not events:
        return [], _empty_selected_radar(radar)

    initial_observation = _initial_observation(
        events[0],
        association="prediction-nis",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial_observation is None:
        return [], _empty_selected_radar(radar)

    initial_measurement = initial_observation.measurement
    initial_selected_row = initial_observation.selected_radar_row
    initial_current_track_id: int | None = None
    initial_decision: RadarDecisionNode | None = None
    if initial_selected_row is not None:
        initial_selected_row = initial_selected_row.copy()
        initial_selected_row["association_mode"] = "stateful-learned-likelihood"
        initial_selected_row["association_action"] = "stateful_bootstrap"
        initial_current_track_id = _optional_track_id(initial_selected_row.get("track_id"))
        initial_decision = RadarDecisionNode(
            parent=None,
            event_key=_selected_row_event_key(initial_selected_row),
            time_s=float(initial_selected_row["time_s"]),
            selected=initial_selected_row,
        )
    initial_tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    hypotheses = [
        _BeamHypothesis(
            tracker=initial_tracker,
            log_cost=0.0,
            current_track_id=initial_current_track_id,
            decision=initial_decision,
            missed_frames=0,
        )
    ]

    for event in events[1:]:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            for hypothesis in hypotheses:
                hypothesis.tracker.update(
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
            hypotheses = _prune_hypotheses(hypotheses, cfg.max_hypotheses)
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        time_s = float(event["time_s"])
        event_key = _radar_event_key(candidates)
        branches: list[_BeamHypothesis] = []

        for hypothesis in hypotheses:
            predicted_tracker = _clone_tracker(hypothesis.tracker)
            predicted_tracker.predict_to(time_s)

            pool = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
            if pool.empty:
                branches.append(
                    _missed_branch(hypothesis, predicted_tracker, cfg, event_key, time_s)
                )
                continue

            scored = _nis_scored_candidates(pool, predicted_tracker, covariance)
            if scored.empty:
                branches.append(
                    _missed_branch(hypothesis, predicted_tracker, cfg, event_key, time_s)
                )
                continue

            learned_scored = score_radar_candidates_with_learned_likelihood(
                scored,
                model=learned_model,
                tracker_state=predicted_tracker.state,
                current_track_id=hypothesis.current_track_id,
            )
            costed = _score_stateful_candidates(
                learned_scored,
                current_track_id=hypothesis.current_track_id,
                config=cfg,
            )
            ordered = costed.nsmallest(
                int(cfg.max_candidates_per_hypothesis),
                "stateful_association_cost",
            )

            for _, row in ordered.iterrows():
                selected = row.copy()
                tracker = _clone_tracker(predicted_tracker)
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
                if not diagnostics.accepted:
                    continue
                branch_cost = float(selected["stateful_association_cost"])
                total_cost = float(hypothesis.log_cost + branch_cost)
                selected["association_mode"] = "stateful-learned-likelihood"
                selected["association_action"] = "stateful_beam"
                selected["stateful_total_log_cost"] = total_cost
                selected["stateful_parent_log_cost"] = float(hypothesis.log_cost)
                selected["stateful_branch_log_cost"] = branch_cost
                selected["stateful_hypothesis_missed_frames"] = int(hypothesis.missed_frames)
                selected["association_candidate_rows"] = int(len(costed))

                track_id = _optional_track_id(selected.get("track_id"))
                next_track_id = (
                    track_id
                    if diagnostics.accepted and track_id is not None
                    else hypothesis.current_track_id
                )
                branches.append(
                    _BeamHypothesis(
                        tracker=tracker,
                        log_cost=total_cost,
                        current_track_id=next_track_id,
                        decision=RadarDecisionNode(
                            parent=hypothesis.decision,
                            event_key=event_key,
                            time_s=time_s,
                            selected=selected,
                        ),
                        missed_frames=0,
                    )
                )

            if cfg.allow_missed_detection:
                branches.append(
                    _missed_branch(hypothesis, predicted_tracker, cfg, event_key, time_s)
                )

        hypotheses = _prune_hypotheses(branches, cfg.max_hypotheses)
        hypotheses = _apply_fixed_lag_commitment(hypotheses, time_s, cfg.lag_s)
        if not hypotheses:
            return [], _empty_selected_radar(radar)

    best = min(hypotheses, key=lambda item: item.log_cost)
    selected_rows = _reconstruct_selected_rows(best.decision)
    records = _replay_stateful_radar_path(
        events=events,
        selected_rows=selected_rows,
        initial_measurement=initial_measurement,
        initial_selected_row=initial_selected_row,
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        tracker_cls=AsyncConstantVelocityKalmanTracker,
        record_fn=_record,
        gate_threshold_fn=_gate_threshold_for_measurement,
        max_residual_norm_fn=_max_residual_norm_for_measurement,
        robust_update_fn=_robust_update_for_measurement,
        inflation_alpha_fn=_inflation_alpha_for_measurement,
        radar_row_to_measurement_fn=_radar_row_to_measurement,
    )
    return records, _selected_rows_frame(radar, selected_rows)


def _clone_tracker(tracker: Any) -> Any:
    """Return an independent tracker copy for hypothesis branching."""

    try:
        state = tracker.state
        covariance = tracker.covariance_matrix
        cloned = tracker.__class__(
            initial_position=state.copy(),
            initial_time_s=float(tracker.current_time_s),
            acceleration_std_mps2=float(tracker.acceleration_std_mps2),
        )
        cloned.mean = state.copy()
        cloned.covariance = covariance.copy()
        cloned.filter = tracker.filter.__class__((cloned.mean.copy(), cloned.covariance.copy()))
        cloned._sync_from_filter()
        return cloned
    except Exception:
        return copy.deepcopy(tracker)


def _score_stateful_candidates(
    candidates: pd.DataFrame,
    *,
    current_track_id: int | None,
    config: StatefulAssociationConfig,
) -> pd.DataFrame:
    """Append stateful track-continuity costs to learned candidate scores."""

    scored = candidates.copy()
    branch_costs: list[float] = []
    switch_costs: list[float] = []
    missing_track_costs: list[float] = []
    for _, row in scored.iterrows():
        base = _candidate_negative_log_likelihood(row, config.min_candidate_probability)
        track_id = _optional_track_id(row.get("track_id"))
        switch_cost = 0.0
        missing_track_cost = 0.0
        if current_track_id is not None:
            if track_id is None:
                missing_track_cost = float(config.missing_track_id_cost)
            elif track_id != current_track_id:
                switch_cost = float(config.track_switch_cost)
        branch_costs.append(float(base + switch_cost + missing_track_cost))
        switch_costs.append(float(switch_cost))
        missing_track_costs.append(float(missing_track_cost))
    scored["stateful_association_cost"] = branch_costs
    scored["stateful_track_switch_cost"] = switch_costs
    scored["stateful_missing_track_id_cost"] = missing_track_costs
    return scored.sort_values("stateful_association_cost", kind="mergesort").reset_index(drop=True)


def _candidate_negative_log_likelihood(row: pd.Series, min_probability: float) -> float:
    score = _optional_float(row.get("association_score"))
    if score is not None and np.isfinite(score):
        return float(score)
    probability = _optional_float(row.get("association_learned_probability"))
    if probability is None or not np.isfinite(probability):
        return float(-math.log(float(min_probability)))
    return float(-math.log(max(float(probability), float(min_probability))))


def _missed_branch(
    hypothesis: _BeamHypothesis,
    predicted_tracker: Any,
    config: StatefulAssociationConfig,
    event_key: tuple[str, int | float],
    time_s: float,
) -> _BeamHypothesis:
    miss_cost = float(config.missed_detection_cost) + float(config.consecutive_miss_cost) * float(
        hypothesis.missed_frames
    )
    return _BeamHypothesis(
        tracker=predicted_tracker,
        log_cost=float(hypothesis.log_cost + miss_cost),
        current_track_id=hypothesis.current_track_id,
        decision=RadarDecisionNode(
            parent=hypothesis.decision,
            event_key=event_key,
            time_s=float(time_s),
            selected=None,
        ),
        missed_frames=int(hypothesis.missed_frames + 1),
    )


def _prune_hypotheses(
    hypotheses: list[_BeamHypothesis],
    max_hypotheses: int,
) -> list[_BeamHypothesis]:
    """Keep the lowest-cost hypotheses with deterministic tie breaking."""

    ordered = sorted(
        hypotheses,
        key=lambda item: (
            float(item.log_cost),
            int(item.missed_frames),
            -1 if item.current_track_id is None else int(item.current_track_id),
        ),
    )
    return ordered[: int(max_hypotheses)]


def _apply_fixed_lag_commitment(
    hypotheses: list[_BeamHypothesis],
    current_time_s: float,
    lag_s: float | None,
) -> list[_BeamHypothesis]:
    """Discard hypotheses that disagree with the best path outside the lag window."""

    if lag_s is None or not hypotheses:
        return hypotheses
    cutoff_s = float(current_time_s) - float(lag_s)
    if cutoff_s < 0.0:
        return hypotheses
    best = min(hypotheses, key=lambda item: item.log_cost)
    best_prefix = _decision_prefix(best.decision, cutoff_s)
    if not best_prefix:
        return hypotheses
    filtered = [
        hypothesis
        for hypothesis in hypotheses
        if _decision_prefix(hypothesis.decision, cutoff_s) == best_prefix
    ]
    return filtered if filtered else [best]


def _decision_prefix(
    node: RadarDecisionNode | None,
    cutoff_s: float,
) -> tuple[tuple[tuple[str, int | float], tuple[object, ...]], ...]:
    items: list[tuple[tuple[str, int | float], tuple[object, ...]]] = []
    current = node
    while current is not None:
        if current.time_s <= cutoff_s:
            items.append((current.event_key, _decision_signature(current.selected)))
        current = current.parent
    items.reverse()
    return tuple(items)


def _decision_signature(selected: pd.Series | None) -> tuple[object, ...]:
    if selected is None:
        return ("miss",)
    return (
        "select",
        _optional_track_id(selected.get("track_id")),
        _rounded_optional_float(selected.get("east_m")),
        _rounded_optional_float(selected.get("north_m")),
        _rounded_optional_float(selected.get("up_m")),
    )


def _reconstruct_selected_rows(node: RadarDecisionNode | None) -> list[pd.Series]:
    """Return selected radar rows from a decision linked list in time order."""

    rows: list[pd.Series] = []
    current = node
    while current is not None:
        if current.selected is not None:
            rows.append(current.selected)
        current = current.parent
    rows.reverse()
    return rows


def _replay_stateful_radar_path(
    *,
    events: list[dict[str, object]],
    selected_rows: list[pd.Series],
    initial_measurement: Any,
    initial_selected_row: pd.Series | None,
    acceleration_std_mps2: float,
    covariance: np.ndarray,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    tracker_cls: Any,
    record_fn: Any,
    gate_threshold_fn: Any,
    max_residual_norm_fn: Any,
    robust_update_fn: Any,
    inflation_alpha_fn: Any,
    radar_row_to_measurement_fn: Any,
) -> list[dict[str, object]]:
    from raft_uav.baselines.kalman import bootstrap_tracking_diagnostics

    selected_by_key = {_selected_row_event_key(row): row for row in selected_rows}
    tracker = tracker_cls(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []

    records.append(
        record_fn(
            initial_measurement,
            tracker,
            bootstrap_tracking_diagnostics(initial_measurement),
            track_id=None
            if initial_selected_row is None
            else _optional_track_id(initial_selected_row.get("track_id")),
            association_nis=None
            if initial_selected_row is None
            else _optional_float(initial_selected_row.get("association_nis")),
            association_score=None
            if initial_selected_row is None
            else _optional_float(initial_selected_row.get("association_score")),
            association_mode=None
            if initial_selected_row is None
            else "stateful-learned-likelihood",
        )
    )

    for event in events[1:]:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            diagnostics = tracker.update(
                measurement,
                gate_threshold=gate_threshold_fn(
                    measurement,
                    gate_probabilities_by_source=gate_probabilities_by_source,
                    gate_thresholds_by_source=gate_thresholds_by_source,
                ),
                safety_gate_threshold=gate_threshold_fn(
                    measurement,
                    gate_probabilities_by_source=safety_gate_probabilities_by_source,
                    gate_thresholds_by_source=safety_gate_thresholds_by_source,
                ),
                max_residual_norm=max_residual_norm_fn(
                    measurement,
                    max_residual_norms_by_source=max_residual_norms_by_source,
                ),
                robust_update=robust_update_fn(
                    measurement,
                    robust_update_by_source=robust_update_by_source,
                ),
                inflation_alpha=inflation_alpha_fn(
                    measurement,
                    inflation_alpha_by_source=inflation_alpha_by_source,
                ),
            )
            records.append(record_fn(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        selected = selected_by_key.get(_radar_event_key(candidates))
        if selected is None:
            continue
        measurement = radar_row_to_measurement_fn(selected, covariance)
        diagnostics = tracker.update(
            measurement,
            gate_threshold=gate_threshold_fn(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
            ),
            safety_gate_threshold=gate_threshold_fn(
                measurement,
                gate_probabilities_by_source=safety_gate_probabilities_by_source,
                gate_thresholds_by_source=safety_gate_thresholds_by_source,
            ),
            max_residual_norm=max_residual_norm_fn(
                measurement,
                max_residual_norms_by_source=max_residual_norms_by_source,
            ),
            robust_update=robust_update_fn(
                measurement,
                robust_update_by_source=robust_update_by_source,
            ),
            inflation_alpha=inflation_alpha_fn(
                measurement,
                inflation_alpha_by_source=inflation_alpha_by_source,
            ),
        )
        records.append(
            record_fn(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected.get("track_id")),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode="stateful-learned-likelihood",
            )
        )
    return records


def _radar_event_key(candidates: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in candidates.columns and not candidates.empty:
        value = _optional_float(candidates["frame_index"].iloc[0])
        if value is not None:
            return ("frame_index", int(value))
    if "time_s" not in candidates.columns or candidates.empty:
        return ("time_s", float("nan"))
    median_time = pd.to_numeric(candidates["time_s"], errors="coerce").median()
    return ("time_s", round(float(median_time), 9))


def _selected_row_event_key(row: pd.Series) -> tuple[str, int | float]:
    frame_index = _optional_float(row.get("frame_index"))
    if frame_index is not None:
        return ("frame_index", int(frame_index))
    time_s = _optional_float(row.get("time_s"))
    if time_s is None:
        return ("time_s", float("nan"))
    return ("time_s", round(float(time_s), 9))


def _rounded_optional_float(value: object, digits: int = 3) -> float | None:
    number = _optional_float(value)
    return None if number is None else round(float(number), int(digits))


def _optional_track_id(value: object) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
