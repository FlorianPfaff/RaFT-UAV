"""Learned sequence-level radar tracklet association.

This module combines the existing truth-free tracklet-Viterbi dynamic program
with the learned per-candidate radar likelihood.  The learned model supplies a
data-driven unary cost, while the existing Viterbi transition model still keeps
track-ID continuity, motion feasibility, missed detections, range penalties,
and RF-anchor reacquisition behavior.

The goal is to remove the most brittle hand-tuned part of the current best
association path without changing the downstream Kalman/IMM replay contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.learned_radar_likelihood import (
    LearnedRadarAssociationModel,
    STATEFUL_COST_METADATA_KEY,
    radar_association_feature_frame,
)
from raft_uav.baselines.radar_association import (
    _empty_selected_radar,
    _events,
    _initial_measurement,
    _selected_rows_frame,
)
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    RadarCovarianceFn,
    _AnchorState,
    _ViterbiNode,
    _build_rf_anchor_states,
    _candidate_cost_terms,
    _first_rf_bootstrap_index,
    _optional_track_id,
    _radar_covariance_for_row,
    _radar_event_key,
    _replay_selected_tracklet_path,
    _row_position,
    _row_velocity,
    _selected_rows_from_viterbi_path,
    _transition_cost,
)
from raft_uav.baselines.tracklet_viterbi_retention import (
    _candidate_pool_for_retention,
    _catprob_threshold_penalty,
    _retain_top_and_track_representatives,
    _track_support_by_event_prefix,
    _track_support_cost,
    _write_track_support_diagnostics,
)


def run_async_cv_baseline_with_learned_tracklet_viterbi_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    model: LearnedRadarAssociationModel | str | Path,
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
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
    learned_unary_weight: float = 1.0,
    hand_unary_weight: float = 0.25,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run fusion after learned-likelihood tracklet-Viterbi selection.

    ``learned_unary_weight`` scales the learned negative log-likelihood.
    ``hand_unary_weight`` keeps a small amount of the original RF/range/class
    hand prior so the model is still well behaved early in a flight and under
    out-of-distribution frames.
    """

    if learned_unary_weight < 0.0:
        raise ValueError("learned_unary_weight must be nonnegative")
    if hand_unary_weight < 0.0:
        raise ValueError("hand_unary_weight must be nonnegative")

    learned_model = (
        model
        if isinstance(model, LearnedRadarAssociationModel)
        else LearnedRadarAssociationModel.load(model)
    )
    cfg = config or TrackletViterbiAssociationConfig()
    cfg = _config_with_learned_transition_costs(cfg, learned_model)
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _empty_selected_radar(radar)
    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        return [], _empty_selected_radar(radar)
    events = events[bootstrap_index:]
    initial = _initial_measurement(
        events[0],
        association="learned-tracklet-viterbi",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial is None:
        return [], _empty_selected_radar(radar)

    anchors = _build_rf_anchor_states(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )
    selected = _select_learned_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        model=learned_model,
        learned_unary_weight=learned_unary_weight,
        track_support_by_event=_track_support_by_event_prefix(events),
        hand_unary_weight=hand_unary_weight,
        radar_covariance_fn=radar_covariance_fn,
    )
    records, accepted = _replay_selected_tracklet_path(
        events=events,
        selected_rows=selected,
        initial_measurement=initial,
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        radar_covariance_fn=radar_covariance_fn,
    )
    return records, _selected_rows_frame(radar, accepted)


def _select_learned_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    model: LearnedRadarAssociationModel,
    learned_unary_weight: float,
    hand_unary_weight: float,
    track_support_by_event: Mapping[int, Mapping[int, Mapping[str, float]]] | None = None,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> list[pd.Series]:
    frames = [
        _learned_nodes_for_radar_frame(
            event_index=i,
            candidates=event["candidates"],
            anchor=anchors.get(i),
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
            model=model,
            learned_unary_weight=learned_unary_weight,
            track_support_by_id=(track_support_by_event or {}).get(i, {}),
            hand_unary_weight=hand_unary_weight,
            radar_covariance_fn=radar_covariance_fn,
        )
        for i, event in enumerate(events)
        if event["kind"] == "radar"
    ]
    if not frames:
        return []

    costs = [np.array([n.unary_cost + (config.missed_detection_cost if n.is_miss else 0.0) for n in frames[0]])]
    miss_streaks = [np.array([1 if n.is_miss else 0 for n in frames[0]], dtype=int)]
    parents = [np.full(len(frames[0]), -1, dtype=int)]
    for frame_index in range(1, len(frames)):
        previous, current = frames[frame_index - 1], frames[frame_index]
        current_costs = np.empty(len(current), dtype=float)
        current_miss_streaks = np.empty(len(current), dtype=int)
        current_parents = np.empty(len(current), dtype=int)
        for j, node in enumerate(current):
            transition = np.array(
                [
                    costs[-1][k]
                    + _transition_cost(
                        prev,
                        node,
                        config,
                        previous_miss_streak=int(miss_streaks[-1][k]),
                    )
                    for k, prev in enumerate(previous)
                ]
            )
            parent = int(np.argmin(transition))
            current_parents[j] = parent
            current_costs[j] = node.unary_cost + float(transition[parent])
            current_miss_streaks[j] = int(miss_streaks[-1][parent]) + 1 if node.is_miss else 0
        costs.append(current_costs)
        miss_streaks.append(current_miss_streaks)
        parents.append(current_parents)

    best = int(np.argmin(costs[-1]))
    path_cost = float(costs[-1][best])
    path: list[_ViterbiNode] = []
    for frame_index in range(len(frames) - 1, -1, -1):
        path.append(frames[frame_index][best])
        best = int(parents[frame_index][best])
        if best < 0:
            break
    path.reverse()
    return _selected_rows_from_viterbi_path(path, path_cost, config)


def _learned_nodes_for_radar_frame(
    *,
    event_index: int,
    candidates: pd.DataFrame,
    anchor: _AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    model: LearnedRadarAssociationModel,
    learned_unary_weight: float,
    hand_unary_weight: float,
    track_support_by_id: Mapping[int, Mapping[str, float]] | None = None,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> list[_ViterbiNode]:
    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    event_key = _radar_event_key(candidates)
    pool = _candidate_pool_for_retention(candidates, candidate_catprob_threshold, config)
    if pool.empty:
        return [_miss_node(event_index, event_key, time_s)]

    scored_rows: list[pd.Series] = []
    candidate_terms: list[tuple[np.ndarray, np.ndarray | None, float, float, float, float]] = []
    for _, row in pool.iterrows():
        position = _row_position(row)
        if position is None:
            continue
        row_covariance = _radar_covariance_for_row(
            row,
            covariance,
            radar_covariance_fn,
        )
        anchor_nis, catprob_cost, range_cost = _candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=row_covariance,
            config=config,
        )
        soft_threshold_cost = _catprob_threshold_penalty(
            row,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        track_support_cost, track_support = _track_support_cost(
            row,
            track_support_by_id=track_support_by_id or {},
            config=config,
        )
        hand_cost = (
            float(config.anchor_nis_weight) * anchor_nis
            + catprob_cost
            + range_cost
            + soft_threshold_cost
            + track_support_cost
        )
        enriched = row.copy()
        enriched["association_nis"] = float(anchor_nis)
        enriched["association_hand_unary_cost"] = float(hand_cost)
        enriched["association_catprob_soft_penalty"] = float(soft_threshold_cost)
        _write_track_support_diagnostics(enriched, track_support_cost, track_support)
        scored_rows.append(enriched)
        candidate_terms.append((position, _row_velocity(row), anchor_nis, catprob_cost, range_cost, hand_cost))

    if not scored_rows:
        return [_miss_node(event_index, event_key, time_s)]

    scored = pd.DataFrame(scored_rows).reset_index(drop=True)
    tracker_state = _feature_tracker_state(anchor, candidate_terms)
    features = radar_association_feature_frame(
        scored,
        tracker_state=tracker_state,
        current_track_id=None,
    )
    probabilities = np.clip(model.predict_proba_features(features), 1.0e-12, 1.0)
    learned_costs = -np.log(probabilities)

    scored_nodes: list[tuple[float, int, _ViterbiNode]] = []
    for row_index, row in scored.iterrows():
        position, velocity, anchor_nis, catprob_cost, range_cost, hand_cost = candidate_terms[row_index]
        learned_cost = float(learned_costs[row_index])
        unary_cost = float(learned_unary_weight) * learned_cost + float(hand_unary_weight) * float(hand_cost)
        selected = row.copy()
        selected["association_mode"] = "learned-tracklet-viterbi"
        selected["association_action"] = "learned_viterbi_selected"
        selected["association_learned_probability"] = float(probabilities[row_index])
        selected["association_learned_cost"] = learned_cost
        selected["association_score"] = unary_cost
        selected["association_learned_unary_weight"] = float(learned_unary_weight)
        selected["association_hand_unary_weight"] = float(hand_unary_weight)
        node = (
            _ViterbiNode(
                event_index=event_index,
                event_key=event_key,
                time_s=float(row.get("time_s", time_s)),
                row=selected,
                position=position,
                velocity=velocity,
                track_id=_optional_track_id(row.get("track_id")),
                unary_cost=unary_cost,
                anchor_nis=float(anchor_nis),
                catprob_cost=float(catprob_cost),
                range_cost=float(range_cost),
                has_anchor=bool(config.use_rf_anchor and anchor is not None),
            )
        )
        scored_nodes.append((float(unary_cost), int(row_index), node))
    nodes = _retain_top_and_track_representatives(scored_nodes, config)
    return nodes + [_miss_node(event_index, event_key, time_s)]


class _ConfigOverlay:
    """Read-only config overlay for learned transition-cost calibration."""

    def __init__(self, base: object, **overrides: float) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _config_with_learned_transition_costs(
    config: TrackletViterbiAssociationConfig,
    model: LearnedRadarAssociationModel,
) -> TrackletViterbiAssociationConfig:
    """Overlay empirical sequence costs stored in the association model metadata.

    The association trainer already estimates data-derived costs for missed
    detections, consecutive misses, missing track IDs, and Fortem track-ID
    switches.  Applying those costs here removes one of the most brittle
    hand-tuned parts of the learned tracklet decoder while keeping the protocol
    leakage-safe: the model and the metadata are fitted only on training folds.
    """

    transition_costs = dict(
        (getattr(model, "metadata", {}) or {}).get(STATEFUL_COST_METADATA_KEY) or {}
    )
    if not transition_costs:
        return config

    updates: dict[str, float] = {}
    for name in (
        "missed_detection_cost",
        "consecutive_miss_cost",
        "track_switch_cost",
        "missing_track_id_cost",
    ):
        value = _optional_nonnegative_float(transition_costs.get(name))
        if value is not None:
            updates[name] = value
    if not updates:
        return config
    return _ConfigOverlay(config, **updates)  # type: ignore[return-value]


def _optional_nonnegative_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _feature_tracker_state(
    anchor: _AnchorState | None,
    candidate_terms: list[tuple[np.ndarray, np.ndarray | None, float, float, float, float]],
) -> np.ndarray:
    if anchor is not None:
        return np.asarray(anchor.state, dtype=float).reshape(6)
    positions = np.vstack([item[0] for item in candidate_terms])
    velocities = [item[1] for item in candidate_terms if item[1] is not None]
    state = np.zeros(6, dtype=float)
    state[:3] = np.nanmedian(positions, axis=0)
    if velocities:
        state[3:6] = np.nanmedian(np.vstack(velocities), axis=0)
    return np.where(np.isfinite(state), state, 0.0)


def _miss_node(event_index: int, event_key: tuple[str, int | float], time_s: float) -> _ViterbiNode:
    return _ViterbiNode(
        event_index,
        event_key,
        time_s,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
        0.0,
        0.0,
        True,
    )
