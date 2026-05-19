"""Truth-free sequence-level radar tracklet association.

This module selects one coherent radar path over all radar frames and then
replays that path through the existing asynchronous CV Kalman baseline.  The
objective combines RF-anchor consistency, Fortem track-ID continuity, CV motion
feasibility, radar velocity consistency, class probability, range, and a
missed-detection branch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    bootstrap_tracking_diagnostics,
    constant_velocity_matrix,
    white_acceleration_process_noise,
)


@dataclass(frozen=True)
class TrackletViterbiAssociationConfig:
    """Configuration for sequence-level radar association."""

    max_candidates_per_frame: int = 8
    path_beam_width: int = 1
    replay_nis_weight: float = 0.0
    replay_rejection_cost: float = 0.0
    replay_roughness_weight: float = 0.0
    missed_detection_cost: float = 7.0
    consecutive_miss_cost: float = 1.0
    track_switch_cost: float = 8.0
    missing_track_id_cost: float = 1.0
    catprob_weight: float = 2.5
    anchor_nis_weight: float = 0.35
    transition_nis_weight: float = 1.0
    velocity_nis_weight: float = 0.15
    transition_position_std_m: float = 40.0
    transition_speed_std_mps: float = 18.0
    velocity_std_mps: float = 12.0
    max_speed_mps: float = 55.0
    max_speed_penalty: float = 10.0
    range_gate_m: float | None = 850.0
    range_gate_slack_m: float = 150.0
    range_penalty: float = 10.0
    use_rf_anchor: bool = True
    rf_anchor_mode: str = "causal"
    min_catprob: float = 1.0e-3
    association_reranker_path: str | Path | None = None
    association_reranker_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be positive")
        if self.path_beam_width < 1:
            raise ValueError("path_beam_width must be positive")
        positive = (
            "transition_position_std_m",
            "transition_speed_std_mps",
            "velocity_std_mps",
            "max_speed_mps",
        )
        nonnegative = (
            "missed_detection_cost",
            "consecutive_miss_cost",
            "track_switch_cost",
            "missing_track_id_cost",
            "catprob_weight",
            "anchor_nis_weight",
            "transition_nis_weight",
            "velocity_nis_weight",
            "max_speed_penalty",
            "range_gate_slack_m",
            "range_penalty",
            "replay_nis_weight",
            "replay_rejection_cost",
            "replay_roughness_weight",
            "association_reranker_weight",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.range_gate_m is not None and float(self.range_gate_m) <= 0.0:
            raise ValueError("range_gate_m must be positive or None")
        if self.rf_anchor_mode not in ("causal", "smoothed"):
            raise ValueError('rf_anchor_mode must be either "causal" or "smoothed"')
        if not 0.0 < float(self.min_catprob) <= 1.0:
            raise ValueError("min_catprob must be in (0, 1]")


@dataclass(frozen=True)
class _AnchorState:
    state: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class _ViterbiNode:
    event_index: int
    event_key: tuple[str, int | float]
    time_s: float
    row: pd.Series | None
    position: np.ndarray | None
    velocity: np.ndarray | None
    track_id: int | None
    unary_cost: float
    anchor_nis: float
    catprob_cost: float
    range_cost: float
    is_miss: bool = False
    reranker_cost: float = 0.0
    reranker_probability: float | None = None
    reranker_logit: float | None = None


@dataclass(frozen=True)
class _ViterbiPath:
    cost: float
    nodes: tuple[_ViterbiNode, ...]


def run_async_cv_baseline_with_tracklet_viterbi_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
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
    tracker_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion after Viterbi radar-tracklet selection."""

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_observation,
        _selected_rows_frame,
    )

    cfg = config or TrackletViterbiAssociationConfig()
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
    initial_observation = _initial_observation(
        events[0],
        association="tracklet-viterbi",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial_observation is None:
        return [], _empty_selected_radar(radar)
    initial = initial_observation.measurement

    anchors = _build_rf_anchor_states_for_config(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        config=cfg,
        tracker_factory=tracker_factory,
    )
    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
    )
    records, accepted = _replay_selected_tracklet_path(
        events=events,
        selected_rows=selected,
        initial_measurement=initial,
        initial_selected_row=initial_observation.selected_radar_row,
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        tracker_factory=tracker_factory,
    )
    return records, _selected_rows_frame(radar, accepted)


def _first_rf_bootstrap_index(events: list[dict[str, object]]) -> int | None:
    """Return the causal bootstrap index for non-oracle tracklet replay.

    The tracklet selector is truth-free and should not initialize from an
    arbitrary pre-RF radar candidate selected only by class probability.  If RF
    measurements are present, start at the first RF event and ignore earlier
    radar frames.  Radar-only inputs keep their historical radar bootstrap.
    """

    if not events:
        return None
    for index, event in enumerate(events):
        if event.get("kind") == "rf":
            return index
    return 0


def _build_rf_anchor_states(
    *,
    events: list[dict[str, object]],
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    tracker_factory: Callable[..., Any] | None = None,
) -> dict[int, _AnchorState]:
    """Return RF-only CV predictions at radar event indices."""

    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _robust_update_for_measurement,
    )

    tracker: Any | None = None
    anchors: dict[int, _AnchorState] = {}
    for event_index, event in enumerate(events):
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            if tracker is None:
                tracker = _make_tracker(
                    tracker_factory,
                    initial_position=measurement.vector,
                    initial_time_s=measurement.time_s,
                    acceleration_std_mps2=acceleration_std_mps2,
                )
                continue
            tracker.update(
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
        elif tracker is not None:
            tracker.predict_to(time_s)
            anchors[event_index] = _AnchorState(
                state=tracker.state.copy(),
                covariance=tracker.covariance_matrix.copy(),
            )
    return anchors


def _build_rf_anchor_states_for_config(
    *,
    events: list[dict[str, object]],
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    config: TrackletViterbiAssociationConfig,
    tracker_factory: Callable[..., Any] | None = None,
) -> dict[int, _AnchorState]:
    """Build RF anchors according to the configured information pattern.

    ``causal`` preserves the online/default behavior. ``smoothed`` performs an
    RF-only fixed-interval RTS smoothing pass over the event timeline before
    scoring radar candidates.  The smoothed mode is intentionally opt-in because
    it uses future RF measurements and is therefore an offline/SOTA ablation,
    not a causal tracking result.
    """

    mode = str(getattr(config, "rf_anchor_mode", "causal"))
    if mode == "causal":
        return _build_rf_anchor_states(
            events=events,
            acceleration_std_mps2=acceleration_std_mps2,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            tracker_factory=tracker_factory,
        )
    if mode == "smoothed":
        return _build_smoothed_rf_anchor_states(
            events=events,
            acceleration_std_mps2=acceleration_std_mps2,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            tracker_factory=tracker_factory,
        )
    raise ValueError('rf_anchor_mode must be either "causal" or "smoothed"')


def _build_smoothed_rf_anchor_states(
    *,
    events: list[dict[str, object]],
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    tracker_factory: Callable[..., Any] | None = None,
) -> dict[int, _AnchorState]:
    """Return offline RF-smoothed CV states at radar event indices.

    The forward pass stores the RF-only filtered/predicted state after every
    event in the chronological stream. RF events perform the usual gated and
    robust update; radar events are prediction-only support points. A backward
    RTS pass then conditions radar support points on both preceding and future
    RF measurements.
    """

    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _robust_update_for_measurement,
    )

    snapshots: list[dict[str, object]] = []
    tracker: Any | None = None
    previous_time_s: float | None = None

    for event_index, event in enumerate(events):
        time_s = float(event["time_s"])
        if tracker is None:
            if event["kind"] != "rf":
                continue
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            tracker = _make_tracker(
                tracker_factory,
                initial_position=measurement.vector,
                initial_time_s=measurement.time_s,
                acceleration_std_mps2=acceleration_std_mps2,
            )
            tracker.update(
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
            snapshots.append(
                {
                    "event_index": event_index,
                    "kind": "rf",
                    "state": tracker.state.copy(),
                    "covariance": tracker.covariance_matrix.copy(),
                    "transition": None,
                    "process_noise": None,
                }
            )
            previous_time_s = time_s
            continue

        assert previous_time_s is not None
        dt_s = float(time_s) - float(previous_time_s)
        if dt_s < -1.0e-9:
            raise ValueError("events must be processed in chronological order")
        transition = constant_velocity_matrix(max(dt_s, 0.0))
        process_noise = white_acceleration_process_noise(max(dt_s, 0.0), acceleration_std_mps2)
        tracker.predict_to(time_s)
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            tracker.update(
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
        snapshots.append(
            {
                "event_index": event_index,
                "kind": str(event["kind"]),
                "state": tracker.state.copy(),
                "covariance": tracker.covariance_matrix.copy(),
                "transition": transition,
                "process_noise": process_noise,
            }
        )
        previous_time_s = time_s

    if not snapshots:
        return {}

    filtered_states = [np.asarray(snapshot["state"], dtype=float) for snapshot in snapshots]
    filtered_covariances = [
        np.asarray(snapshot["covariance"], dtype=float) for snapshot in snapshots
    ]
    smoothed_states = [state.copy() for state in filtered_states]
    smoothed_covariances = [covariance.copy() for covariance in filtered_covariances]

    for index in range(len(snapshots) - 2, -1, -1):
        transition = np.asarray(snapshots[index + 1]["transition"], dtype=float)
        process_noise = np.asarray(snapshots[index + 1]["process_noise"], dtype=float)
        filtered_state = filtered_states[index]
        filtered_covariance = filtered_covariances[index]
        predicted_state = transition @ filtered_state
        predicted_covariance = _symmetrized_covariance(
            transition @ filtered_covariance @ transition.T + process_noise
        )
        smoother_gain = filtered_covariance @ transition.T @ np.linalg.pinv(predicted_covariance)
        smoothed_states[index] = filtered_state + smoother_gain @ (
            smoothed_states[index + 1] - predicted_state
        )
        smoothed_covariances[index] = _symmetrized_covariance(
            filtered_covariance
            + smoother_gain
            @ (smoothed_covariances[index + 1] - predicted_covariance)
            @ smoother_gain.T
        )

    anchors: dict[int, _AnchorState] = {}
    for snapshot, state, covariance in zip(snapshots, smoothed_states, smoothed_covariances):
        if snapshot["kind"] != "radar":
            continue
        anchors[int(snapshot["event_index"])] = _AnchorState(
            state=state.copy(),
            covariance=covariance.copy(),
        )
    return anchors


def _symmetrized_covariance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _select_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    """Return selected radar rows from the lowest-cost Viterbi path."""

    paths = _select_tracklet_viterbi_paths(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
    )
    return paths[0] if paths else []


def _select_tracklet_viterbi_paths(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[list[pd.Series]]:
    """Return up to ``path_beam_width`` radar paths ordered by Viterbi cost."""

    candidates = _select_tracklet_viterbi_path_candidates(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
    )
    return [rows for _, rows in candidates]


def _select_tracklet_viterbi_path_candidates(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[tuple[float, list[pd.Series]]]:
    """Return candidate Viterbi paths as ``(path_cost, selected_rows)`` tuples."""

    frames = _tracklet_viterbi_frames(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
    )
    paths = _beam_viterbi_paths(
        frames=frames,
        config=config,
        max_paths=int(config.path_beam_width),
    )
    return [
        (float(path.cost), _viterbi_path_to_rows(path, path_rank=path_rank, config=config))
        for path_rank, path in enumerate(paths)
    ]


def _tracklet_viterbi_frames(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[list[_ViterbiNode]]:
    return [
        _nodes_for_radar_frame(
            event_index=i,
            candidates=event["candidates"],
            anchor=anchors.get(i),
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        for i, event in enumerate(events)
        if event["kind"] == "radar"
    ]


def _beam_viterbi_paths(
    *,
    frames: list[list[_ViterbiNode]],
    config: TrackletViterbiAssociationConfig,
    max_paths: int,
) -> list[_ViterbiPath]:
    """Return the best Viterbi paths using a bounded per-node beam."""

    if not frames:
        return []
    beam_width = max(1, int(max_paths))
    states_by_node: list[list[_ViterbiPath]] = []
    for node in frames[0]:
        initial_cost = node.unary_cost + (config.missed_detection_cost if node.is_miss else 0.0)
        states_by_node.append([_ViterbiPath(float(initial_cost), (node,))])

    for frame_index in range(1, len(frames)):
        previous, current = frames[frame_index - 1], frames[frame_index]
        next_states_by_node: list[list[_ViterbiPath]] = []
        for node in current:
            candidates: list[_ViterbiPath] = []
            for previous_index, previous_node in enumerate(previous):
                for partial in states_by_node[previous_index]:
                    cost = partial.cost + _transition_cost(previous_node, node, config) + node.unary_cost
                    candidates.append(_ViterbiPath(float(cost), partial.nodes + (node,)))
            candidates.sort(key=lambda path: path.cost)
            next_states_by_node.append(candidates[:beam_width])
        states_by_node = next_states_by_node

    paths = [path for node_paths in states_by_node for path in node_paths]
    paths.sort(key=lambda path: path.cost)
    return paths[:beam_width]


def _viterbi_path_to_rows(
    path: _ViterbiPath,
    *,
    path_rank: int,
    config: TrackletViterbiAssociationConfig | None = None,
) -> list[pd.Series]:
    rows: list[pd.Series] = []
    for node in path.nodes:
        if node.is_miss or node.row is None:
            continue
        row = node.row.copy()
        row["association_mode"] = "tracklet-viterbi"
        row["association_action"] = "viterbi_selected"
        row["association_nis"] = float(node.anchor_nis)
        row["association_score"] = float(node.unary_cost)
        row["association_anchor_nis"] = float(node.anchor_nis)
        row["association_rf_anchor_mode"] = str(getattr(config, "rf_anchor_mode", "causal"))
        row["association_catprob_cost"] = float(node.catprob_cost)
        row["association_range_cost"] = float(node.range_cost)
        row["association_reranker_cost"] = float(node.reranker_cost)
        row["association_reranker_probability"] = node.reranker_probability
        row["association_reranker_logit"] = node.reranker_logit
        row["association_viterbi_path_cost"] = float(path.cost)
        row["association_viterbi_path_rank"] = int(path_rank)
        rows.append(row)
    return rows


def _nodes_for_radar_frame(
    *,
    event_index: int,
    candidates: pd.DataFrame,
    anchor: _AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[_ViterbiNode]:
    from raft_uav.baselines.radar_association import _catprob_candidate_pool

    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    event_key = _radar_event_key(candidates)
    reranker = _association_reranker_from_config(config)
    scored: list[tuple[float, _ViterbiNode]] = []
    for _, row in _catprob_candidate_pool(candidates, candidate_catprob_threshold).iterrows():
        position = _row_position(row)
        if position is None:
            continue
        anchor_nis, catprob_cost, range_cost = _candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=covariance,
            config=config,
        )
        base_unary_cost = float(config.anchor_nis_weight) * anchor_nis + catprob_cost + range_cost
        reranker_features = _candidate_reranker_features(
            row=row,
            position=position,
            anchor_nis=anchor_nis,
            catprob_cost=catprob_cost,
            range_cost=range_cost,
            base_unary_cost=base_unary_cost,
            time_s=time_s,
        )
        reranker_cost, reranker_probability, reranker_logit = _candidate_reranker_cost(
            reranker,
            reranker_features,
            config,
        )
        unary_cost = base_unary_cost + reranker_cost
        node = _ViterbiNode(
            event_index=event_index,
            event_key=event_key,
            time_s=float(row.get("time_s", time_s)),
            row=row.copy(),
            position=position,
            velocity=_row_velocity(row),
            track_id=_optional_track_id(row.get("track_id")),
            unary_cost=float(unary_cost),
            anchor_nis=float(anchor_nis),
            catprob_cost=float(catprob_cost),
            range_cost=float(range_cost),
            reranker_cost=float(reranker_cost),
            reranker_probability=reranker_probability,
            reranker_logit=reranker_logit,
        )
        scored.append((float(unary_cost), node))
    scored.sort(key=lambda item: item[0])
    nodes = [node for _, node in scored[: int(config.max_candidates_per_frame)]]
    nodes.append(
        _ViterbiNode(event_index, event_key, time_s, None, None, None, None, 0.0, 0.0, 0.0, 0.0, True)
    )
    return nodes


def _candidate_cost_terms(
    *,
    row: pd.Series,
    position: np.ndarray,
    anchor: _AnchorState | None,
    covariance: np.ndarray,
    config: TrackletViterbiAssociationConfig,
) -> tuple[float, float, float]:
    anchor_nis = 0.0
    if config.use_rf_anchor and anchor is not None:
        anchor_nis = _quadratic_form(
            position - np.asarray(anchor.state[:3], dtype=float),
            np.asarray(anchor.covariance[:3, :3], dtype=float) + covariance,
        )
    catprob = _optional_float(row.get("cat_prob_uav"))
    catprob = 1.0 if catprob is None else float(np.clip(catprob, config.min_catprob, 1.0))
    catprob_cost = float(config.catprob_weight) * float(-math.log(catprob))
    range_cost = 0.0
    range_m = _optional_float(row.get("range_m"))
    if config.range_gate_m is not None and range_m is not None:
        excess_m = max(0.0, float(range_m) - float(config.range_gate_m))
        if excess_m > 0.0:
            scale = max(float(config.range_gate_slack_m), 1.0)
            range_cost = float(config.range_penalty) * (excess_m / scale) ** 2
    return float(anchor_nis), float(catprob_cost), float(range_cost)


def _association_reranker_from_config(
    config: TrackletViterbiAssociationConfig,
) -> Mapping[str, object] | None:
    path = config.association_reranker_path
    if path is None:
        return None
    return _load_association_reranker(str(path))


@lru_cache(maxsize=8)
def _load_association_reranker(path: str) -> Mapping[str, object]:
    """Load a lightweight calibrated linear association scorer from JSON.

    The scorer is intentionally dependency-free so experiments do not need a
    heavy ML runtime.  Expected schema::

        {
          "intercept": -1.0,
          "coefficients": {"cat_prob_uav": 2.0, "anchor_nis": -0.1},
          "feature_means": {"anchor_nis": 4.0},
          "feature_scales": {"anchor_nis": 2.0},
          "probability_floor": 1e-6,
          "probability_ceiling": 0.999999
        }

    ``coefficients`` are applied to the standardized feature value when a mean
    or scale is supplied and to the raw value otherwise.  The resulting logit is
    converted into a candidate likelihood and injected as ``-log(p)`` cost.
    """

    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("association reranker JSON must contain an object")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, Mapping) or not coefficients:
        raise ValueError("association reranker JSON requires non-empty coefficients")
    for key, value in coefficients.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"association reranker coefficient {key!r} is not numeric") from exc
        if not np.isfinite(number):
            raise ValueError(f"association reranker coefficient {key!r} is not finite")
    return payload


def _candidate_reranker_features(
    *,
    row: pd.Series,
    position: np.ndarray,
    anchor_nis: float,
    catprob_cost: float,
    range_cost: float,
    base_unary_cost: float,
    time_s: float,
) -> dict[str, float]:
    features: dict[str, float] = {
        "time_s": float(time_s),
        "east_m": float(position[0]),
        "north_m": float(position[1]),
        "up_m": float(position[2]),
        "anchor_nis": float(anchor_nis),
        "catprob_cost": float(catprob_cost),
        "range_cost": float(range_cost),
        "base_unary_cost": float(base_unary_cost),
    }
    catprob = _optional_float(row.get("cat_prob_uav"))
    if catprob is not None:
        features["cat_prob_uav"] = float(catprob)
        features["neg_log_cat_prob_uav"] = float(-math.log(max(catprob, 1.0e-12)))
    range_m = _optional_float(row.get("range_m"))
    if range_m is not None:
        features["range_m"] = float(range_m)
        features["log1p_range_m"] = float(math.log1p(max(range_m, 0.0)))
    velocity = _row_velocity(row)
    if velocity is not None:
        features["speed_mps"] = float(np.linalg.norm(velocity))
        features["horizontal_speed_mps"] = float(np.linalg.norm(velocity[:2]))
        features["vertical_speed_mps"] = float(velocity[2])
    track_id = _optional_track_id(row.get("track_id"))
    features["has_track_id"] = 0.0 if track_id is None else 1.0

    for key, value in row.items():
        number = _optional_float(value)
        if number is None:
            continue
        features.setdefault(str(key), float(number))
        features[f"row_{key}"] = float(number)
    return {key: value for key, value in features.items() if np.isfinite(value)}


def _candidate_reranker_cost(
    model: Mapping[str, object] | None,
    features: Mapping[str, float],
    config: TrackletViterbiAssociationConfig,
) -> tuple[float, float | None, float | None]:
    if model is None or float(config.association_reranker_weight) == 0.0:
        return 0.0, None, None
    coefficients = model.get("coefficients")
    assert isinstance(coefficients, Mapping)
    means = model.get("feature_means")
    scales = model.get("feature_scales")
    if not isinstance(means, Mapping):
        means = {}
    if not isinstance(scales, Mapping):
        scales = {}
    logit = _optional_float(model.get("intercept")) or 0.0
    for name, coefficient in coefficients.items():
        value = float(features.get(str(name), 0.0))
        mean = _optional_float(means.get(str(name))) or 0.0
        scale = _optional_float(scales.get(str(name))) or 1.0
        if scale == 0.0:
            scale = 1.0
        logit += float(coefficient) * ((value - mean) / scale)
    probability = _logistic(logit)
    floor = _optional_float(model.get("probability_floor")) or 1.0e-9
    ceiling = _optional_float(model.get("probability_ceiling")) or (1.0 - 1.0e-9)
    probability = float(np.clip(probability, floor, ceiling))
    cost = float(config.association_reranker_weight) * float(-math.log(probability))
    return cost, probability, float(logit)


def _logistic(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def _transition_cost(
    previous: _ViterbiNode,
    current: _ViterbiNode,
    config: TrackletViterbiAssociationConfig,
) -> float:
    """Return dynamic-programming transition cost between two radar nodes."""

    if current.is_miss:
        return float(config.missed_detection_cost) + (
            float(config.consecutive_miss_cost) if previous.is_miss else 0.0
        )
    if previous.is_miss or previous.position is None or current.position is None:
        return 0.0
    dt_s = max(float(current.time_s) - float(previous.time_s), 1.0e-3)
    predicted = previous.position if previous.velocity is None else previous.position + previous.velocity * dt_s
    position_std = float(config.transition_position_std_m) + float(config.transition_speed_std_mps) * dt_s
    motion_nis = float(np.sum(((current.position - predicted) / position_std) ** 2))
    displacement_velocity = (current.position - previous.position) / dt_s
    speed_excess = max(0.0, float(np.linalg.norm(displacement_velocity)) - float(config.max_speed_mps))
    speed_cost = float(config.max_speed_penalty) * (speed_excess / float(config.transition_speed_std_mps)) ** 2
    velocity_nis = 0.0
    if current.velocity is not None:
        velocity_nis += float(np.sum(((current.velocity - displacement_velocity) / config.velocity_std_mps) ** 2))
    if previous.velocity is not None and current.velocity is not None:
        velocity_nis += 0.25 * float(np.sum((((current.velocity - previous.velocity) / dt_s) / 8.0) ** 2))
    return float(
        config.transition_nis_weight * motion_nis
        + config.velocity_nis_weight * velocity_nis
        + speed_cost
        + _track_continuity_cost(previous.track_id, current.track_id, config)
    )


def _track_continuity_cost(
    previous_track_id: int | None,
    current_track_id: int | None,
    config: TrackletViterbiAssociationConfig,
) -> float:
    if previous_track_id is None:
        return 0.0
    if current_track_id is None:
        return float(config.missing_track_id_cost)
    return 0.0 if int(previous_track_id) == int(current_track_id) else float(config.track_switch_cost)


def _replay_selected_tracklet_path(
    *,
    events: list[dict[str, object]],
    selected_rows: list[pd.Series],
    initial_measurement: TrackingMeasurement,
    initial_selected_row: pd.Series | None = None,
    acceleration_std_mps2: float,
    covariance: np.ndarray,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    tracker_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, object]], list[pd.Series]]:
    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _radar_row_to_measurement,
        _record,
        _robust_update_for_measurement,
    )

    selected_by_key = {_selected_row_event_key(row): row for row in selected_rows}
    tracker = _make_tracker(
        tracker_factory,
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    accepted_rows: list[pd.Series] = []

    if initial_selected_row is not None:
        replayed = initial_selected_row.copy()
        replayed["association_replay_accepted"] = True
        replayed["association_replay_nis"] = float("nan")
        accepted_rows.append(replayed)
    records.append(
        _record(
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
            association_mode=None if initial_selected_row is None else "tracklet-viterbi",
        )
    )

    for event in events[1:]:
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
        selected = selected_by_key.get(_radar_event_key(candidates))
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
        replayed = selected.copy()
        replayed["association_replay_accepted"] = bool(diagnostics.accepted)
        replayed["association_replay_nis"] = float(diagnostics.nis)
        if diagnostics.accepted:
            accepted_rows.append(replayed)
        records.append(
            _record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected.get("track_id")),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode="tracklet-viterbi",
            )
        )
    return records, accepted_rows


def _radar_event_key(candidates: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in candidates.columns and not candidates.empty:
        frame_index = _optional_float(candidates["frame_index"].iloc[0])
        if frame_index is not None:
            return "frame_index", int(frame_index)
    if "time_s" not in candidates.columns or candidates.empty:
        return "time_s", float("nan")
    return "time_s", round(float(pd.to_numeric(candidates["time_s"], errors="coerce").median()), 9)


def _selected_row_event_key(row: pd.Series) -> tuple[str, int | float]:
    frame_index = _optional_float(row.get("frame_index"))
    if frame_index is not None:
        return "frame_index", int(frame_index)
    time_s = _optional_float(row.get("time_s"))
    return ("time_s", float("nan")) if time_s is None else ("time_s", round(float(time_s), 9))


def _make_tracker(
    tracker_factory: Callable[..., Any] | None,
    *,
    initial_position: np.ndarray,
    initial_time_s: float,
    acceleration_std_mps2: float,
) -> Any:
    from raft_uav.baselines.radar_association import _make_tracker as make_tracker

    return make_tracker(
        tracker_factory,
        initial_position=initial_position,
        initial_time_s=initial_time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )


def _row_position(row: pd.Series) -> np.ndarray | None:
    try:
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
    except (KeyError, TypeError, ValueError):
        return None
    return position if np.isfinite(position).all() else None


def _row_velocity(row: pd.Series) -> np.ndarray | None:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row.index for column in required):
        return None
    try:
        velocity = np.array(
            [
                float(row["velocity_east_mps"]),
                float(row["velocity_north_mps"]),
                -float(row["velocity_down_mps"]),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _quadratic_form(residual: np.ndarray, covariance: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    value = float(residual.T @ solved)
    return value if np.isfinite(value) else 0.0


def _optional_track_id(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
