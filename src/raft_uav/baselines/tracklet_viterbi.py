"""Truth-free sequence-level radar tracklet association.

This module selects one coherent radar path over all radar frames and then
replays that path through the existing asynchronous CV Kalman baseline.  The
objective combines RF-anchor consistency, Fortem track-ID continuity, CV motion
feasibility, radar velocity consistency, class probability, range, and a
missed-detection branch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker, TrackingMeasurement


@dataclass(frozen=True)
class TrackletViterbiAssociationConfig:
    """Configuration for sequence-level radar association."""

    max_candidates_per_frame: int = 8
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
    min_catprob: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be positive")
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
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.range_gate_m is not None and float(self.range_gate_m) <= 0.0:
            raise ValueError("range_gate_m must be positive or None")
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


TRACKLET_REPLAY_TRACKER_KINDS = ("cv", "imm")


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
    replay_tracker_kind: str = "cv",
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion after Viterbi radar-tracklet selection."""

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_measurement,
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
    initial = _initial_measurement(
        events[0],
        association="tracklet-viterbi",
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
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        replay_tracker_kind=replay_tracker_kind,
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


def _normalize_tracklet_replay_tracker_kind(kind: str) -> str:
    """Return a normalized replay tracker identifier or raise a clear error."""

    normalized = str(kind).strip().lower()
    if normalized not in TRACKLET_REPLAY_TRACKER_KINDS:
        valid = ", ".join(TRACKLET_REPLAY_TRACKER_KINDS)
        raise ValueError(f"replay_tracker_kind must be one of {valid}; got {kind!r}")
    return normalized


def _make_tracklet_replay_tracker(
    replay_tracker_kind: str,
    *,
    initial_measurement: TrackingMeasurement,
    acceleration_std_mps2: float,
):
    """Create the tracker used to replay the selected tracklet-Viterbi path."""

    kind = _normalize_tracklet_replay_tracker_kind(replay_tracker_kind)
    if kind == "cv":
        return AsyncConstantVelocityKalmanTracker(
            initial_position=initial_measurement.vector,
            initial_time_s=initial_measurement.time_s,
            acceleration_std_mps2=acceleration_std_mps2,
        )
    if kind == "imm":
        from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker

        return AsyncInteractingMultipleModelTracker(
            initial_position=initial_measurement.vector,
            initial_time_s=initial_measurement.time_s,
            acceleration_std_mps2=acceleration_std_mps2,
        )
    raise AssertionError(f"unhandled replay tracker kind {kind!r}")


def _tracklet_replay_record(
    measurement: TrackingMeasurement,
    tracker: object,
    diagnostics: Any,
    *,
    replay_tracker_kind: str,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    """Return a standard tracking record plus replay-tracker diagnostics."""

    from raft_uav.baselines.radar_association import _record

    record = _record(
        measurement,
        tracker,
        diagnostics,
        track_id=track_id,
        association_nis=association_nis,
        association_score=association_score,
        association_mode=association_mode,
    )
    record["tracklet_replay_tracker"] = replay_tracker_kind
    _attach_imm_record_diagnostics(record, tracker)
    return record


def _attach_imm_record_diagnostics(record: dict[str, object], tracker: object) -> None:
    """Add IMM mode diagnostics when the replay tracker exposes them."""

    mode_names = getattr(tracker, "mode_names", None)
    if mode_names is None:
        return
    try:
        mode_probabilities = np.asarray(getattr(tracker, "mode_probabilities"), dtype=float)
    except (AttributeError, TypeError, ValueError):
        return
    record["mode_names"] = tuple(str(name) for name in mode_names)
    record["mode_probabilities"] = mode_probabilities.copy()
    mode_probability_map = getattr(tracker, "mode_probability_map", None)
    if mode_probability_map is not None:
        record["mode_probability_map"] = dict(mode_probability_map)
    most_likely = getattr(tracker, "most_likely_mode_name", None)
    if most_likely is not None:
        record["most_likely_mode"] = str(most_likely)


def _annotate_replayed_row_with_tracker(
    row: pd.Series,
    *,
    tracker: object,
    replay_tracker_kind: str,
) -> None:
    """Attach replay-tracker metadata to a selected radar row."""

    row["association_replay_tracker"] = replay_tracker_kind
    most_likely = getattr(tracker, "most_likely_mode_name", None)
    if most_likely is not None:
        row["association_replay_most_likely_mode"] = str(most_likely)


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
) -> dict[int, _AnchorState]:
    """Return RF-only CV predictions at radar event indices."""

    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _robust_update_for_measurement,
    )

    tracker: AsyncConstantVelocityKalmanTracker | None = None
    anchors: dict[int, _AnchorState] = {}
    for event_index, event in enumerate(events):
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            if tracker is None:
                tracker = AsyncConstantVelocityKalmanTracker(
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
        elif tracker is not None:
            tracker.predict_to(time_s)
            anchors[event_index] = _AnchorState(
                state=tracker.state.copy(),
                covariance=tracker.covariance_matrix.copy(),
            )
    return anchors


def _select_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    """Return selected radar rows from the lowest-cost Viterbi path."""

    frames = [
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
    if not frames:
        return []

    costs = [np.array([n.unary_cost + (config.missed_detection_cost if n.is_miss else 0.0) for n in frames[0]])]
    parents = [np.full(len(frames[0]), -1, dtype=int)]
    for frame_index in range(1, len(frames)):
        previous, current = frames[frame_index - 1], frames[frame_index]
        current_costs = np.empty(len(current), dtype=float)
        current_parents = np.empty(len(current), dtype=int)
        for j, node in enumerate(current):
            transition = np.array(
                [costs[-1][k] + _transition_cost(prev, node, config) for k, prev in enumerate(previous)]
            )
            parent = int(np.argmin(transition))
            current_parents[j] = parent
            current_costs[j] = node.unary_cost + float(transition[parent])
        costs.append(current_costs)
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

    rows: list[pd.Series] = []
    for node in path:
        if node.is_miss or node.row is None:
            continue
        row = node.row.copy()
        row["association_mode"] = "tracklet-viterbi"
        row["association_action"] = "viterbi_selected"
        row["association_nis"] = float(node.anchor_nis)
        row["association_score"] = float(node.unary_cost)
        row["association_anchor_nis"] = float(node.anchor_nis)
        row["association_catprob_cost"] = float(node.catprob_cost)
        row["association_range_cost"] = float(node.range_cost)
        row["association_viterbi_path_cost"] = path_cost
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
        unary_cost = float(config.anchor_nis_weight) * anchor_nis + catprob_cost + range_cost
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
    acceleration_std_mps2: float,
    covariance: np.ndarray,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    replay_tracker_kind: str = "cv",
) -> tuple[list[dict[str, object]], list[pd.Series]]:
    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _radar_row_to_measurement,
        _robust_update_for_measurement,
    )

    selected_by_key = {_selected_row_event_key(row): row for row in selected_rows}
    tracker_kind = _normalize_tracklet_replay_tracker_kind(replay_tracker_kind)
    tracker = _make_tracklet_replay_tracker(
        tracker_kind,
        initial_measurement=initial_measurement,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    accepted_rows: list[pd.Series] = []
    for event in events:
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
            records.append(
                _tracklet_replay_record(
                    measurement,
                    tracker,
                    diagnostics,
                    replay_tracker_kind=tracker_kind,
                )
            )
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
        _annotate_replayed_row_with_tracker(
            replayed,
            tracker=tracker,
            replay_tracker_kind=tracker_kind,
        )
        if diagnostics.accepted:
            accepted_rows.append(replayed)
        records.append(
            _tracklet_replay_record(
                measurement,
                tracker,
                diagnostics,
                replay_tracker_kind=tracker_kind,
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
