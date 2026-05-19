"""Fixed-lag tracklet-Viterbi radar association.

The offline tracklet-Viterbi baseline selects a single path over the full flight.
This module keeps the same scoring objective but constrains each committed radar
decision to use only a bounded look-ahead window.  Committed decisions are made
sequentially: each window is prepended with the previous committed radar choice
as a forced prefix candidate, so later decisions remain dynamically consistent
with earlier fixed-lag commitments.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _ViterbiNode,
    _build_rf_anchor_states,
    _first_rf_bootstrap_index,
    _nodes_for_radar_frame,
    _radar_event_key,
    _selected_row_event_key,
    _transition_cost,
)
from raft_uav.baselines.tracklet_viterbi_result import (
    _empty_replayed_rows,
    _replay_selected_tracklet_path_with_replay,
)


def run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    lag_s: float,
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
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Run fixed-lag tracklet-Viterbi and replay committed radar choices."""

    if lag_s <= 0.0:
        raise ValueError("lag_s must be positive")

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_measurement,
        _selected_rows_frame,
    )

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [
            float(radar_xy_std_m) ** 2,
            float(radar_xy_std_m) ** 2,
            float(radar_z_std_m) ** 2,
        ]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        empty = _empty_selected_radar(radar)
        return [], empty, _empty_replayed_rows(empty)

    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        empty = _empty_selected_radar(radar)
        return [], empty, _empty_replayed_rows(empty)
    events = events[bootstrap_index:]

    initial = _initial_measurement(
        events[0],
        association="tracklet-viterbi-fixed-lag",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial is None:
        empty = _empty_selected_radar(radar)
        return [], empty, _empty_replayed_rows(empty)

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
    selected = select_fixed_lag_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        lag_s=lag_s,
    )
    records, accepted, replayed = _replay_selected_tracklet_path_with_replay(
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
    accepted_frame = _selected_rows_frame(radar, accepted)
    replayed_frame = _selected_rows_frame(radar, replayed)
    return records, accepted_frame, replayed_frame


def select_fixed_lag_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, object],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    lag_s: float,
) -> list[pd.Series]:
    """Commit radar decisions with bounded future context and prefix memory.

    For each radar event at time ``t``, solve the ordinary Viterbi objective on
    a local window ending at ``t + lag_s``.  When a previous radar decision has
    already been committed, it is prepended to the local window as a single
    zero-cost prefix candidate.  The first newly committed choice is therefore
    selected by a proper prefix-constrained Viterbi objective while still using
    at most ``lag_s`` seconds of future information.
    """

    if lag_s <= 0.0:
        raise ValueError("lag_s must be positive")

    radar_indices = [index for index, event in enumerate(events) if event.get("kind") == "radar"]
    committed: dict[tuple[str, int | float], pd.Series] = {}
    previous_committed: pd.Series | None = None

    for global_index in radar_indices:
        candidates = events[global_index]["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        event_key = _radar_event_key(candidates)
        if event_key in committed:
            continue

        start_s = float(events[global_index]["time_s"])
        end_s = start_s + float(lag_s)
        window_indices = [
            index
            for index, event in enumerate(events)
            if start_s <= float(event["time_s"]) <= end_s
        ]
        if global_index not in window_indices:
            window_indices.append(global_index)
            window_indices.sort()

        local_events = [events[index] for index in window_indices]
        local_anchors = {
            local_index: anchors[global_index]
            for local_index, global_index in enumerate(window_indices)
            if global_index in anchors
        }
        prefix_time_s = None
        if previous_committed is not None:
            prefix_time_s = float(previous_committed.get("time_s", start_s))
            local_events = [_prefix_event(previous_committed)] + local_events
            local_anchors = {local_index + 1: anchor for local_index, anchor in local_anchors.items()}

        selected_window = _select_prefix_constrained_tracklet_viterbi_path(
            events=local_events,
            anchors=local_anchors,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        selected_by_key = {_selected_row_event_key(row): row for row in selected_window}
        selected = selected_by_key.get(event_key)
        if selected is None:
            selected = _prefix_continuation_candidate(candidates, previous_committed)
        if selected is None:
            continue

        row = selected.copy()
        row["association_mode"] = "tracklet-viterbi-fixed-lag"
        row["association_lag_s"] = float(lag_s)
        row["association_lag_window_start_s"] = start_s
        row["association_lag_window_end_s"] = end_s
        row["association_lag_window_event_count"] = int(len(local_events))
        row["association_lag_window_radar_count"] = int(
            sum(event.get("kind") == "radar" for event in local_events)
        )
        row["association_lag_commit_time_s"] = end_s
        row["association_lag_commit_delay_s"] = max(
            0.0,
            end_s - float(row.get("time_s", start_s)),
        )
        if previous_committed is not None:
            row["association_prefix_constrained"] = True
            row["association_prefix_track_id"] = previous_committed.get("track_id", np.nan)
            row["association_prefix_time_s"] = prefix_time_s
        committed[event_key] = row
        previous_committed = row

    return list(committed.values())


def _prefix_continuation_candidate(
    candidates: pd.DataFrame,
    previous_committed: pd.Series | None,
) -> pd.Series | None:
    if previous_committed is None or "track_id" not in candidates.columns:
        return None
    previous_id = pd.to_numeric(pd.Series([previous_committed.get("track_id")]), errors="coerce").iloc[0]
    if not np.isfinite(previous_id):
        return None
    track_ids = pd.to_numeric(candidates["track_id"], errors="coerce")
    mask = track_ids.notna() & np.isclose(track_ids.to_numpy(dtype=float), float(previous_id))
    matching = candidates.loc[mask]
    if matching.empty:
        return None
    return matching.iloc[0].copy()


def _select_prefix_constrained_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, object],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    """Return the local Viterbi path with any fixed-lag prefix forced.

    The base selector appends a miss node to every radar frame.  That is right
    for ordinary frames but not for the synthetic prefix frame used by the
    fixed-lag selector: the prefix represents a decision that has already been
    committed and must therefore be part of all later windows.  Without this
    helper, low or zero missed-detection costs can let the local path skip the
    prefix and choose a later high-class-probability track that is inconsistent
    with the committed history.
    """

    frames: list[list[_ViterbiNode]] = []
    for event_index, event in enumerate(events):
        if event.get("kind") != "radar":
            continue
        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        nodes = _nodes_for_radar_frame(
            event_index=event_index,
            candidates=candidates,
            anchor=anchors.get(event_index),
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        if _is_forced_prefix_event(event):
            nodes = [node for node in nodes if not node.is_miss]
            if len(nodes) != 1:
                raise RuntimeError(
                    "fixed-lag prefix event must contain exactly one radar candidate"
                )
        frames.append(nodes)
    if not frames:
        return []
    return _rows_from_viterbi_frames(frames, config)


def _rows_from_viterbi_frames(
    frames: list[list[_ViterbiNode]],
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    costs = [
        np.array(
            [
                node.unary_cost + (config.missed_detection_cost if node.is_miss else 0.0)
                for node in frames[0]
            ],
            dtype=float,
        )
    ]
    parents = [np.full(len(frames[0]), -1, dtype=int)]
    for frame_index in range(1, len(frames)):
        previous, current = frames[frame_index - 1], frames[frame_index]
        current_costs = np.empty(len(current), dtype=float)
        current_parents = np.empty(len(current), dtype=int)
        for j, node in enumerate(current):
            transition = np.array(
                [
                    costs[-1][k] + _transition_cost(prev, node, config)
                    for k, prev in enumerate(previous)
                ],
                dtype=float,
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
        row["association_mode"] = "tracklet-viterbi-fixed-lag"
        row["association_action"] = "viterbi_selected"
        row["association_nis"] = float(node.anchor_nis)
        row["association_score"] = float(node.unary_cost)
        row["association_anchor_nis"] = float(node.anchor_nis)
        row["association_catprob_cost"] = float(node.catprob_cost)
        row["association_range_cost"] = float(node.range_cost)
        row["association_viterbi_path_cost"] = path_cost
        rows.append(row)
    return rows


def _is_forced_prefix_event(event: Mapping[str, object]) -> bool:
    candidates = event.get("candidates")
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return False
    if "association_fixed_lag_forced_prefix" not in candidates.columns:
        return False
    return bool(candidates["association_fixed_lag_forced_prefix"].fillna(False).all())


def _prefix_event(row: pd.Series) -> dict[str, object]:
    """Return a synthetic one-candidate radar event that forces the prefix row."""

    prefix = row.copy()
    prefix["cat_prob_uav"] = 1.0
    prefix["association_score"] = 0.0
    if "range_m" in prefix.index:
        prefix["range_m"] = 0.0
    prefix["association_fixed_lag_forced_prefix"] = True
    return {
        "kind": "radar",
        "time_s": float(prefix.get("time_s", 0.0)),
        "candidates": pd.DataFrame([prefix]),
    }
