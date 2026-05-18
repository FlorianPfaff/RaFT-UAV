"""Fixed-lag tracklet-Viterbi radar association.

The offline tracklet-Viterbi baseline selects a single path over the full flight.
This module keeps the same scoring objective but constrains each committed radar
decision to use only a bounded look-ahead window.  Committed decisions are made
sequentially: each window is constrained by the previous committed radar choice,
so later windows cannot silently ignore an earlier association decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _build_rf_anchor_states,
    _optional_track_id,
    _radar_event_key,
    _select_tracklet_viterbi_path,
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
    a local window ending at ``t + lag_s``.  Unlike independent per-frame window
    solving, this routine prepends the previous committed row as a fixed prefix
    node.  The newly committed choice must therefore be dynamically consistent
    with the already committed sequence while still using at most ``lag_s``
    seconds of future information.
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
        selected_window = _select_tracklet_viterbi_path(
            events=local_events,
            anchors=local_anchors,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        selected_by_key = {_selected_row_event_key(row): row for row in selected_window}
        selected = selected_by_key.get(event_key)
        selected = _choose_prefix_consistent_selection(
            selected=selected,
            candidates=candidates,
            previous_committed=previous_committed,
            config=config,
            window_start_s=start_s,
            lag_s=lag_s,
        )
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
        if previous_committed is not None:
            row["association_prefix_track_id"] = previous_committed.get("track_id", np.nan)
            row["association_prefix_time_s"] = previous_committed.get("time_s", np.nan)
        committed[event_key] = row
        previous_committed = row

    return list(committed.values())


def _choose_prefix_consistent_selection(
    *,
    selected: pd.Series | None,
    candidates: pd.DataFrame,
    previous_committed: pd.Series | None,
    config: TrackletViterbiAssociationConfig,
    window_start_s: float,
    lag_s: float,
) -> pd.Series | None:
    """Return the first-window decision conditioned on the committed prefix.

    If the unconstrained window solver selects a row that is already consistent
    with the prefix, keep it.  Otherwise evaluate all rows in the current radar
    frame using the prefix transition cost plus the original unary score and
    commit the best prefix-compatible candidate.
    """

    if selected is None or previous_committed is None:
        return selected
    selected_cost = _prefix_adjusted_cost(
        previous_committed=previous_committed,
        candidate=selected,
        config=config,
    )
    best = selected
    best_cost = selected_cost
    for _, candidate in candidates.iterrows():
        candidate_cost = _prefix_adjusted_cost(
            previous_committed=previous_committed,
            candidate=candidate,
            config=config,
        )
        if candidate_cost < best_cost:
            best = candidate.copy()
            best_cost = candidate_cost
    out = best.copy()
    out["association_prefix_adjusted_cost"] = float(best_cost)
    out["association_prefix_unconstrained_cost"] = float(selected_cost)
    out["association_prefix_adjusted"] = _track_id(out) != _track_id(selected)
    out["association_prefix_lag_window_start_s"] = float(window_start_s)
    out["association_prefix_lag_s"] = float(lag_s)
    return out


def _prefix_adjusted_cost(
    *,
    previous_committed: pd.Series,
    candidate: pd.Series,
    config: TrackletViterbiAssociationConfig,
) -> float:
    from raft_uav.baselines.tracklet_viterbi import _ViterbiNode, _row_position, _row_velocity

    previous_position = _row_position(previous_committed)
    current_position = _row_position(candidate)
    if previous_position is None or current_position is None:
        return float("inf")
    previous_node = _ViterbiNode(
        event_index=-1,
        event_key=_selected_row_event_key(previous_committed),
        time_s=float(previous_committed.get("time_s", 0.0)),
        row=previous_committed,
        position=previous_position,
        velocity=_row_velocity(previous_committed),
        track_id=_optional_track_id(previous_committed.get("track_id")),
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )
    current_node = _ViterbiNode(
        event_index=-1,
        event_key=_selected_row_event_key(candidate),
        time_s=float(candidate.get("time_s", 0.0)),
        row=candidate,
        position=current_position,
        velocity=_row_velocity(candidate),
        track_id=_optional_track_id(candidate.get("track_id")),
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )
    unary = float(candidate.get("association_score", 0.0))
    if not np.isfinite(unary):
        unary = 0.0
    return float(unary + _transition_cost(previous_node, current_node, config))


def _track_id(row: pd.Series) -> int | None:
    return _optional_track_id(row.get("track_id"))
