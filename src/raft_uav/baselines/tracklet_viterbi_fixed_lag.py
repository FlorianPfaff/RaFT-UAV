"""Fixed-lag tracklet-Viterbi radar association.

The offline tracklet-Viterbi baseline selects a single path over the full flight.
This module keeps the same scoring objective but constrains each committed radar
decision to use only a bounded look-ahead window.  Committed decisions are made
sequentially: each window is prepended with the previous committed radar choice
as a forced prefix candidate, so later decisions remain dynamically consistent
with earlier fixed-lag commitments.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _build_rf_anchor_states_for_config,
    _first_rf_bootstrap_index,
    _radar_event_key,
    _select_tracklet_viterbi_path,
    _selected_row_event_key,
)
from raft_uav.baselines.tracklet_viterbi_result import (
    TrackletViterbiResult,
    _empty_candidate_ledger,
    _empty_replayed_rows,
    _replay_selected_tracklet_path_with_replay,
    _tracklet_candidate_ledger,
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
    tracker_factory: Callable[..., Any] | None = None,
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
        tracker_factory=tracker_factory,
    )
    accepted_frame = _selected_rows_frame(radar, accepted)
    replayed_frame = _selected_rows_frame(radar, replayed)
    return records, accepted_frame, replayed_frame


def run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_result(
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
) -> TrackletViterbiResult:
    """Run fixed-lag tracklet-Viterbi and keep replay/artifact tables.

    The legacy fixed-lag entry point returns only the replay records plus the
    accepted/replayed radar paths.  This wrapper preserves that behavior for
    callers while also reconstructing the candidate ledger used by the shared
    artifact writer, so fixed-lag runs can be the main experiment path without
    losing association diagnostics.
    """

    rf_measurement_list = list(rf_measurements)
    records, accepted, replayed = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=rf_measurement_list,
            radar=radar,
            lag_s=lag_s,
            acceleration_std_mps2=acceleration_std_mps2,
            radar_xy_std_m=radar_xy_std_m,
            radar_z_std_m=radar_z_std_m,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
    )
    ledger = _fixed_lag_candidate_ledger(
        rf_measurements=rf_measurement_list,
        radar=radar,
        replayed=replayed,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=radar_xy_std_m,
        radar_z_std_m=radar_z_std_m,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
    )
    return TrackletViterbiResult(records, accepted, replayed, ledger)


def _fixed_lag_candidate_ledger(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    replayed: pd.DataFrame,
    acceleration_std_mps2: float,
    radar_xy_std_m: float,
    radar_z_std_m: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig | None,
) -> pd.DataFrame:
    """Rebuild the fixed-lag candidate ledger from the replayed choices."""

    if replayed.empty:
        return _empty_candidate_ledger(radar)

    from raft_uav.baselines.radar_association import _events

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return _empty_candidate_ledger(radar)
    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        return _empty_candidate_ledger(radar)
    events = events[bootstrap_index:]
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
    return _tracklet_candidate_ledger(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        selected_rows=_replayed_rows_as_selected_rows(replayed),
    )


def _replayed_rows_as_selected_rows(replayed: pd.DataFrame) -> list[pd.Series]:
    rows: list[pd.Series] = []
    for _, row in replayed.iterrows():
        selected = row.copy()
        selected.name = None
        rows.append(selected)
    return rows


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

        selected_window = _select_tracklet_viterbi_path(
            events=local_events,
            anchors=local_anchors,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        selected_by_key = {_selected_row_event_key(row): row for row in selected_window}
        selected = selected_by_key.get(event_key)
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
            row["association_prefix_constrained"] = True
            row["association_prefix_track_id"] = previous_committed.get("track_id", np.nan)
            row["association_prefix_time_s"] = prefix_time_s
        committed[event_key] = row
        previous_committed = row

    return list(committed.values())


def _prefix_event(row: pd.Series) -> dict[str, object]:
    """Return a synthetic one-candidate radar event that forces the prefix row."""

    prefix = row.copy()
    prefix["cat_prob_uav"] = 1.0
    prefix["association_score"] = 0.0
    if "range_m" in prefix.index:
        prefix["range_m"] = 0.0
    return {
        "kind": "radar",
        "time_s": float(prefix.get("time_s", 0.0)),
        "candidates": pd.DataFrame([prefix]),
    }
