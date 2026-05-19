"""Native retention-aware tracklet-Viterbi association.

This module keeps the public retention entry point while avoiding runtime
monkey-patching.  It reuses the base tracklet-Viterbi event handling,
RF-anchor construction, transition costs, and replay logic, but performs node
construction and Viterbi selection locally with track-aware candidate retention,
optional soft catProb penalties, and a truth-free Fortem track-support prior.
Track-support statistics are computed from prefix-only radar history for each
radar event, so the reward does not leak future Fortem detections into earlier
association decisions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig
DEFAULT_BELOW_CATPROB_THRESHOLD_PENALTY = 3.0
DEFAULT_TRACK_SUPPORT_WEIGHT = 0.45
DEFAULT_MAX_TRACK_SUPPORT_REWARD = 4.0


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
    """Run tracklet-Viterbi with native retention-aware node construction."""

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
    bootstrap_index = _base._first_rf_bootstrap_index(events)
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

    anchors = _base._build_rf_anchor_states(
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
        track_support_by_event=_track_support_by_event_prefix(events),
    )
    records, accepted = _base._replay_selected_tracklet_path(
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


def _select_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _base._AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    track_support_by_id: Mapping[int, Mapping[str, float]] | None = None,
    track_support_by_event: Mapping[int, Mapping[int, Mapping[str, float]]] | None = None,
) -> list[pd.Series]:
    """Return selected radar rows from retention-aware Viterbi scoring.

    ``track_support_by_event`` maps each global radar event index to support
    statistics computed from radar frames that precede that event.  The older
    ``track_support_by_id`` argument is kept for focused unit tests and explicit
    offline ablations, but callers must not use it for causal/default results.
    """

    if track_support_by_id is not None and track_support_by_event is not None:
        raise ValueError("provide either track_support_by_id or track_support_by_event, not both")

    frames: list[list[_base._ViterbiNode]] = []
    for i, event in enumerate(events):
        if event["kind"] != "radar":
            continue
        support_by_id = (
            track_support_by_event.get(i, {})
            if track_support_by_event is not None
            else track_support_by_id
        )
        frames.append(
            _nodes_for_radar_frame_with_track_retention(
                event_index=i,
                candidates=event["candidates"],
                anchor=anchors.get(i),
                covariance=covariance,
                candidate_catprob_threshold=candidate_catprob_threshold,
                config=config,
                track_support_by_id=support_by_id,
            )
        )
    if not frames:
        return []

    costs = [
        np.array(
            [n.unary_cost + (config.missed_detection_cost if n.is_miss else 0.0) for n in frames[0]]
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
                    costs[-1][k] + _base._transition_cost(prev, node, config)
                    for k, prev in enumerate(previous)
                ]
            )
            parent = int(np.argmin(transition))
            current_parents[j] = parent
            current_costs[j] = node.unary_cost + float(transition[parent])
        costs.append(current_costs)
        parents.append(current_parents)

    best = int(np.argmin(costs[-1]))
    path_cost = float(costs[-1][best])
    path: list[_base._ViterbiNode] = []
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


def _nodes_for_radar_frame_with_track_retention(
    *,
    event_index: int,
    candidates: pd.DataFrame,
    anchor: _base._AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    track_support_by_id: Mapping[int, Mapping[str, float]] | None = None,
) -> list[_base._ViterbiNode]:
    """Build Viterbi nodes while keeping top-K plus per-track candidates."""

    support_by_id = {} if track_support_by_id is None else track_support_by_id
    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    event_key = _base._radar_event_key(candidates)
    scored: list[tuple[float, int, _base._ViterbiNode]] = []
    for candidate_rank, (_, row) in enumerate(
        _candidate_pool_for_retention(candidates, candidate_catprob_threshold, config).iterrows()
    ):
        position = _base._row_position(row)
        if position is None:
            continue
        anchor_nis, base_catprob_cost, range_cost = _base._candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=covariance,
            config=config,
        )
        soft_threshold_cost = _catprob_threshold_penalty(
            row,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        track_support_cost, track_support = _track_support_cost(
            row,
            track_support_by_id=support_by_id,
            config=config,
        )
        catprob_cost = float(base_catprob_cost + soft_threshold_cost)
        unary_cost = (
            float(config.anchor_nis_weight) * anchor_nis
            + catprob_cost
            + range_cost
            + track_support_cost
        )
        selected_row = row.copy()
        _write_track_support_diagnostics(selected_row, track_support_cost, track_support)
        if candidate_catprob_threshold is not None:
            selected_row["association_catprob_threshold"] = float(candidate_catprob_threshold)
            selected_row["association_catprob_soft_penalty"] = float(soft_threshold_cost)
            selected_row["association_catprob_below_threshold"] = bool(soft_threshold_cost > 0.0)
        node = _base._ViterbiNode(
            event_index=event_index,
            event_key=event_key,
            time_s=float(row.get("time_s", time_s)),
            row=selected_row,
            position=position,
            velocity=_base._row_velocity(row),
            track_id=_base._optional_track_id(row.get("track_id")),
            unary_cost=float(unary_cost),
            anchor_nis=float(anchor_nis),
            catprob_cost=float(catprob_cost),
            range_cost=float(range_cost),
        )
        scored.append((float(unary_cost), int(candidate_rank), node))

    nodes = _retain_top_and_track_representatives(scored, config)
    nodes.append(
        _base._ViterbiNode(
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
    )
    return nodes


def _candidate_pool_for_retention(
    candidates: pd.DataFrame,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> pd.DataFrame:
    """Return candidate rows for hard or soft catProb retention ablations."""

    mode = str(getattr(config, "catprob_retention_mode", "soft")).strip().lower()
    if mode == "hard":
        from raft_uav.baselines.radar_association import _catprob_candidate_pool

        return _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    if mode == "soft":
        return candidates
    raise ValueError("catprob_retention_mode must be 'hard' or 'soft'")


def _write_track_support_diagnostics(
    row: pd.Series,
    track_support_cost: float,
    track_support: Mapping[str, float],
) -> None:
    row["association_track_support_cost"] = float(track_support_cost)
    row["association_track_support_score"] = float(track_support.get("score", 0.0))
    row["association_track_support_count"] = float(track_support.get("count", 0.0))
    row["association_track_support_span_s"] = float(track_support.get("span_s", 0.0))
    row["association_track_support_continuity"] = float(track_support.get("continuity", 0.0))
    row["association_track_support_median_catprob"] = float(
        track_support.get("median_catprob", 0.0)
    )


def _track_support_by_event_prefix(
    events: Iterable[dict[str, object]],
) -> dict[int, dict[int, dict[str, float]]]:
    """Return prefix-only Fortem track support for each radar event.

    The support attached to event ``i`` is computed from radar candidates in
    events with lower indices only.  This keeps the track-support reward causal
    with respect to the radar stream, while still allowing persistent track IDs
    to be rewarded once they have already survived previous frames.
    """

    support_by_event: dict[int, dict[int, dict[str, float]]] = {}
    previous_radar_frames: list[pd.DataFrame] = []
    for event_index, event in enumerate(events):
        if event.get("kind") != "radar":
            continue
        if previous_radar_frames:
            prefix = pd.concat(previous_radar_frames, ignore_index=True, sort=False)
            support_by_event[event_index] = _track_support_by_id(prefix)
        else:
            support_by_event[event_index] = {}
        candidates = event.get("candidates")
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            previous_radar_frames.append(candidates)
    return support_by_event


def _track_support_by_id(radar: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Return truth-free support statistics for finite Fortem track IDs in ``radar``."""

    if radar.empty or "track_id" not in radar.columns:
        return {}
    track_ids = pd.to_numeric(radar["track_id"], errors="coerce")
    finite = np.isfinite(track_ids.to_numpy(dtype=float))
    if not finite.any():
        return {}
    frame = radar.loc[finite].copy()
    frame["_track_id_int"] = track_ids.loc[finite].astype(int).to_numpy()
    support: dict[int, dict[str, float]] = {}
    for track_id, group in frame.groupby("_track_id_int", sort=False):
        count = float(len(group))
        span_s = _finite_span(group["time_s"]) if "time_s" in group.columns else 0.0
        frame_span = max(count, 1.0)
        if "frame_index" in group.columns:
            frame_indices = pd.to_numeric(group["frame_index"], errors="coerce").dropna()
            if len(frame_indices):
                frame_span = float(max(1.0, frame_indices.max() - frame_indices.min() + 1.0))
        continuity = float(np.clip(count / max(frame_span, 1.0), 0.0, 1.0))
        median_catprob = _median_catprob(group)
        score = float(
            np.log1p(count)
            + 0.5 * np.log1p(max(span_s, 0.0))
            + 0.5 * continuity
            + 0.5 * median_catprob
        )
        support[int(track_id)] = {
            "count": count,
            "span_s": float(span_s),
            "frame_span": float(frame_span),
            "continuity": continuity,
            "median_catprob": median_catprob,
            "score": score,
        }
    return support


def _track_support_cost(
    row: pd.Series,
    *,
    track_support_by_id: Mapping[int, Mapping[str, float]],
    config: TrackletViterbiAssociationConfig,
) -> tuple[float, Mapping[str, float]]:
    """Return a bounded reward for stable Fortem track IDs."""

    track_id = _base._optional_track_id(row.get("track_id"))
    if track_id is None:
        return 0.0, {}
    support = track_support_by_id.get(int(track_id), {})
    if not support:
        return 0.0, {}
    weight = max(0.0, float(getattr(config, "track_support_weight", DEFAULT_TRACK_SUPPORT_WEIGHT)))
    max_reward = max(
        0.0,
        float(getattr(config, "max_track_support_reward", DEFAULT_MAX_TRACK_SUPPORT_REWARD)),
    )
    if weight <= 0.0 or max_reward <= 0.0:
        return 0.0, support
    reward = min(max_reward, weight * max(0.0, float(support.get("score", 0.0))))
    return -float(reward), support


def _finite_span(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    return float(numeric.max() - numeric.min())


def _median_catprob(group: pd.DataFrame) -> float:
    if "cat_prob_uav" not in group.columns:
        return 0.0
    catprob = pd.to_numeric(group["cat_prob_uav"], errors="coerce").dropna()
    if not len(catprob):
        return 0.0
    return float(np.clip(catprob.median(), 0.0, 1.0))


def _catprob_threshold_penalty(
    row: pd.Series,
    *,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> float:
    """Return a soft penalty for candidates below the class-probability threshold."""

    if str(getattr(config, "catprob_retention_mode", "soft")).strip().lower() != "soft":
        return 0.0
    if candidate_catprob_threshold is None or "cat_prob_uav" not in row.index:
        return 0.0
    threshold = float(candidate_catprob_threshold)
    if threshold <= 0.0:
        return 0.0
    catprob = _base._optional_float(row.get("cat_prob_uav"))
    if catprob is None or catprob >= threshold:
        return 0.0
    weight = float(
        getattr(
            config,
            "below_catprob_threshold_penalty",
            DEFAULT_BELOW_CATPROB_THRESHOLD_PENALTY,
        )
    )
    normalized_gap = (threshold - max(float(catprob), 0.0)) / threshold
    return float(weight * normalized_gap**2)


def _retain_top_and_track_representatives(
    scored: list[tuple[float, int, _base._ViterbiNode]],
    config: TrackletViterbiAssociationConfig,
) -> list[_base._ViterbiNode]:
    """Retain top unary candidates and best candidates for each track ID."""

    if not scored:
        return []

    ordered = sorted(scored, key=lambda item: (float(item[0]), int(item[1])))
    top_k = int(config.max_candidates_per_frame)
    max_pool = int(getattr(config, "max_candidate_pool_per_frame", max(2 * top_k, top_k + 8)))
    max_per_track = int(getattr(config, "max_candidates_per_track_id", 1))
    max_pool = max(max_pool, top_k)

    keep_ranks: set[int] = set()
    for _, candidate_rank, _ in ordered[:top_k]:
        keep_ranks.add(int(candidate_rank))

    if max_per_track > 0:
        kept_by_track: dict[int, int] = {}
        for _, candidate_rank, node in ordered:
            if node.track_id is None:
                continue
            track_id = int(node.track_id)
            kept = kept_by_track.get(track_id, 0)
            if kept >= max_per_track:
                continue
            keep_ranks.add(int(candidate_rank))
            kept_by_track[track_id] = kept + 1
            if len(keep_ranks) >= max_pool:
                break

    return [node for _, candidate_rank, node in ordered if int(candidate_rank) in keep_ranks][
        :max_pool
    ]
