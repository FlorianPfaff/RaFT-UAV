"""Track-aware candidate retention for tracklet-Viterbi association.

The base Viterbi implementation prunes each radar frame by local unary cost
before applying motion and Fortem track-continuity costs.  This module wraps
the base runner with a node builder that also retains per-track representatives,
so a locally weak but sequence-consistent radar track remains available to the
dynamic program.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager

import numpy as np
import pandas as pd

from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig


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
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run tracklet-Viterbi with track-aware pre-Viterbi retention."""

    with _track_aware_node_builder():
        return _base.run_async_cv_baseline_with_tracklet_viterbi_association(
            rf_measurements=rf_measurements,
            radar=radar,
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


@contextmanager
def _track_aware_node_builder():
    original = _base._nodes_for_radar_frame
    _base._nodes_for_radar_frame = _nodes_for_radar_frame_with_track_retention
    try:
        yield
    finally:
        _base._nodes_for_radar_frame = original


def _nodes_for_radar_frame_with_track_retention(
    *,
    event_index: int,
    candidates: pd.DataFrame,
    anchor: _base._AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
) -> list[_base._ViterbiNode]:
    """Build Viterbi nodes while keeping top-K plus per-track candidates."""

    from raft_uav.baselines.radar_association import _catprob_candidate_pool

    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    event_key = _base._radar_event_key(candidates)
    scored: list[tuple[float, int, _base._ViterbiNode]] = []
    for candidate_rank, (_, row) in enumerate(
        _catprob_candidate_pool(candidates, candidate_catprob_threshold).iterrows()
    ):
        position = _base._row_position(row)
        if position is None:
            continue
        anchor_nis, catprob_cost, range_cost = _base._candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=covariance,
            config=config,
        )
        unary_cost = float(config.anchor_nis_weight) * anchor_nis + catprob_cost + range_cost
        node = _base._ViterbiNode(
            event_index=event_index,
            event_key=event_key,
            time_s=float(row.get("time_s", time_s)),
            row=row.copy(),
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
