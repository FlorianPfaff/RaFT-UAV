"""Backward-compatible aliases for retention-aware tracklet-Viterbi association.

Retention-aware candidate construction, soft catProb penalties, and the Fortem
track-support prior now live directly in :mod:`raft_uav.baselines.tracklet_viterbi`.
This module remains as a compatibility layer for scripts/tests that imported the
previous experimental retention module.
"""

from __future__ import annotations

from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _catprob_threshold_penalty,
    _nodes_for_radar_frame as _nodes_for_radar_frame_with_track_retention,
    _retain_top_and_track_representatives,
    _track_support_by_id,
    _track_support_cost,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)

__all__ = [
    "TrackletViterbiAssociationConfig",
    "_catprob_threshold_penalty",
    "_nodes_for_radar_frame_with_track_retention",
    "_retain_top_and_track_representatives",
    "_track_support_by_id",
    "_track_support_cost",
    "run_async_cv_baseline_with_tracklet_viterbi_association",
]
