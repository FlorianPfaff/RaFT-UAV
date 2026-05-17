from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_retention import (
    _catprob_threshold_penalty,
    _nodes_for_radar_frame_with_track_retention,
)


def _radar_frame(frame_index: int, rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["frame_index"] = frame_index
    frame["time_s"] = float(frame_index)
    frame["up_m"] = frame.get("up_m", 0.0)
    return frame


def test_track_aware_retention_keeps_per_track_representatives() -> None:
    candidates = _radar_frame(
        0,
        [
            {"track_id": 10, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 11, "east_m": 1.0, "north_m": 0.0, "cat_prob_uav": 0.95},
            {"track_id": 12, "east_m": 2.0, "north_m": 0.0, "cat_prob_uav": 0.05},
        ],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=config,
    )

    retained_track_ids = {node.track_id for node in nodes if not node.is_miss}
    assert retained_track_ids == {10, 11, 12}


def test_track_aware_retention_keeps_below_threshold_track_with_soft_penalty() -> None:
    candidates = _radar_frame(
        0,
        [
            {"track_id": 10, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 11, "east_m": 1.0, "north_m": 0.0, "cat_prob_uav": 0.95},
            {"track_id": 12, "east_m": 2.0, "north_m": 0.0, "cat_prob_uav": 0.05},
        ],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=0.4,
        config=config,
    )

    retained = {node.track_id: node for node in nodes if not node.is_miss}
    assert set(retained) == {10, 11, 12}
    low_catprob = retained[12]
    assert low_catprob.row is not None
    assert bool(low_catprob.row["association_catprob_below_threshold"])
    assert float(low_catprob.row["association_catprob_soft_penalty"]) > 0.0


def test_catprob_threshold_penalty_increases_with_gap() -> None:
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    near = pd.Series({"cat_prob_uav": 0.35})
    far = pd.Series({"cat_prob_uav": 0.05})

    near_penalty = _catprob_threshold_penalty(
        near,
        candidate_catprob_threshold=0.4,
        config=config,
    )
    far_penalty = _catprob_threshold_penalty(
        far,
        candidate_catprob_threshold=0.4,
        config=config,
    )

    assert near_penalty > 0.0
    assert far_penalty > near_penalty


def test_track_aware_retention_still_keeps_missed_detection_node() -> None:
    candidates = _radar_frame(
        0,
        [{"track_id": 1, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99}],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=config,
    )

    assert any(node.is_miss for node in nodes)
