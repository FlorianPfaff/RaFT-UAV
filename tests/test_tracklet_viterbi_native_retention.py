from __future__ import annotations

from raft_uav.baselines import tracklet_viterbi as base
from raft_uav.baselines import tracklet_viterbi_retention as retention


def test_retention_module_keeps_base_node_builder_identity() -> None:
    original = base._nodes_for_radar_frame

    assert retention.TrackletViterbiAssociationConfig is base.TrackletViterbiAssociationConfig
    assert base._nodes_for_radar_frame is original


def test_retention_exports_local_node_builder() -> None:
    assert callable(retention._nodes_for_radar_frame_with_track_retention)
    assert retention._nodes_for_radar_frame_with_track_retention is not base._nodes_for_radar_frame
