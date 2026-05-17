from __future__ import annotations

import numpy as np

from raft_uav import tracklet_viterbi_cli
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _select_tracklet_viterbi_path,
)


def test_tracklet_viterbi_wrapper_registers_standard_association_mode(monkeypatch) -> None:
    seen = {}

    def fake_main(argv=None):
        del argv
        seen["modes"] = tracklet_viterbi_cli._base_cli.RADAR_ASSOCIATION_MODES
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli._base_cli, "main", fake_main)

    assert tracklet_viterbi_cli.main([]) == 0
    assert "tracklet-viterbi" in seen["modes"]


def test_tracklet_viterbi_empty_events_returns_no_rows() -> None:
    selected = _select_tracklet_viterbi_path(
        events=[],
        anchors={},
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(),
    )
    assert selected == []
