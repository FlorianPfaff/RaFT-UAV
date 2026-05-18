import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _select_tracklet_viterbi_path,
)
from raft_uav.baselines.tracklet_viterbi_result import _tracklet_candidate_ledger


def test_tracklet_candidate_ledger_marks_viterbi_choice():
    radar = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "frame_index": 7,
                "track_id": 11,
                "track_index": 0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "time_s": 1.0,
                "frame_index": 7,
                "track_id": 12,
                "track_index": 1,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )
    events = [{"kind": "radar", "time_s": 1.0, "candidates": radar}]
    config = TrackletViterbiAssociationConfig(
        max_candidates_per_frame=2,
        catprob_weight=0.0,
        use_rf_anchor=False,
    )
    covariance = np.eye(3)

    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors={},
        covariance=covariance,
        candidate_catprob_threshold=None,
        config=config,
    )
    ledger = _tracklet_candidate_ledger(
        events=events,
        anchors={},
        covariance=covariance,
        candidate_catprob_threshold=None,
        config=config,
        selected_rows=selected,
    )

    assert len(ledger) == 2
    assert ledger["association_candidate_rank"].tolist() == [0, 1]
    assert ledger["association_viterbi_selected"].tolist() == [True, False]
    assert ledger.loc[0, "track_id"] == selected[0]["track_id"]
    assert ledger.loc[0, "association_score"] == selected[0]["association_score"]
