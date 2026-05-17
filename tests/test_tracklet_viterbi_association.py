from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _AnchorState,
    _select_tracklet_viterbi_path,
    _transition_cost,
    _ViterbiNode,
)


def _radar_frame(frame_index: int, rows: list[dict[str, float]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    frame["frame_index"] = frame_index
    frame["time_s"] = float(frame_index)
    frame["up_m"] = frame.get("up_m", 0.0)
    return {"kind": "radar", "time_s": float(frame_index), "candidates": frame}


def _anchor(east_m: float) -> _AnchorState:
    state = np.array([east_m, 0.0, 0.0, 10.0, 0.0, 0.0])
    covariance = np.diag([5.0**2, 5.0**2, 5.0**2, 5.0**2, 5.0**2, 5.0**2])
    return _AnchorState(state=state, covariance=covariance)


def test_tracklet_viterbi_prefers_sequence_consistency_over_local_catprob() -> None:
    events = [
        _radar_frame(
            0,
            [
                {"track_id": 1, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.85},
                {"track_id": 2, "east_m": 0.2, "north_m": 0.0, "cat_prob_uav": 0.99},
            ],
        ),
        _radar_frame(
            1,
            [
                {"track_id": 1, "east_m": 10.0, "north_m": 0.0, "cat_prob_uav": 0.85},
                {"track_id": 2, "east_m": 200.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            ],
        ),
    ]
    covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    anchors = {0: _anchor(0.0), 1: _anchor(10.0)}
    config = TrackletViterbiAssociationConfig(
        track_switch_cost=12.0,
        anchor_nis_weight=1.0,
        max_candidates_per_frame=4,
        range_gate_m=None,
    )

    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=None,
        config=config,
    )

    assert [int(row["track_id"]) for row in selected] == [1, 1]
    assert all(row["association_mode"] == "tracklet-viterbi" for row in selected)


def test_tracklet_viterbi_uses_missed_detection_for_bad_frame() -> None:
    events = [
        _radar_frame(
            0,
            [{"track_id": 1, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.9}],
        ),
        _radar_frame(
            1,
            [{"track_id": 99, "east_m": 800.0, "north_m": 0.0, "cat_prob_uav": 0.99}],
        ),
        _radar_frame(
            2,
            [{"track_id": 1, "east_m": 20.0, "north_m": 0.0, "cat_prob_uav": 0.9}],
        ),
    ]
    covariance = np.diag([10.0**2, 10.0**2, 10.0**2])
    anchors = {0: _anchor(0.0), 1: _anchor(10.0), 2: _anchor(20.0)}
    config = TrackletViterbiAssociationConfig(
        missed_detection_cost=1.0,
        anchor_nis_weight=2.0,
        track_switch_cost=10.0,
        max_candidates_per_frame=4,
        range_gate_m=None,
    )

    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=None,
        config=config,
    )

    assert [int(row["frame_index"]) for row in selected] == [0, 2]
    assert [int(row["track_id"]) for row in selected] == [1, 1]


def test_transition_cost_penalizes_track_switches() -> None:
    config = TrackletViterbiAssociationConfig(track_switch_cost=9.0, range_gate_m=None)
    previous = _ViterbiNode(
        event_index=0,
        event_key=("frame_index", 0),
        time_s=0.0,
        row=None,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([10.0, 0.0, 0.0]),
        track_id=1,
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )
    same = _ViterbiNode(
        event_index=1,
        event_key=("frame_index", 1),
        time_s=1.0,
        row=None,
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([10.0, 0.0, 0.0]),
        track_id=1,
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )
    switched = _ViterbiNode(
        event_index=1,
        event_key=("frame_index", 1),
        time_s=1.0,
        row=None,
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([10.0, 0.0, 0.0]),
        track_id=2,
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )

    assert _transition_cost(previous, switched, config) > _transition_cost(previous, same, config)
