from __future__ import annotations

import pandas as pd

from raft_uav.baselines.stateful_learned_radar_association import (
    RadarDecisionNode,
    StatefulAssociationConfig,
    _BeamHypothesis,
    _apply_fixed_lag_commitment,
    _prune_hypotheses,
    _reconstruct_selected_rows,
    _score_stateful_candidates,
)


def test_stateful_cost_prefers_current_track_when_likelihoods_are_close() -> None:
    candidates = pd.DataFrame(
        [
            {"track_id": 7, "association_score": 1.0},
            {"track_id": 9, "association_score": 0.25},
        ]
    )
    config = StatefulAssociationConfig(track_switch_cost=1.0)

    scored = _score_stateful_candidates(candidates, current_track_id=7, config=config)

    same_track = scored.loc[scored["track_id"] == 7].iloc[0]
    switched_track = scored.loc[scored["track_id"] == 9].iloc[0]
    assert same_track["stateful_association_cost"] < switched_track[
        "stateful_association_cost"
    ]
    assert switched_track["stateful_track_switch_cost"] == 1.0


def test_stateful_cost_falls_back_to_probability_when_score_is_absent() -> None:
    candidates = pd.DataFrame(
        [
            {"track_id": 1, "association_learned_probability": 0.8},
            {"track_id": 1, "association_learned_probability": 0.2},
        ]
    )
    config = StatefulAssociationConfig()

    scored = _score_stateful_candidates(candidates, current_track_id=1, config=config)

    assert list(scored["association_learned_probability"]) == [0.8, 0.2]
    assert scored.iloc[0]["stateful_association_cost"] < scored.iloc[1][
        "stateful_association_cost"
    ]


def test_reconstruct_selected_rows_ignores_misses_and_preserves_time_order() -> None:
    first = pd.Series({"frame_index": 1, "track_id": 5})
    second = pd.Series({"frame_index": 3, "track_id": 5})
    node1 = RadarDecisionNode(
        parent=None,
        event_key=("frame_index", 1),
        time_s=1.0,
        selected=first,
    )
    miss = RadarDecisionNode(
        parent=node1,
        event_key=("frame_index", 2),
        time_s=2.0,
        selected=None,
    )
    node3 = RadarDecisionNode(
        parent=miss,
        event_key=("frame_index", 3),
        time_s=3.0,
        selected=second,
    )

    rows = _reconstruct_selected_rows(node3)

    assert [int(row["frame_index"]) for row in rows] == [1, 3]


def test_prune_hypotheses_keeps_lowest_costs() -> None:
    hypotheses = [
        _BeamHypothesis(None, log_cost=3.0, current_track_id=1, decision=None),
        _BeamHypothesis(None, log_cost=1.0, current_track_id=2, decision=None),
        _BeamHypothesis(None, log_cost=2.0, current_track_id=3, decision=None),
    ]

    pruned = _prune_hypotheses(hypotheses, max_hypotheses=2)

    assert [item.log_cost for item in pruned] == [1.0, 2.0]


def test_fixed_lag_commitment_discards_old_disagreement() -> None:
    selected_a = pd.Series(
        {"frame_index": 1, "track_id": 5, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0}
    )
    selected_b = pd.Series(
        {"frame_index": 1, "track_id": 8, "east_m": 2.0, "north_m": 0.0, "up_m": 0.0}
    )
    node_a = RadarDecisionNode(None, ("frame_index", 1), 1.0, selected_a)
    node_b = RadarDecisionNode(None, ("frame_index", 1), 1.0, selected_b)
    hypotheses = [
        _BeamHypothesis(None, log_cost=1.0, current_track_id=5, decision=node_a),
        _BeamHypothesis(None, log_cost=2.0, current_track_id=8, decision=node_b),
    ]

    committed = _apply_fixed_lag_commitment(hypotheses, current_time_s=5.0, lag_s=2.0)

    assert len(committed) == 1
    assert committed[0].current_track_id == 5
