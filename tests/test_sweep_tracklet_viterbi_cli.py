import json

from raft_uav.sweep_tracklet_viterbi_cli import (
    TrackletSweepConfig,
    build_sweep_configs,
    flatten_metrics,
    parse_float_grid,
    parse_int_grid,
)


def test_parse_grids_and_build_configs():
    assert parse_float_grid("1,2.5") == (1.0, 2.5)
    assert parse_int_grid("3,4") == (3, 4)
    configs = build_sweep_configs(
        track_switch_costs=(1.0, 2.0),
        anchor_nis_weights=(0.1,),
        missed_detection_costs=(5.0,),
        max_candidates=(4, 6),
    )
    assert len(configs) == 4
    assert configs[0].config_id == "sw1_anc0p1_miss5_cand4"
    assert configs[-1].max_candidates == 6


def test_tracklet_config_environment():
    config = TrackletSweepConfig(
        track_switch_cost=12.0,
        anchor_nis_weight=0.5,
        missed_detection_cost=7.0,
        max_candidates=8,
    )
    assert config.environment()["RAFT_UAV_TRACKLET_TRACK_SWITCH_COST"] == "12.0"
    assert config.environment()["RAFT_UAV_TRACKLET_ANCHOR_NIS_WEIGHT"] == "0.5"
    assert config.environment()["RAFT_UAV_TRACKLET_MISSED_DETECTION_COST"] == "7.0"
    assert config.environment()["RAFT_UAV_TRACKLET_MAX_CANDIDATES"] == "8"


def test_flatten_metrics_extracts_key_scores(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "position_error_2d": {"rmse_m": 1.0, "p95_m": 2.0},
                "position_error_3d": {"rmse_m": 3.0, "p95_m": 4.0},
                "accepted_measurements": 5,
                "rejected_measurements": 6,
                "selected_radar_rows": 7,
                "selected_radar_track_ids": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    flattened = flatten_metrics(metrics_path)
    assert flattened["rmse_3d_m"] == 3.0
    assert flattened["p95_3d_m"] == 4.0
    assert flattened["selected_radar_track_ids"] == "1,2"
