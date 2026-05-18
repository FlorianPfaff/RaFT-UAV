import argparse
import os

import pytest

from raft_uav.baselines.tracklet_viterbi_runtime import _config_from_environment
from raft_uav.runtime_cli_config import (
    add_runtime_configuration_arguments,
    apply_runtime_environment,
    parse_runtime_config,
    runtime_config_from_args,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args(argv)


def test_runtime_config_from_args_records_tracklet_and_radar_settings():
    args = _parse_args(
        [
            "--radar-covariance-mode",
            "fixed",
            "--radar-range-std-m",
            "8",
            "--radar-origin-east-m",
            "12.5",
            "--tracklet-track-switch-cost",
            "14",
            "--tracklet-anchor-nis-weight",
            "0.6",
            "--tracklet-range-gate-m",
            "0",
            "--tracklet-catprob-retention-mode",
            "hard",
            "--tracklet-below-catprob-threshold-penalty",
            "2.25",
            "--tracklet-track-support-weight",
            "0.75",
            "--tracklet-max-track-support-reward",
            "5.5",
            "--tracklet-max-candidate-pool-per-frame",
            "32",
            "--tracklet-max-candidates-per-track-id",
            "2",
            "--disable-tracklet-rf-anchor",
        ]
    )

    config = runtime_config_from_args(args)

    assert config["radar_covariance"]["mode"] == "fixed"
    assert config["radar_covariance"]["range_std_m"] == 8.0
    assert config["radar_covariance"]["origin_east_m"] == 12.5
    assert config["tracklet_viterbi"]["track_switch_cost"] == 14.0
    assert config["tracklet_viterbi"]["anchor_nis_weight"] == 0.6
    assert config["tracklet_viterbi"]["range_gate_m"] is None
    assert config["tracklet_viterbi"]["catprob_retention_mode"] == "hard"
    assert config["tracklet_viterbi"]["below_catprob_threshold_penalty"] == 2.25
    assert config["tracklet_viterbi"]["track_support_weight"] == 0.75
    assert config["tracklet_viterbi"]["max_track_support_reward"] == 5.5
    assert config["tracklet_viterbi"]["max_candidate_pool_per_frame"] == 32
    assert config["tracklet_viterbi"]["max_candidates_per_track_id"] == 2
    assert config["tracklet_viterbi"]["use_rf_anchor"] is False


def test_parse_runtime_config_preserves_standard_cli_arguments():
    config, remaining = parse_runtime_config(
        [
            "run-baseline",
            "/data/aerpaw",
            "--flight",
            "Opt1",
            "--tracklet-track-switch-cost",
            "16",
            "--tracklet-track-support-weight",
            "0.9",
            "--radar-covariance-mode",
            "fixed",
            "--smoother",
            "fixed-lag",
        ]
    )

    assert remaining == [
        "run-baseline",
        "/data/aerpaw",
        "--flight",
        "Opt1",
        "--smoother",
        "fixed-lag",
    ]
    assert config["tracklet_viterbi"]["track_switch_cost"] == 16.0
    assert config["tracklet_viterbi"]["track_support_weight"] == 0.9
    assert config["radar_covariance"]["mode"] == "fixed"


def test_apply_runtime_environment_sets_expected_variables(monkeypatch):
    args = _parse_args(
        [
            "--radar-covariance-mode",
            "range-angle",
            "--radar-azimuth-std-deg",
            "3",
            "--tracklet-max-candidates",
            "12",
            "--tracklet-track-switch-cost",
            "16",
            "--tracklet-catprob-retention-mode",
            "hard",
            "--tracklet-track-support-weight",
            "0.8",
            "--tracklet-max-candidate-pool-per-frame",
            "20",
            "--tracklet-max-candidates-per-track-id",
            "2",
        ]
    )
    config = runtime_config_from_args(args)

    apply_runtime_environment(config)
    parsed = _config_from_environment()

    assert os.environ["RAFT_UAV_RADAR_COVARIANCE_MODE"] == "range-angle"
    assert os.environ["RAFT_UAV_RADAR_AZIMUTH_STD_DEG"] == "3.0"
    assert os.environ["RAFT_UAV_TRACKLET_MAX_CANDIDATES"] == "12"
    assert os.environ["RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_FRAME"] == "12"
    assert os.environ["RAFT_UAV_TRACKLET_TRACK_SWITCH_COST"] == "16.0"
    assert os.environ["RAFT_UAV_TRACKLET_CATPROB_RETENTION_MODE"] == "hard"
    assert os.environ["RAFT_UAV_TRACKLET_SUPPORT_WEIGHT"] == "0.8"
    assert os.environ["RAFT_UAV_TRACKLET_MAX_CANDIDATE_POOL_PER_FRAME"] == "20"
    assert os.environ["RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_TRACK_ID"] == "2"
    assert parsed.max_candidates_per_frame == 12
    assert parsed.track_switch_cost == 16.0
    assert parsed.catprob_retention_mode == "hard"
    assert parsed.track_support_weight == 0.8
    assert parsed.max_candidate_pool_per_frame == 20
    assert parsed.max_candidates_per_track_id == 2


def test_runtime_config_rejects_invalid_radar_covariance_range():
    args = _parse_args(
        [
            "--radar-covariance-min-std-m",
            "10",
            "--radar-covariance-max-std-m",
            "5",
        ]
    )

    with pytest.raises(ValueError, match="max_std"):
        runtime_config_from_args(args)
