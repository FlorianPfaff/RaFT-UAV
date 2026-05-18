from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_lofo_radar_covariance_tuning.py"
spec = importlib.util.spec_from_file_location("run_lofo_radar_covariance_tuning", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_parse_float_list_rejects_nonpositive_values():
    assert module._parse_float_list("1,2.5") == [1.0, 2.5]
    try:
        module._parse_float_list("1,0")
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_candidate_environment_uses_raft_uav_radar_variables():
    candidate = module.RadarCovarianceCandidate(
        candidate_id="cov0000",
        range_std_m=5.0,
        azimuth_std_deg=2.0,
        elevation_std_deg=3.0,
        min_std_m=4.0,
        max_std_m=250.0,
    )

    env = candidate.environment()

    assert env["RAFT_UAV_RADAR_COVARIANCE_MODE"] == "range-angle"
    assert env["RAFT_UAV_RADAR_RANGE_STD_M"] == "5"
    assert env["RAFT_UAV_RADAR_AZIMUTH_STD_DEG"] == "2"
    assert env["RAFT_UAV_RADAR_ELEVATION_STD_DEG"] == "3"
    assert env["RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M"] == "4"
    assert env["RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M"] == "250"


def test_select_candidate_aggregates_training_metric():
    sweep = pd.DataFrame(
        [
            {"candidate_id": "bad", "metric_value": 10.0},
            {"candidate_id": "bad", "metric_value": 12.0},
            {"candidate_id": "good", "metric_value": 8.0},
            {"candidate_id": "good", "metric_value": 9.0},
        ]
    )

    selected = module._select_candidate(sweep, metric_column="metric_value", aggregate="mean")

    assert selected["candidate_id"] == "good"
    assert selected["aggregate_metric_value"] == 8.5
    assert selected["finite_train_flights"] == 2
