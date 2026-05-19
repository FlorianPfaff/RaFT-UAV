from __future__ import annotations

import numpy as np

from raft_uav import tracklet_viterbi_cli
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _select_tracklet_viterbi_path,
)


def test_enabled_radar_association_modes_include_tracklet_viterbi() -> None:
    modes = tracklet_viterbi_cli.enabled_radar_association_modes()

    assert "tracklet-viterbi" in modes
    assert "rf-anchored-nis" in modes
    assert "rf-gated-nis" in modes
    assert "stable-segments" in modes
    assert "stable-segments-hybrid" in modes
    assert "stable-segments-interpolated" in modes
    assert len(modes) == len(set(modes))


def test_tracklet_viterbi_wrapper_registers_standard_association_mode(monkeypatch) -> None:
    seen = {}

    def fake_main(argv=None):
        del argv
        seen["modes"] = tracklet_viterbi_cli._base_cli.RADAR_ASSOCIATION_MODES
        seen["dispatcher"] = tracklet_viterbi_cli._base_cli.run_async_cv_baseline_with_radar_association
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli._base_cli, "main", fake_main)

    assert tracklet_viterbi_cli.main([]) == 0
    assert "tracklet-viterbi" in seen["modes"]
    assert seen["dispatcher"] is tracklet_viterbi_cli.run_async_cv_baseline_with_radar_association


def test_tracklet_viterbi_cli_strips_and_applies_tuning_flags(monkeypatch) -> None:
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = list(argv or [])
        seen["support_weight"] = tracklet_viterbi_cli._tracklet_config_from_environment().track_support_weight
        seen["max_reward"] = tracklet_viterbi_cli._tracklet_config_from_environment().max_track_support_reward
        seen["max_pool"] = tracklet_viterbi_cli._tracklet_config_from_environment().max_candidate_pool_per_frame
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli._base_cli, "main", fake_main)

    status = tracklet_viterbi_cli.main(
        [
            "run-baseline",
            "data/raw/AADM2025Dryad",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-track-support-weight",
            "0.75",
            "--tracklet-max-track-support-reward",
            "5.5",
            "--tracklet-max-candidate-pool-per-frame",
            "32",
        ]
    )

    assert status == 0
    assert "--tracklet-track-support-weight" not in seen["argv"]
    assert "--tracklet-max-track-support-reward" not in seen["argv"]
    assert "--tracklet-max-candidate-pool-per-frame" not in seen["argv"]
    assert seen["argv"] == [
        "run-baseline",
        "data/raw/AADM2025Dryad",
        "--radar-association",
        "tracklet-viterbi",
    ]
    assert seen["support_weight"] == 0.75
    assert seen["max_reward"] == 5.5
    assert seen["max_pool"] == 32


def test_tracklet_viterbi_cli_strips_fixed_lag_flags(monkeypatch) -> None:
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = list(argv or [])
        seen["variant"] = tracklet_viterbi_cli._tracklet_runner_from_environment()
        seen["lag_s"] = tracklet_viterbi_cli._env_float(
            tracklet_viterbi_cli._VITERBI_LAG_S_ENV,
            20.0,
        )
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli._base_cli, "main", fake_main)

    status = tracklet_viterbi_cli.main(
        [
            "run-baseline",
            "data/raw/AADM2025Dryad",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-variant",
            "fixed-lag",
            "--tracklet-viterbi-lag-s",
            "15.0",
        ]
    )

    assert status == 0
    assert "--tracklet-variant" not in seen["argv"]
    assert "--tracklet-viterbi-lag-s" not in seen["argv"]
    assert seen["lag_s"] == 15.0
    assert seen["variant"] is tracklet_viterbi_cli._run_fixed_lag_tracklet_viterbi_association


def test_tracklet_viterbi_cli_restores_environment(monkeypatch) -> None:
    monkeypatch.setenv(tracklet_viterbi_cli._TRACK_SUPPORT_WEIGHT_ENV, "0.25")

    def fake_main(argv=None):
        del argv
        assert tracklet_viterbi_cli._tracklet_config_from_environment().track_support_weight == 0.9
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli._base_cli, "main", fake_main)

    assert tracklet_viterbi_cli.main(["--tracklet-track-support-weight", "0.9"]) == 0
    assert tracklet_viterbi_cli._tracklet_config_from_environment().track_support_weight == 0.25


def test_tracklet_viterbi_empty_events_returns_no_rows() -> None:
    selected = _select_tracklet_viterbi_path(
        events=[],
        anchors={},
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(),
    )
    assert selected == []
