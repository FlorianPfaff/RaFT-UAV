from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from raft_uav import cli
from raft_uav.baselines.kalman import TrackingMeasurement


def test_run_baseline_forwards_real_measurements_to_smoother(tmp_path, monkeypatch):
    """Regression test for robust-map using posterior pseudo-measurements in CLI."""

    rf_measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        source="rf",
    )
    radar_measurement = TrackingMeasurement(
        time_s=1.0,
        vector=np.array([1.0, 0.0, 0.0]),
        covariance=np.eye(3),
        source="radar",
    )
    captured: dict[str, list[TrackingMeasurement]] = {}

    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    rf = pd.DataFrame({"time_s": [0.0]})
    radar = pd.DataFrame({"time_s": [1.0]})
    selected_radar = pd.DataFrame({"time_s": [1.0], "track_id": [7]})
    flight = SimpleNamespace(
        name="synthetic",
        truth_txt=Path("truth.txt"),
        rf_csv=Path("rf.csv"),
        radar_json=Path("radar.json"),
    )

    monkeypatch.setattr(cli, "select_flight", lambda _root, _name: flight)
    monkeypatch.setattr(cli, "read_truth", lambda _path: pd.DataFrame())
    monkeypatch.setattr(cli, "normalize_truth", lambda _raw: (truth, object(), 0.0))
    monkeypatch.setattr(cli, "read_rf_csv", lambda _path: pd.DataFrame())
    monkeypatch.setattr(cli, "normalize_rf", lambda _raw, _projector, _origin: rf)
    monkeypatch.setattr(cli, "read_radar_tracks_json", lambda _path: pd.DataFrame())
    monkeypatch.setattr(cli, "normalize_radar", lambda _raw, _projector, _origin: radar)
    monkeypatch.setattr(cli, "_inside_truth_window", lambda frame, _truth: frame)
    monkeypatch.setattr(cli, "rf_measurements_to_enu", lambda _rf: [rf_measurement])
    monkeypatch.setattr(
        cli,
        "radar_measurements_to_enu",
        lambda _radar: [radar_measurement],
    )
    monkeypatch.setattr(
        cli,
        "select_radar_measurement_rows",
        lambda _radar, **_kwargs: selected_radar,
    )

    def fake_run_async_cv_baseline(measurements, **_kwargs):
        captured["filter_measurements"] = list(measurements)
        return [
            {
                "time_s": 1.0,
                "source": "radar",
                "track_id": 7,
                "measurement_dim": 3,
                "accepted": True,
                "update_action": "updated",
                "state": np.zeros(6),
                "covariance": np.eye(6),
            }
        ]

    def fake_smooth_tracking_records(records, **kwargs):
        captured["smoother_measurements"] = list(kwargs["measurements"])
        return records

    monkeypatch.setattr(cli, "run_async_cv_baseline", fake_run_async_cv_baseline)
    monkeypatch.setattr(cli, "smooth_tracking_records", fake_smooth_tracking_records)
    monkeypatch.setattr(
        cli,
        "_baseline_metrics",
        lambda **_kwargs: {
            "accepted_measurements": 1,
            "rejected_measurements": 0,
            "reweighted_measurements": 0,
            "selected_radar_track_ids": [7],
            "position_error_2d": {"rmse_m": 0.0},
            "position_error_3d": {"rmse_m": 0.0},
        },
    )
    monkeypatch.setattr(cli, "build_diagnostic_summary", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "_write_trajectory_plot", lambda *_args, **_kwargs: None)

    result = cli._run_baseline(
        dataset_root=tmp_path,
        flight_name="synthetic",
        output_dir=tmp_path / "out",
        acceleration_std=4.0,
        radar_association="catprob",
        legacy_radar_selection=None,
        radar_catprob_threshold=0.5,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        track_switch_nis_ratio=0.5,
        geometry_velocity_std=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        track_bank_max_hypotheses=16,
        track_bank_max_assignments=16,
        track_bank_max_candidates=16,
        track_bank_gate_prob=0.9999999,
        track_bank_detection_prob=0.999,
        track_bank_clutter_intensity=1.0e-12,
        track_bank_prune_delta=80.0,
        stable_segment_min_frames=100,
        stable_segment_max_transition_speed_mps=65.0,
        stable_segment_range_gate_m=800.0,
        smoother="robust-map",
        smoother_lag_s=20.0,
        max_eval_time_delta_s=2.0,
        enable_gating=False,
        robust_update="none",
        rf_gate_prob=0.99,
        radar_gate_prob=0.99,
        enable_association_safety_gate=True,
        rf_safety_gate_prob=0.9999999,
        radar_safety_gate_prob=0.9999999,
        rf_max_residual_m=750.0,
        radar_max_residual_m=0.0,
        rf_inflation_alpha=1.0,
        radar_inflation_alpha=1.0,
    )

    assert result == 0
    assert captured["filter_measurements"][0] is rf_measurement
    assert captured["filter_measurements"][1] is radar_measurement
    assert captured["smoother_measurements"][0] is rf_measurement
    assert captured["smoother_measurements"][1] is radar_measurement
