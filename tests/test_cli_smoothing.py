from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import raft_uav.cli as cli


def test_run_baseline_passes_selected_measurements_to_map_smoother(
    monkeypatch,
    tmp_path,
):
    """The CLI must give MAP smoothers the real observation factors.

    Without the ``measurements=...`` keyword, robust-map and fixed-lag-map
    smoothers fall back to posterior pseudo-measurements.  That changes the
    objective optimized by the paper-facing run-baseline command.
    """

    flight = SimpleNamespace(
        name="OptTest",
        truth_txt=tmp_path / "truth.txt",
        rf_csv=tmp_path / "rf.csv",
        radar_json=tmp_path / "radar.json",
    )
    truth = pd.DataFrame({"time_s": [0.0, 1.0]})
    rf = pd.DataFrame({"time_s": [0.0]})
    radar = pd.DataFrame({"time_s": [1.0]})
    selected_radar = pd.DataFrame({"time_s": [1.0], "track_id": [7]})

    rf_measurement = object()
    radar_measurement = object()
    expected_measurements = [rf_measurement, radar_measurement]

    monkeypatch.setattr(cli, "select_flight", lambda dataset_root, flight_name: flight)
    monkeypatch.setattr(cli, "read_truth", lambda path: object())
    monkeypatch.setattr(cli, "normalize_truth", lambda raw: (truth, object(), 0.0))
    monkeypatch.setattr(cli, "_inside_truth_window", lambda frame, truth_frame: frame)
    monkeypatch.setattr(cli, "read_rf_csv", lambda path: rf)
    monkeypatch.setattr(cli, "normalize_rf", lambda frame, projector, origin_time, **kwargs: frame)
    monkeypatch.setattr(cli, "rf_measurements_to_enu", lambda frame: [rf_measurement])
    monkeypatch.setattr(cli, "read_radar_tracks_json", lambda path: radar)
    monkeypatch.setattr(
        cli,
        "normalize_radar",
        lambda frame, projector, origin_time, **kwargs: frame,
    )
    monkeypatch.setattr(
        cli,
        "select_radar_measurement_rows",
        lambda *args, **kwargs: selected_radar,
    )
    monkeypatch.setattr(
        cli,
        "radar_measurements_to_enu",
        lambda frame: [radar_measurement],
    )

    records = [
        {
            "time_s": 0.0,
            "source": "rf",
            "state": np.zeros(6),
            "measurement_dim": 2,
            "accepted": True,
            "update_action": "updated",
        },
        {
            "time_s": 1.0,
            "source": "radar",
            "state": np.ones(6),
            "measurement_dim": 3,
            "accepted": True,
            "update_action": "updated",
        },
    ]

    def fake_run_async_cv_baseline(measurements, **kwargs):
        assert measurements == expected_measurements
        return records

    captured_smoother_kwargs = {}

    def fake_smooth_tracking_records(records_arg, **kwargs):
        captured_smoother_kwargs.update(kwargs)
        return records_arg

    monkeypatch.setattr(cli, "run_async_cv_baseline", fake_run_async_cv_baseline)
    monkeypatch.setattr(cli, "smooth_tracking_records", fake_smooth_tracking_records)
    monkeypatch.setattr(
        cli,
        "_baseline_metrics",
        lambda **kwargs: {
            "accepted_measurements": 2,
            "rejected_measurements": 0,
            "reweighted_measurements": 0,
            "selected_radar_track_ids": [7],
            "clock_offsets_s": {"rf": 0.0, "radar": 0.0},
            "radar_catprob_fallback_rows": 0,
            "position_error_2d": {"rmse_m": 0.0},
            "position_error_3d": {"rmse_m": 0.0},
        },
    )
    monkeypatch.setattr(cli, "build_diagnostic_summary", lambda **kwargs: {})
    monkeypatch.setattr(cli, "_write_trajectory_plot", lambda *args, **kwargs: None)

    exit_code = cli._run_baseline(
        dataset_root=tmp_path,
        flight_name="OptTest",
        output_dir=tmp_path / "out",
        tracker="cv",
        acceleration_std=4.0,
        imm_mode_switch_time_constant=5.0,
        rf_clock_offset_s=0.0,
        radar_clock_offset_s=0.0,
        radar_association="catprob",
        legacy_radar_selection=None,
        radar_catprob_threshold=0.5,
        radar_catprob_fallback_top_k=0,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        track_switch_nis_ratio=0.5,
        geometry_velocity_std=12.0,
        geometry_velocity_weight=0.25,
        geometry_switch_penalty=4.0,
        geometry_catprob_weight=2.0,
        rf_anchor_weight=0.0,
        rf_anchor_time_gate_s=1.0,
        rf_anchor_nis_cap=20.0,
        rf_anchor_gate_nis=20.0,
        pda_nis_temperature=1.0,
        pda_catprob_exponent=1.0,
        track_bank_max_hypotheses=16,
        track_bank_max_assignments=16,
        track_bank_max_candidates=16,
        track_bank_gate_prob=0.9999999,
        track_bank_detection_prob=0.999,
        track_bank_clutter_intensity=1.0e-12,
        track_bank_prune_delta=80.0,
        stable_segment_min_frames=3,
        stable_segment_max_transition_speed_mps=80.0,
        stable_segment_range_gate_m=0.0,
        stable_segment_interpolation_max_gap_s=0.0,
        stable_segment_interpolation_max_speed_mps=0.0,
        stable_segment_interpolation_std_scale=1.0,
        stable_segment_interpolation_gap_std_mps=0.0,
        stable_segment_rf_score_weight=0.0,
        stable_segment_rf_time_gate_s=1.0,
        stable_segment_rf_nis_cap=20.0,
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
        radar_max_residual_m=750.0,
        rf_inflation_alpha=1.0,
        radar_inflation_alpha=1.0,
    )

    assert exit_code == 0
    assert captured_smoother_kwargs["method"] == "robust-map"
    assert captured_smoother_kwargs["measurements"] == expected_measurements
