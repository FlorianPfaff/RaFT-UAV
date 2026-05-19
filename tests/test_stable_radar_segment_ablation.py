from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_stable_radar_segment_ablation as ablation  # noqa: E402


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "catprob_thresholds": [0.4],
        "range_gates_m": [800.0],
        "interpolation_max_gaps_s": [0.0],
        "interpolation_max_speeds_mps": [0.0],
        "min_segment_frames": [100],
        "max_transition_speeds_mps": [65.0],
        "ranking_min_coverage": 0.95,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_configs_build_stable_segment_grid() -> None:
    configs = ablation._configs(
        _args(
            catprob_thresholds=[0.4, 0.5],
            min_segment_frames=[75, 100],
            max_transition_speeds_mps=[35.0],
        )
    )

    assert len(configs) == 4
    assert configs[0].name == "stable_cat0p40_rg800_gapnone_isnone_min75_v35"
    assert configs[-1].name == "stable_cat0p50_rg800_gapnone_isnone_min100_v35"


def test_rows_from_table_extracts_only_stable_ablation_methods(tmp_path: Path) -> None:
    table_path = tmp_path / "config" / "Opt1" / "paper_table.csv"
    table_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "RF raw",
                "candidate_count": 10,
                "selected_count": 10,
                "coverage": 1.0,
            },
            {
                "method": "radar-longest-track-range-gated-interpolated",
                "candidate_count": 100,
                "selected_count": 90,
                "matched_count": 90,
                "coverage": 0.9,
                "track_switches": 0,
            },
            {
                "method": "radar-stable-segments-range-gated-interpolated",
                "candidate_count": 100,
                "selected_count": 100,
                "selected_interpolated_count": 100,
                "selected_interpolated_fraction": 1.0,
                "matched_count": 100,
                "coverage": 1.0,
                "track_switches": 0,
                "interpolation_anchor_count": 50,
                "interpolation_max_anchor_gap_s": 3.0,
                "interpolation_candidate_frame_count": 120,
                "interpolation_dropped_frame_count": 20,
                "error_3d_mean_m": 12.34567,
                "error_3d_rmse_m": 20.0,
                "error_3d_p95_m": 40.0,
                "error_3d_max_m": 50.0,
                "error_2d_mean_m": 10.0,
                "error_2d_rmse_m": 18.0,
                "error_2d_p95_m": 35.0,
            },
        ]
    ).to_csv(table_path, index=False)
    config = ablation.StableSegmentConfig(
        name="stable_cat0p40_rg800_gapnone_isnone_min100_v65",
        radar_catprob_threshold=0.4,
        radar_range_gate_m=800.0,
        radar_interpolation_max_gap_s=None,
        radar_interpolation_max_speed_mps=None,
        stable_segment_min_frames=100,
        stable_segment_max_transition_speed_mps=65.0,
    )

    rows = ablation._rows_from_table(config, table_path)
    stable_row = next(
        row
        for row in rows
        if row["method"] == "radar-stable-segments-range-gated-interpolated"
    )

    assert len(rows) == 2
    assert {row["method"] for row in rows} == {
        "radar-longest-track-range-gated-interpolated",
        "radar-stable-segments-range-gated-interpolated",
    }
    assert stable_row["flight"] == "Opt1"
    assert stable_row["config"] == "stable_cat0p40_rg800_gapnone_isnone_min100_v65"
    assert stable_row["error_3d_mean_m"] == 12.346
    assert stable_row["radar_interpolation_max_gap_s"] == ""
    assert stable_row["radar_interpolation_max_speed_mps"] == ""
    assert stable_row["selected_interpolated_count"] == 100
    assert stable_row["selected_interpolated_fraction"] == 1.0
    assert stable_row["interpolation_anchor_count"] == 50
    assert stable_row["interpolation_max_anchor_gap_s"] == 3.0
    assert stable_row["interpolation_candidate_frame_count"] == 120
    assert stable_row["interpolation_dropped_frame_count"] == 20
    assert stable_row["stable_segment_min_frames"] == 100


def test_aggregate_and_ranking_rows_sort_by_mean_then_tail() -> None:
    rows = [
        {
            "flight": "Opt1",
            "method": "radar-stable-segments-range-gated",
            "config": "stable_a",
            "radar_catprob_threshold": 0.4,
            "radar_range_gate_m": 800.0,
            "radar_interpolation_max_gap_s": "",
            "radar_interpolation_max_speed_mps": "",
            "stable_segment_min_frames": 75,
            "stable_segment_max_transition_speed_mps": 65.0,
            "flight_count": 1,
            "candidate_count": 10,
            "selected_count": 8,
            "matched_count": 8,
            "coverage": 0.8,
            "track_switches": 1,
            "error_3d_mean_m": 50.0,
            "error_3d_rmse_m": 60.0,
            "error_3d_p95_m": 100.0,
            "error_3d_max_m": 120.0,
            "error_2d_mean_m": 40.0,
            "error_2d_rmse_m": 55.0,
            "error_2d_p95_m": 90.0,
            "table_path": "a",
        },
        {
            "flight": "Opt2",
            "method": "radar-stable-segments-range-gated",
            "config": "stable_a",
            "radar_catprob_threshold": 0.4,
            "radar_range_gate_m": 800.0,
            "radar_interpolation_max_gap_s": "",
            "radar_interpolation_max_speed_mps": "",
            "stable_segment_min_frames": 75,
            "stable_segment_max_transition_speed_mps": 65.0,
            "flight_count": 1,
            "candidate_count": 20,
            "selected_count": 12,
            "matched_count": 10,
            "coverage": 0.5,
            "track_switches": 2,
            "error_3d_mean_m": 70.0,
            "error_3d_rmse_m": 80.0,
            "error_3d_p95_m": 120.0,
            "error_3d_max_m": 180.0,
            "error_2d_mean_m": 60.0,
            "error_2d_rmse_m": 75.0,
            "error_2d_p95_m": 110.0,
            "table_path": "b",
        },
        {
            "flight": "Opt1",
            "method": "radar-stable-segments-range-gated",
            "config": "stable_b",
            "radar_catprob_threshold": 0.5,
            "radar_range_gate_m": 800.0,
            "radar_interpolation_max_gap_s": "",
            "radar_interpolation_max_speed_mps": "",
            "stable_segment_min_frames": 100,
            "stable_segment_max_transition_speed_mps": 65.0,
            "flight_count": 1,
            "candidate_count": 10,
            "selected_count": 10,
            "matched_count": 10,
            "coverage": 1.0,
            "track_switches": 0,
            "error_3d_mean_m": 40.0,
            "error_3d_rmse_m": 70.0,
            "error_3d_p95_m": 150.0,
            "error_3d_max_m": 200.0,
            "error_2d_mean_m": 35.0,
            "error_2d_rmse_m": 65.0,
            "error_2d_p95_m": 140.0,
            "table_path": "c",
        },
        {
            "flight": "Opt1",
            "method": "radar-longest-track-range-gated",
            "config": "low_coverage",
            "radar_catprob_threshold": 0.4,
            "radar_range_gate_m": 800.0,
            "radar_interpolation_max_gap_s": "",
            "radar_interpolation_max_speed_mps": "",
            "stable_segment_min_frames": 100,
            "stable_segment_max_transition_speed_mps": 65.0,
            "flight_count": 1,
            "candidate_count": 10,
            "selected_count": 2,
            "matched_count": 2,
            "coverage": 0.2,
            "track_switches": 0,
            "error_3d_mean_m": 5.0,
            "error_3d_rmse_m": 10.0,
            "error_3d_p95_m": 20.0,
            "error_3d_max_m": 25.0,
            "error_2d_mean_m": 4.0,
            "error_2d_rmse_m": 9.0,
            "error_2d_p95_m": 18.0,
            "table_path": "d",
        },
        {
            "flight": "Opt1",
            "method": "radar-stable-segments-range-gated-interpolated",
            "config": "dominated",
            "radar_catprob_threshold": 0.4,
            "radar_range_gate_m": 800.0,
            "radar_interpolation_max_gap_s": "",
            "radar_interpolation_max_speed_mps": "",
            "stable_segment_min_frames": 100,
            "stable_segment_max_transition_speed_mps": 65.0,
            "flight_count": 1,
            "candidate_count": 10,
            "selected_count": 5,
            "matched_count": 5,
            "coverage": 0.5,
            "track_switches": 0,
            "error_3d_mean_m": 80.0,
            "error_3d_rmse_m": 100.0,
            "error_3d_p95_m": 170.0,
            "error_3d_max_m": 220.0,
            "error_2d_mean_m": 70.0,
            "error_2d_rmse_m": 90.0,
            "error_2d_p95_m": 160.0,
            "table_path": "e",
        },
    ]

    aggregate_rows = ablation._aggregate_rows(rows)
    ranking_rows = ablation._ranking_rows(aggregate_rows, min_coverage=0.95)

    stable_a = next(row for row in aggregate_rows if row["config"] == "stable_a")
    assert stable_a["flight"] == "aggregate"
    assert stable_a["flight_count"] == 2
    assert stable_a["candidate_count"] == 30
    assert stable_a["selected_count"] == 20
    assert stable_a["selected_interpolated_count"] == 0
    assert stable_a["selected_interpolated_fraction"] == 0.0
    assert stable_a["matched_count"] == 18
    assert stable_a["coverage"] == 0.6
    assert stable_a["track_switches"] == 3
    assert stable_a["error_3d_mean_m"] == 60.0
    assert ranking_rows[0]["rank"] == 1
    assert ranking_rows[0]["config"] == "stable_b"
    assert ranking_rows[0]["eligible_for_recommendation"] is True
    assert [row["eligible_for_recommendation"] for row in ranking_rows] == [
        True,
        False,
        False,
        False,
    ]
    assert ranking_rows[0]["coverage_penalized_error_3d_mean_m"] == 40.0
    assert ranking_rows[0]["coverage_penalized_error_3d_p95_m"] == 150.0
    assert ranking_rows[0]["pareto_front"] is True
    low_coverage = next(row for row in ranking_rows if row["config"] == "low_coverage")
    assert low_coverage["eligible_for_recommendation"] is False
    assert low_coverage["ranking_min_coverage"] == 0.95
    assert low_coverage["coverage_penalized_error_3d_mean_m"] == 25.0
    assert low_coverage["pareto_front"] is True
    dominated = next(row for row in ranking_rows if row["config"] == "dominated")
    assert dominated["pareto_front"] is False


def test_recommendation_payload_selects_decision_rows(tmp_path: Path) -> None:
    ranking_rows = [
        {
            "rank": 1,
            "eligible_for_recommendation": True,
            "ranking_min_coverage": 0.95,
            "method": "radar-stable-segments-range-gated-interpolated",
            "config": "stable_full",
            "coverage": 1.0,
            "coverage_penalized_error_3d_mean_m": 80.0,
            "coverage_penalized_error_3d_p95_m": 150.0,
            "pareto_front": True,
            "error_3d_mean_m": 80.0,
            "error_3d_p95_m": 150.0,
        },
        {
            "rank": 2,
            "eligible_for_recommendation": False,
            "ranking_min_coverage": 0.95,
            "method": "radar-longest-track-range-gated",
            "config": "partial_clean",
            "coverage": 0.4,
            "coverage_penalized_error_3d_mean_m": 100.0,
            "coverage_penalized_error_3d_p95_m": 200.0,
            "pareto_front": True,
            "error_3d_mean_m": 40.0,
            "error_3d_p95_m": 80.0,
        },
    ]

    payload = ablation._recommendation_payload(
        ranking_rows,
        summary_output=tmp_path / "summary.csv",
        ranking_output=tmp_path / "ranking.csv",
        min_coverage=0.95,
    )
    output = tmp_path / "recommendation.json"
    ablation._write_recommendation(output, payload)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    assert loaded["ranking_rows"] == 2
    assert loaded["eligible_rows"] == 1
    assert loaded["pareto_front_rows"] == 2
    assert loaded["best_eligible"]["config"] == "stable_full"
    assert loaded["best_ineligible_pareto_front"]["config"] == "partial_clean"


def test_validate_args_rejects_empty_and_nonpositive_grids() -> None:
    for kwargs in (
        {"catprob_thresholds": []},
        {"range_gates_m": [0.0]},
        {"interpolation_max_gaps_s": [-1.0]},
        {"interpolation_max_speeds_mps": [-1.0]},
        {"min_segment_frames": [0]},
        {"max_transition_speeds_mps": [0.0]},
        {"ranking_min_coverage": 1.1},
    ):
        try:
            ablation._validate_args(_args(**kwargs))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected invalid grid to fail: {kwargs}")
