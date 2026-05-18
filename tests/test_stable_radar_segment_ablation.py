from __future__ import annotations

import argparse
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
        "min_segment_frames": [100],
        "max_transition_speeds_mps": [65.0],
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
    assert configs[0].name == "stable_cat0p40_rg800_min75_v35"
    assert configs[-1].name == "stable_cat0p50_rg800_min100_v35"


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
                "method": "radar-stable-segments-range-gated-interpolated",
                "candidate_count": 100,
                "selected_count": 100,
                "matched_count": 100,
                "coverage": 1.0,
                "track_switches": 0,
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
        name="stable_cat0p40_rg800_min100_v65",
        radar_catprob_threshold=0.4,
        radar_range_gate_m=800.0,
        stable_segment_min_frames=100,
        stable_segment_max_transition_speed_mps=65.0,
    )

    rows = ablation._rows_from_table(config, table_path)

    assert len(rows) == 1
    assert rows[0]["flight"] == "Opt1"
    assert rows[0]["method"] == "radar-stable-segments-range-gated-interpolated"
    assert rows[0]["error_3d_mean_m"] == 12.346
    assert rows[0]["stable_segment_min_frames"] == 100


def test_validate_args_rejects_empty_and_nonpositive_grids() -> None:
    for kwargs in (
        {"catprob_thresholds": []},
        {"range_gates_m": [0.0]},
        {"min_segment_frames": [0]},
        {"max_transition_speeds_mps": [0.0]},
    ):
        try:
            ablation._validate_args(_args(**kwargs))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"expected invalid grid to fail: {kwargs}")
