import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_sota_readiness_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("run_sota_readiness_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_error_columns_preserve_counts_and_paper_metrics():
    report = _load_report_module()

    columns = report.paper_error_columns(
        {
            "count": 12.0,
            "eval_sample_count": 20,
            "coverage": 0.6,
            "mean_3d_error_m": 11.2345,
            "std_3d_error_m": 2.3456,
            "rmse_3d_error_m": 12.3456,
            "p95_3d_error_m": 17.6543,
            "max_3d_error_m": 23.4567,
        }
    )

    assert columns["matched_count"] == 12
    assert columns["eval_sample_count"] == 20
    assert columns["coverage"] == 0.6
    assert columns["mean_3d_error_m"] == 11.234
    assert columns["std_3d_error_m"] == 2.346
    assert columns["max_3d_error_m"] == 23.457
    assert columns["rmse_3d_error_m"] == 12.346


def test_leaderboard_ranks_by_mean_3d_error_with_rmse_secondary_only():
    report = _load_report_module()
    rows = [
        {
            "flight": "Opt1",
            "method": "low-rmse-high-mean",
            "row_type": "tracking",
            "mean_3d_error_m": 15.0,
            "rmse_3d_error_m": 15.1,
        },
        {
            "flight": "Opt1",
            "method": "high-rmse-low-mean",
            "row_type": "tracking",
            "mean_3d_error_m": 10.0,
            "rmse_3d_error_m": 30.0,
        },
        {
            "flight": "Opt1",
            "method": "oracle",
            "row_type": "oracle",
            "mean_3d_error_m": 1.0,
            "rmse_3d_error_m": 1.1,
        },
        {
            "flight": "Opt2",
            "method": "missing-metric",
            "row_type": "tracking",
            "mean_3d_error_m": "",
            "rmse_3d_error_m": 0.1,
        },
    ]

    leaderboard = report.build_leaderboard_rows(rows)
    opt1_tracking = [
        row for row in leaderboard if row["flight"] == "Opt1" and row["row_type"] == "tracking"
    ]
    opt1_oracle = [
        row for row in leaderboard if row["flight"] == "Opt1" and row["row_type"] == "oracle"
    ]
    opt2_tracking = [
        row for row in leaderboard if row["flight"] == "Opt2" and row["row_type"] == "tracking"
    ]

    assert [row["method"] for row in opt1_tracking] == [
        "high-rmse-low-mean",
        "low-rmse-high-mean",
    ]
    assert [row["rank"] for row in opt1_tracking] == [1, 2]
    assert opt1_oracle[0]["rank"] == 1
    assert opt2_tracking[0]["rank"] == ""
    assert all(row["paper_primary_metric"] == "mean_3d_error_m" for row in leaderboard)
