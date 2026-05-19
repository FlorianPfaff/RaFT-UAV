from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_leave_flight_out_sota as lfo  # noqa: E402


def test_truth_coverage_counts_truth_timestamps_with_nearby_estimate() -> None:
    truth_times = np.array([0.0, 1.0, 2.0, 3.0])
    estimate_times = np.array([0.1, 2.2, 10.0])

    coverage = lfo.truth_coverage(truth_times, estimate_times, max_time_delta_s=0.25)

    assert coverage["truth_rows"] == 4
    assert coverage["covered_truth_rows"] == 2
    assert coverage["truth_coverage_rate"] == 0.5


def test_summarize_scalar_errors_includes_tail_metrics() -> None:
    summary = lfo.summarize_scalar_errors(np.array([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4.0
    assert np.isclose(summary["rmse_m"], np.sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0))
    assert np.isclose(summary["p50_m"], 2.5)
    assert np.isclose(summary["p90_m"], 3.7)
    assert np.isclose(summary["p95_m"], 3.85)
    assert np.isclose(summary["p99_m"], 3.97)


def test_aggregate_method_rows_pools_errors_and_ranks_methods() -> None:
    methods = [
        lfo.MethodSpec("method_a", "baseline", "a"),
        lfo.MethodSpec("method_b", "baseline", "b"),
    ]
    evaluations = {
        "method_a": [
            lfo.RunEvaluation(
                row={"posterior_records": 2, "selected_radar_rows": 1},
                errors_2d_m=np.array([1.0, 2.0]),
                errors_3d_m=np.array([2.0, 3.0]),
                covered_truth_rows=8,
                truth_rows=10,
            )
        ],
        "method_b": [
            lfo.RunEvaluation(
                row={"posterior_records": 2, "selected_radar_rows": 1},
                errors_2d_m=np.array([0.5, 1.0]),
                errors_3d_m=np.array([1.0, 1.5]),
                covered_truth_rows=10,
                truth_rows=10,
            )
        ],
    }

    rows = lfo._aggregate_method_rows(methods, evaluations)
    by_method = {row["method"]: row for row in rows}

    assert by_method["method_a"]["truth_coverage_rate"] == 0.8
    assert by_method["method_b"]["truth_coverage_rate"] == 1.0
    assert by_method["method_b"]["rank_rmse_3d"] == 1
    assert by_method["method_a"]["rank_rmse_3d"] == 2


def test_heteroscedastic_sota_methods_expose_stronger_association() -> None:
    method = lfo.METHODS["hetero_cv_rf_gated_nis_fixed_lag"]

    assert method.runner == "hetero"
    assert method.association == "rf-gated-nis"
    assert method.fixed_lag is True
