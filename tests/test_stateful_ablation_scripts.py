from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / ".github" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aggregate_stateful_ablation = _load_script("aggregate_stateful_ablation")
summarize_stateful_ablation_run = _load_script("summarize_stateful_ablation_run")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_stateful_ablation_summary_counts_update_actions(tmp_path: Path) -> None:
    metrics_path = tmp_path / "run" / "Opt1" / "metrics.json"
    diagnostic_summary_path = tmp_path / "run" / "Opt1" / "diagnostic_summary.json"
    diagnostics_path = tmp_path / "run" / "Opt1" / "diagnostics.csv"
    output_path = tmp_path / "ablation_summary.json"
    _write_json(
        metrics_path,
        {
            "flight": "Opt1",
            "radar_association": "learned-likelihood",
            "learned_radar_association_mode": "stateful-beam",
            "selected_radar_rows": 2,
            "posterior_records": 4,
            "accepted_measurements": 3,
            "rejected_measurements": 1,
            "reweighted_measurements": 1,
            "position_error_2d": {"rmse_m": 10.0, "p95_m": 20.0},
            "position_error_3d": {"rmse_m": 30.0, "p95_m": 40.0},
        },
    )
    _write_json(
        diagnostic_summary_path,
        {
            "track_switches": {"selected_radar": {"count": 1}},
            "covariance_inflation": {"count": 2},
        },
    )
    _write_text(
        diagnostics_path,
        "time_s,update_action\n0,updated\n1,inflated\n2,missed_detection\n3,rejected\n",
    )

    summary = summarize_stateful_ablation_run.build_summary(
        argparse.Namespace(
            flight="Opt1",
            variant="gate-on_switch-3_lag-20_radar-alpha-0p5",
            gating="on",
            beam_track_switch_cost=3.0,
            beam_lag_s=20.0,
            radar_inflation_alpha=0.5,
            rf_inflation_alpha=0.5,
            metrics_path=metrics_path,
            diagnostic_summary_path=diagnostic_summary_path,
            diagnostics_path=diagnostics_path,
            output=output_path,
        )
    )

    assert summary["status"] == "ok"
    assert summary["missed_detection_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["inflated_count"] == 1
    assert summary["track_switch_count"] == 1
    assert summary["covariance_inflation_count"] == 2


def test_stateful_ablation_aggregate_detects_expected_variant(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    variant = aggregate_stateful_ablation.variant_name(
        gating="off",
        track_switch_cost=3.0,
        beam_lag_s=20.0,
        radar_inflation_alpha=0.5,
    )
    _write_json(
        artifacts / "stateful-ablation-Opt1-off" / "ablation_summary.json",
        {
            "flight": "Opt1",
            "variant": variant,
            "status": "ok",
            "gating": "off",
            "beam_track_switch_cost": 3.0,
            "beam_lag_s": 20.0,
            "radar_inflation_alpha": 0.5,
            "rf_inflation_alpha": 0.5,
            "selected_radar_rows": 2,
            "posterior_records": 3,
        },
    )
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"

    rc = aggregate_stateful_ablation.main(
        [
            "--artifacts-dir",
            str(artifacts),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--flights-json",
            '["Opt1"]',
            "--gating-modes-json",
            '["off"]',
            "--track-switch-costs-json",
            "[3.0]",
            "--beam-lags-json",
            "[20]",
            "--radar-inflation-alphas-json",
            "[0.5]",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["expected_runs"] == 1
    assert payload["ok_runs"] == 1
    assert payload["missing_runs"] == []
    assert "gate-off_switch-3_lag-20_radar-alpha-0p5" in output_csv.read_text(
        encoding="utf-8"
    )
