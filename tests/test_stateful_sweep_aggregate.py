from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_summary(root: Path, flight: str, variant: str, *, idf1: float, mota: float) -> None:
    path = root / f"stateful-sweep-{flight}-{variant}" / "sweep_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "flight": flight,
        "variant": variant,
        "status": "ok",
        "rmse_3d_m": 10.0 if variant == "good" else 20.0,
        "p95_3d_m": 20.0 if variant == "good" else 40.0,
        "selected_radar_rows": 5,
        "track_switch_count": 1,
        "selected_radar_mot": {
            "gt": 10,
            "estimates": 10,
            "tp": 8,
            "fp": 1,
            "fn": 2,
            "idsw": 0 if variant == "good" else 2,
            "fragmentations": 1,
            "idtp": 8 if variant == "good" else 4,
            "idfp": 1 if variant == "good" else 5,
            "idfn": 2 if variant == "good" else 6,
            "idf1": idf1,
            "mota": mota,
            "fragmentation_per_match": 0.125,
        },
        "estimate_mot": {
            "gt": 10,
            "estimates": 10,
            "tp": 8,
            "fp": 1,
            "fn": 2,
            "idsw": 0,
            "fragmentations": 1,
            "idtp": 8,
            "idfp": 1,
            "idfn": 2,
            "idf1": idf1,
            "mota": mota,
            "fragmentation_per_match": 0.125,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_ranks_by_selected_radar_idf1(tmp_path: Path) -> None:
    script = load_script(Path(".github/scripts/aggregate_stateful_sweep.py"))
    artifacts = tmp_path / "artifacts"
    for flight in ("Opt1", "Opt2"):
        write_summary(artifacts, flight, "good", idf1=0.84, mota=0.70)
        write_summary(artifacts, flight, "bad", idf1=0.50, mota=0.50)

    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    output_runs_csv = tmp_path / "runs.csv"
    exit_code = script.main(
        [
            "--artifacts-dir",
            str(artifacts),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-runs-csv",
            str(output_runs_csv),
            "--expected-flights-json",
            '["Opt1","Opt2"]',
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["best_variant"]["variant"] == "good"
    assert payload["best_variant"]["rank"] == 1
    assert output_csv.exists()
    assert output_runs_csv.exists()


def test_aggregate_fails_on_missing_requested_flight(tmp_path: Path) -> None:
    script = load_script(Path(".github/scripts/aggregate_stateful_sweep.py"))
    artifacts = tmp_path / "artifacts"
    write_summary(artifacts, "Opt1", "only-one-flight", idf1=0.8, mota=0.6)

    exit_code = script.main(
        [
            "--artifacts-dir",
            str(artifacts),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-csv",
            str(tmp_path / "summary.csv"),
            "--output-runs-csv",
            str(tmp_path / "runs.csv"),
            "--expected-flights-json",
            '["Opt1","Opt2"]',
        ]
    )

    assert exit_code == 1
