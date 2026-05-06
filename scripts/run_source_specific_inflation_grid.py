"""Run a source-specific NIS covariance-inflation alpha grid."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_specific_grid"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/source_specific_inflation_grid_opt1_opt3.csv"),
    )
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--rf-alphas", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--radar-alphas", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for rf_alpha in args.rf_alphas:
        for radar_alpha in args.radar_alphas:
            combo_name = _combo_name(rf_alpha, radar_alpha)
            combo_dir = args.output_dir / combo_name
            for flight in args.flights:
                metrics_path = combo_dir / flight / "metrics.json"
                if not (args.skip_existing and metrics_path.exists()):
                    _run_one(
                        dataset_root=args.dataset_root,
                        output_dir=combo_dir,
                        flight=flight,
                        rf_gate_prob=args.rf_gate_prob,
                        radar_gate_prob=args.radar_gate_prob,
                        rf_alpha=rf_alpha,
                        radar_alpha=radar_alpha,
                    )
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(_row(metrics_path, metrics, rf_alpha, radar_alpha))

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _run_one(
    *,
    dataset_root: Path,
    output_dir: Path,
    flight: str,
    rf_gate_prob: float,
    radar_gate_prob: float,
    rf_alpha: float,
    radar_alpha: float,
) -> None:
    command = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--robust-update",
        "nis-inflate",
        "--rf-gate-prob",
        str(rf_gate_prob),
        "--radar-gate-prob",
        str(radar_gate_prob),
        "--rf-inflation-alpha",
        str(rf_alpha),
        "--radar-inflation-alpha",
        str(radar_alpha),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _row(
    metrics_path: Path,
    metrics: dict[str, Any],
    rf_alpha: float,
    radar_alpha: float,
) -> dict[str, object]:
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}
    reweighted_by_source = metrics.get("reweighted_by_source") or {}
    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": "cv_nis_inflated",
        "rf_inflation_alpha": rf_alpha,
        "radar_inflation_alpha": radar_alpha,
        "posterior_records": int(metrics.get("posterior_records", 0)),
        "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
        "reweighted_rf": int(reweighted_by_source.get("rf", 0)),
        "reweighted_radar": int(reweighted_by_source.get("radar", 0)),
        "rmse_2d_m": _rounded(error_2d.get("rmse_m")),
        "mae_2d_m": _rounded(error_2d.get("mae_m")),
        "p50_2d_m": _rounded(error_2d.get("p50_m")),
        "p95_2d_m": _rounded(error_2d.get("p95_m")),
        "rmse_3d_m": _rounded(error_3d.get("rmse_m")),
        "mae_3d_m": _rounded(error_3d.get("mae_m")),
        "p50_3d_m": _rounded(error_3d.get("p50_m")),
        "p95_3d_m": _rounded(error_3d.get("p95_m")),
        "metrics_path": str(metrics_path),
    }


def _combo_name(rf_alpha: float, radar_alpha: float) -> str:
    return f"rf{_tag(rf_alpha)}_radar{_tag(radar_alpha)}"


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def _rounded(value: object) -> object:
    if value is None:
        return ""
    return round(float(value), 3)


if __name__ == "__main__":
    raise SystemExit(main())
