"""Collect ungated/gated baseline metrics into a paper-evidence CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--gated-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for method, root in (("cv", args.baseline_dir), ("cv_nis_gated", args.gated_dir)):
        for flight in args.flights:
            metrics_path = root / flight / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(_row(method, metrics_path, metrics))

    if not rows:
        raise RuntimeError("No metrics.json files found for the requested flights")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def _row(method: str, metrics_path: Path, metrics: dict[str, Any]) -> dict[str, object]:
    rejected_by_source = metrics.get("rejected_by_source") or {}
    accepted_by_source = metrics.get("accepted_by_source") or {}
    source_counts = metrics.get("source_counts") or {}
    gating = metrics.get("gating") or {}
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}

    posterior_records = int(metrics.get("posterior_records", 0))
    accepted = int(metrics.get("accepted_measurements", posterior_records))
    rejected = int(metrics.get("rejected_measurements", 0))

    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "gating_enabled": bool(gating.get("enabled", False)),
        "rf_gate_probability": _empty_if_none(gating.get("rf_gate_probability")),
        "radar_gate_probability": _empty_if_none(gating.get("radar_gate_probability")),
        "posterior_records": posterior_records,
        "accepted_measurements": accepted,
        "rejected_measurements": rejected,
        "accepted_rf": int(
            accepted_by_source.get("rf", source_counts.get("rf", 0) if rejected == 0 else 0)
        ),
        "accepted_radar": int(
            accepted_by_source.get("radar", source_counts.get("radar", 0) if rejected == 0 else 0)
        ),
        "rejected_rf": int(rejected_by_source.get("rf", 0)),
        "rejected_radar": int(rejected_by_source.get("radar", 0)),
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


def _rounded(value: object) -> object:
    if value is None:
        return ""
    return round(float(value), 3)


def _empty_if_none(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
