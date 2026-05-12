#!/usr/bin/env python3
"""Aggregate stateful learned association ablation artifacts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "flight",
    "variant",
    "status",
    "gating",
    "beam_track_switch_cost",
    "beam_lag_s",
    "radar_inflation_alpha",
    "rf_inflation_alpha",
    "selected_radar_rows",
    "posterior_records",
    "accepted_measurements",
    "rejected_measurements",
    "reweighted_measurements",
    "missed_detection_count",
    "rejected_count",
    "inflated_count",
    "track_switch_count",
    "covariance_inflation_count",
    "rmse_2d_m",
    "p95_2d_m",
    "rmse_3d_m",
    "p95_3d_m",
    "metrics_path",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifacts_dir.glob("**/ablation_summary.json")):
        try:
            payload = load_json(summary_path)
            if not isinstance(payload, dict):
                raise TypeError("summary payload is not a JSON object")
            payload["_summary_path"] = str(summary_path)
            rows.append(payload)
        except Exception as exc:
            rows.append(
                {
                    "flight": str(summary_path),
                    "variant": "unreadable",
                    "status": f"failed_to_read_summary: {exc}",
                    "_summary_path": str(summary_path),
                }
            )
    rows.sort(key=lambda row: (str(row.get("flight", "")), str(row.get("variant", ""))))
    return rows


def expected_variants(
    *,
    flights: list[str],
    gating_modes: list[str],
    track_switch_costs: list[float],
    beam_lags_s: list[float],
    radar_inflation_alphas: list[float],
) -> list[tuple[str, str]]:
    expected: list[tuple[str, str]] = []
    for flight, gating, switch_cost, lag_s, alpha in itertools.product(
        flights,
        gating_modes,
        track_switch_costs,
        beam_lags_s,
        radar_inflation_alphas,
    ):
        expected.append(
            (
                flight,
                variant_name(
                    gating=gating,
                    track_switch_cost=switch_cost,
                    beam_lag_s=lag_s,
                    radar_inflation_alpha=alpha,
                ),
            )
        )
    return expected


def variant_name(
    *,
    gating: str,
    track_switch_cost: float,
    beam_lag_s: float,
    radar_inflation_alpha: float,
) -> str:
    return (
        f"gate-{gating}_switch-{slug(track_switch_cost)}"
        f"_lag-{slug(beam_lag_s)}_radar-alpha-{slug(radar_inflation_alpha)}"
    )


def slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CSV_FIELDS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-json", type=Path, default=Path("stateful_ablation_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("stateful_ablation_summary.csv"))
    parser.add_argument("--flights-json", required=True)
    parser.add_argument("--gating-modes-json", required=True)
    parser.add_argument("--track-switch-costs-json", required=True)
    parser.add_argument("--beam-lags-json", required=True)
    parser.add_argument("--radar-inflation-alphas-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    flights = [str(value) for value in json.loads(args.flights_json)]
    gating_modes = [str(value) for value in json.loads(args.gating_modes_json)]
    track_switch_costs = [float(value) for value in json.loads(args.track_switch_costs_json)]
    beam_lags_s = [float(value) for value in json.loads(args.beam_lags_json)]
    radar_inflation_alphas = [float(value) for value in json.loads(args.radar_inflation_alphas_json)]

    rows = collect_rows(args.artifacts_dir)
    by_key = {(str(row.get("flight")), str(row.get("variant"))): row for row in rows}
    expected = expected_variants(
        flights=flights,
        gating_modes=gating_modes,
        track_switch_costs=track_switch_costs,
        beam_lags_s=beam_lags_s,
        radar_inflation_alphas=radar_inflation_alphas,
    )
    missing = [f"{flight}/{variant}" for flight, variant in expected if (flight, variant) not in by_key]
    failed = [
        f"{row.get('flight')}/{row.get('variant')}: {row.get('status')}"
        for row in rows
        if row.get("status") != "ok"
    ]

    public_rows = [public_row(row) for row in rows]
    summary = {
        "expected_runs": len(expected),
        "completed_runs": len(rows),
        "ok_runs": sum(1 for row in rows if row.get("status") == "ok"),
        "missing_runs": missing,
        "failed_runs": failed,
        "rows": public_rows,
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, public_rows)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    if missing:
        print("Missing ablation summaries:")
        for item in missing:
            print(f"- {item}")
    if failed:
        print("Failed ablation summaries:")
        for item in failed:
            print(f"- {item}")
    return 1 if missing or failed else 0


if __name__ == "__main__":
    sys.exit(main())
