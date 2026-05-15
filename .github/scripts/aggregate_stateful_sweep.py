#!/usr/bin/env python3
"""Aggregate stateful learned-association sweep artifacts and rank variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANT_CSV_FIELDS = [
    "rank",
    "variant",
    "status",
    "flights",
    "ok_flights",
    "selected_radar_idf1",
    "selected_radar_mota",
    "selected_radar_fragmentation_per_match",
    "selected_radar_fp",
    "selected_radar_fn",
    "selected_radar_idsw",
    "estimate_idf1",
    "estimate_mota",
    "estimate_fragmentation_per_match",
    "mean_rmse_3d_m",
    "mean_p95_3d_m",
    "mean_selected_radar_rows",
    "mean_track_switch_count",
]

RUN_CSV_FIELDS = [
    "flight",
    "variant",
    "status",
    "selected_radar_idf1",
    "selected_radar_mota",
    "selected_radar_fragmentation_per_match",
    "selected_radar_fp",
    "selected_radar_fn",
    "selected_radar_idsw",
    "estimate_idf1",
    "estimate_mota",
    "estimate_fragmentation_per_match",
    "rmse_3d_m",
    "p95_3d_m",
    "selected_radar_rows",
    "track_switch_count",
    "beam_max_hypotheses",
    "beam_max_candidates",
    "beam_track_switch_cost",
    "beam_missed_detection_cost",
    "beam_consecutive_miss_cost",
    "beam_missing_track_id_cost",
    "beam_lag_s",
    "radar_catprob_threshold",
    "radar_inflation_alpha",
    "association_safety_gate_enabled",
]


@dataclass(frozen=True)
class SweepAggregateResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    variants: list[dict[str, Any]]

    @property
    def should_fail(self) -> bool:
        return bool(self.summary.get("missing_runs") or self.summary.get("failed_runs"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> int:
    number = finite_float(value)
    return 0 if number is None else int(number)


def slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def config_variant_name(config: dict[str, Any]) -> str:
    name = str(config.get("name") or "")
    if not name:
        name = (
            f"h{config.get('beam_max_hypotheses', 32)}_c{config.get('beam_max_candidates', 8)}"
            f"_sw{config.get('beam_track_switch_cost', 3.0)}"
            f"_miss{config.get('beam_missed_detection_cost', 4.0)}"
            f"_lag{config.get('beam_lag_s', 20)}"
        )
    return slug(name)


def expected_variant_names(raw_json: str | None) -> list[str]:
    if not raw_json:
        return []
    payload = json.loads(raw_json)
    if not isinstance(payload, list):
        raise SystemExit("--expected-variants-json must be a JSON array")
    names: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            raise SystemExit("--expected-variants-json entries must be JSON objects")
        names.append(config_variant_name(item))
    return names


def collect_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifacts_dir.glob("**/sweep_summary.json")):
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
    rows.sort(key=lambda row: (str(row.get("variant", "")), str(row.get("flight", ""))))
    return rows


def mot_payload(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def mot_count(row: dict[str, Any], key: str, field: str) -> int:
    return finite_int(mot_payload(row, key).get(field))


def aggregate_mot(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    gt = sum(mot_count(row, key, "gt") for row in rows)
    estimates = sum(mot_count(row, key, "estimates") for row in rows)
    tp = sum(mot_count(row, key, "tp") for row in rows)
    fp = sum(mot_count(row, key, "fp") for row in rows)
    fn = sum(mot_count(row, key, "fn") for row in rows)
    idsw = sum(mot_count(row, key, "idsw") for row in rows)
    fragmentations = sum(mot_count(row, key, "fragmentations") for row in rows)
    idtp = sum(mot_count(row, key, "idtp") for row in rows)
    idfp = sum(mot_count(row, key, "idfp") for row in rows)
    idfn = sum(mot_count(row, key, "idfn") for row in rows)
    idf1_den = (2 * idtp) + idfp + idfn
    return {
        "gt": gt,
        "estimates": estimates,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": idsw,
        "fragmentations": fragmentations,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "mota": None if gt == 0 else 1.0 - ((fp + fn + idsw) / gt),
        "idf1": None if idf1_den == 0 else (2 * idtp) / idf1_den,
        "fragmentation_per_match": None if tp == 0 else fragmentations / tp,
    }


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [finite_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def aggregate_variants(
    rows: list[dict[str, Any]], expected_flights: list[str], expected_variants: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant", ""))].append(row)
    for variant in expected_variants:
        grouped.setdefault(variant, [])

    out: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items()):
        ok_rows = [row for row in variant_rows if row.get("status") == "ok"]
        ok_flights = {str(row.get("flight")) for row in ok_rows}
        missing = [flight for flight in expected_flights if flight not in ok_flights]
        failed = [row for row in variant_rows if row.get("status") != "ok"]
        selected = aggregate_mot(ok_rows, "selected_radar_mot")
        estimate = aggregate_mot(ok_rows, "estimate_mot")
        status = "ok"
        if missing:
            status = "missing_flights"
        elif failed:
            status = "failed_runs"
        out.append(
            {
                "variant": variant,
                "status": status,
                "flights": len(expected_flights),
                "ok_flights": len(ok_flights),
                "missing_flights": missing,
                "failed_runs": [
                    f"{row.get('flight')}: {row.get('status')}" for row in failed
                ],
                "selected_radar_mot": selected,
                "estimate_mot": estimate,
                "selected_radar_idf1": selected.get("idf1"),
                "selected_radar_mota": selected.get("mota"),
                "selected_radar_fragmentation_per_match": selected.get("fragmentation_per_match"),
                "selected_radar_fp": selected.get("fp"),
                "selected_radar_fn": selected.get("fn"),
                "selected_radar_idsw": selected.get("idsw"),
                "estimate_idf1": estimate.get("idf1"),
                "estimate_mota": estimate.get("mota"),
                "estimate_fragmentation_per_match": estimate.get("fragmentation_per_match"),
                "mean_rmse_3d_m": mean_field(ok_rows, "rmse_3d_m"),
                "mean_p95_3d_m": mean_field(ok_rows, "p95_3d_m"),
                "mean_selected_radar_rows": mean_field(ok_rows, "selected_radar_rows"),
                "mean_track_switch_count": mean_field(ok_rows, "track_switch_count"),
            }
        )
    ranked = sorted(out, key=rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    idf1 = finite_float(row.get("selected_radar_idf1"))
    mota = finite_float(row.get("selected_radar_mota"))
    frag = finite_float(row.get("selected_radar_fragmentation_per_match"))
    rmse = finite_float(row.get("mean_rmse_3d_m"))
    return (
        -(idf1 if idf1 is not None else -1.0),
        -(mota if mota is not None else -1.0e9),
        frag if frag is not None else 1.0e9,
        rmse if rmse is not None else 1.0e9,
        str(row.get("variant", "")),
    )


def run_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = mot_payload(row, "selected_radar_mot")
    estimate = mot_payload(row, "estimate_mot")
    out = {field: row.get(field, "") for field in RUN_CSV_FIELDS}
    out.update(
        {
            "selected_radar_idf1": selected.get("idf1", ""),
            "selected_radar_mota": selected.get("mota", ""),
            "selected_radar_fragmentation_per_match": selected.get("fragmentation_per_match", ""),
            "selected_radar_fp": selected.get("fp", ""),
            "selected_radar_fn": selected.get("fn", ""),
            "selected_radar_idsw": selected.get("idsw", ""),
            "estimate_idf1": estimate.get("idf1", ""),
            "estimate_mota": estimate.get("mota", ""),
            "estimate_fragmentation_per_match": estimate.get("fragmentation_per_match", ""),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_sweep_artifacts(
    artifacts_dir: Path, expected_flights: list[str], expected_variants: list[str] | None = None
) -> SweepAggregateResult:
    rows = collect_rows(artifacts_dir)
    variants = aggregate_variants(rows, expected_flights, expected_variants or [])
    missing: list[str] = []
    failed: list[str] = []
    for variant in variants:
        missing.extend(f"{flight}/{variant['variant']}" for flight in variant.get("missing_flights", []))
        failed.extend(str(item) for item in variant.get("failed_runs", []))
    ok_variants = [variant for variant in variants if variant.get("status") == "ok"]
    summary = {
        "expected_flights": expected_flights,
        "expected_variants": expected_variants or [],
        "run_count": len(rows),
        "variant_count": len(variants),
        "missing_runs": missing,
        "failed_runs": failed,
        "best_variant": ok_variants[0] if ok_variants else None,
        "variants": variants,
        "runs": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in rows
        ],
    }
    return SweepAggregateResult(summary=summary, rows=rows, variants=variants)


def append_step_summary(path: Path, result: SweepAggregateResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("## Stateful learned radar association sweep\n\n")
        best = result.summary.get("best_variant")
        if best:
            handle.write(
                f"Best variant: **{best.get('variant')}** "
                f"(selected-radar IDF1={best.get('selected_radar_idf1')}, "
                f"MOTA={best.get('selected_radar_mota')})\n\n"
            )
        handle.write("| Rank | Variant | Status | IDF1 | MOTA | Frag/match | RMSE 3D |\n")
        handle.write("|---:|---|---|---:|---:|---:|---:|\n")
        for row in result.variants[:20]:
            handle.write(
                f"| {row.get('rank')} | {row.get('variant')} | {row.get('status')} | "
                f"{row.get('selected_radar_idf1')} | {row.get('selected_radar_mota')} | "
                f"{row.get('selected_radar_fragmentation_per_match')} | "
                f"{row.get('mean_rmse_3d_m')} |\n"
            )
        if result.summary.get("missing_runs"):
            handle.write("\n### Missing runs\n")
            for item in result.summary["missing_runs"]:
                handle.write(f"- {item}\n")
        if result.summary.get("failed_runs"):
            handle.write("\n### Failed runs\n")
            for item in result.summary["failed_runs"]:
                handle.write(f"- {item}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-json", type=Path, default=Path("stateful_sweep_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("stateful_sweep_summary.csv"))
    parser.add_argument("--output-runs-csv", type=Path, default=Path("stateful_sweep_runs.csv"))
    parser.add_argument("--expected-flights-json", required=True)
    parser.add_argument("--expected-variants-json", default="")
    parser.add_argument("--github-step-summary", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_flights = json.loads(args.expected_flights_json)
    if not isinstance(expected_flights, list) or not all(isinstance(item, str) for item in expected_flights):
        raise SystemExit("--expected-flights-json must be a JSON array of strings")
    expected_variants = expected_variant_names(args.expected_variants_json)
    result = aggregate_sweep_artifacts(args.artifacts_dir, expected_flights, expected_variants)
    args.output_json.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    write_csv(args.output_csv, result.variants, VARIANT_CSV_FIELDS)
    write_csv(args.output_runs_csv, [run_csv_row(row) for row in result.rows], RUN_CSV_FIELDS)
    step_summary = args.github_step_summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        append_step_summary(Path(step_summary), result)
    if result.should_fail:
        print(json.dumps({"missing_runs": result.summary["missing_runs"], "failed_runs": result.summary["failed_runs"]}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
