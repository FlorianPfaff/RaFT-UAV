"""Run fast stable-radar-segment diagnostic ablations."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ablation_common as common  # noqa: E402
from raft_uav.diagnostics.paper_table import run_paper_table_diagnostic  # noqa: E402

RADAR_SELECTIONS = (
    "radar-longest-track-range-gated",
    "radar-stable-segments-range-gated",
    "radar-stable-segments-range-gated-interpolated",
)
SUMMARY_COLUMNS = (
    "flight",
    "method",
    "config",
    "radar_catprob_threshold",
    "radar_range_gate_m",
    "stable_segment_min_frames",
    "stable_segment_max_transition_speed_mps",
    "flight_count",
    "candidate_count",
    "selected_count",
    "matched_count",
    "coverage",
    "track_switches",
    "error_3d_mean_m",
    "error_3d_rmse_m",
    "error_3d_p95_m",
    "error_3d_max_m",
    "error_2d_mean_m",
    "error_2d_rmse_m",
    "error_2d_p95_m",
    "table_path",
)
RANKING_COLUMNS = (
    "rank",
    "eligible_for_recommendation",
    "ranking_min_coverage",
    "method",
    "config",
    "flight_count",
    "radar_catprob_threshold",
    "radar_range_gate_m",
    "stable_segment_min_frames",
    "stable_segment_max_transition_speed_mps",
    "coverage",
    "error_3d_mean_m",
    "error_3d_rmse_m",
    "error_3d_p95_m",
    "error_3d_max_m",
    "track_switches",
    "selected_count",
    "matched_count",
    "candidate_count",
)
SUM_COLUMNS = ("candidate_count", "selected_count", "matched_count", "track_switches")
MEAN_COLUMNS = (
    "error_3d_mean_m",
    "error_3d_rmse_m",
    "error_3d_p95_m",
    "error_3d_max_m",
    "error_2d_mean_m",
    "error_2d_rmse_m",
    "error_2d_p95_m",
)


@dataclass(frozen=True)
class StableSegmentConfig:
    """One stable-radar-segment diagnostic configuration."""

    name: str
    radar_catprob_threshold: float
    radar_range_gate_m: float
    stable_segment_min_frames: int
    stable_segment_max_transition_speed_mps: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/stable_radar_segment_ablation"),
        default_summary_output=Path("outputs/stable_radar_segment_ablation.csv"),
    )
    parser.add_argument("--catprob-thresholds", nargs="*", type=float, default=[0.4])
    parser.add_argument("--range-gates-m", nargs="*", type=float, default=[800.0])
    parser.add_argument("--min-segment-frames", nargs="*", type=int, default=[75, 100, 150])
    parser.add_argument("--max-transition-speeds-mps", nargs="*", type=float, default=[35.0, 65.0, 100.0])
    parser.add_argument("--ranking-output", type=Path, default=None)
    parser.add_argument(
        "--ranking-min-coverage",
        type=float,
        default=0.95,
        help="minimum aggregate coverage for a row to be ranked as recommendation-eligible",
    )
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    rows: list[dict[str, object]] = []
    for config in _configs(args):
        run_dir = args.output_dir / config.name
        for flight in args.flights:
            table_path = run_dir / flight / "paper_table.csv"
            if not (args.skip_existing and table_path.exists()):
                run_paper_table_diagnostic(
                    dataset_root=args.dataset_root,
                    flight_name=flight,
                    output_dir=run_dir,
                    radar_catprob_threshold=config.radar_catprob_threshold,
                    radar_range_gate_m=config.radar_range_gate_m,
                    stable_segment_min_frames=config.stable_segment_min_frames,
                    stable_segment_max_transition_speed_mps=config.stable_segment_max_transition_speed_mps,
                    radar_selections=RADAR_SELECTIONS,
                    truth_time_gate_s=args.truth_time_gate_s,
                    include_fusion=False,
                )
            rows.extend(_rows_from_table(config, table_path))
    aggregate_rows = _aggregate_rows(rows)
    ranking_rows = _ranking_rows(
        aggregate_rows,
        min_coverage=args.ranking_min_coverage,
    )
    ranking_output = args.ranking_output or args.summary_output.with_name(
        f"{args.summary_output.stem}_ranking.csv"
    )
    _write_summary(args.summary_output, [*rows, *aggregate_rows])
    _write_ranking(ranking_output, ranking_rows)
    print(f"wrote {len(rows) + len(aggregate_rows)} rows to {args.summary_output}")
    print(f"wrote {len(ranking_rows)} ranking rows to {ranking_output}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "catprob_thresholds",
        "range_gates_m",
        "min_segment_frames",
        "max_transition_speeds_mps",
    ):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} must not be empty")
    if any(value < 1 for value in args.min_segment_frames):
        raise SystemExit("--min-segment-frames values must be positive")
    if any(value <= 0.0 for value in args.range_gates_m):
        raise SystemExit("--range-gates-m values must be positive")
    if any(value <= 0.0 for value in args.max_transition_speeds_mps):
        raise SystemExit("--max-transition-speeds-mps values must be positive")
    if not 0.0 <= float(args.ranking_min_coverage) <= 1.0:
        raise SystemExit("--ranking-min-coverage must be in [0, 1]")


def _configs(args: argparse.Namespace) -> list[StableSegmentConfig]:
    configs: list[StableSegmentConfig] = []
    for catprob, range_gate, min_frames, max_speed in itertools.product(
        args.catprob_thresholds,
        args.range_gates_m,
        args.min_segment_frames,
        args.max_transition_speeds_mps,
    ):
        configs.append(
            StableSegmentConfig(
                name=_config_name(catprob, range_gate, min_frames, max_speed),
                radar_catprob_threshold=float(catprob),
                radar_range_gate_m=float(range_gate),
                stable_segment_min_frames=int(min_frames),
                stable_segment_max_transition_speed_mps=float(max_speed),
            )
        )
    return configs


def _config_name(
    catprob: float,
    range_gate_m: float,
    min_frames: int,
    max_transition_speed_mps: float,
) -> str:
    return (
        f"stable_cat{common.slug(catprob, precision=2)}"
        f"_rg{common.slug(range_gate_m, precision=0)}"
        f"_min{int(min_frames)}"
        f"_v{common.slug(max_transition_speed_mps, precision=0)}"
    )


def _rows_from_table(
    config: StableSegmentConfig,
    table_path: Path,
) -> list[dict[str, object]]:
    table = pd.read_csv(table_path)
    rows: list[dict[str, object]] = []
    for _, item in table.iterrows():
        method = str(item.get("method", ""))
        if method not in RADAR_SELECTIONS:
            continue
        row: dict[str, object] = {
            "flight": table_path.parent.name,
            "method": method,
            "config": config.name,
            "radar_catprob_threshold": config.radar_catprob_threshold,
            "radar_range_gate_m": config.radar_range_gate_m,
            "stable_segment_min_frames": config.stable_segment_min_frames,
            "stable_segment_max_transition_speed_mps": config.stable_segment_max_transition_speed_mps,
            "flight_count": 1,
            "table_path": str(table_path),
        }
        for column in SUMMARY_COLUMNS:
            if column not in row and column in item.index:
                row[column] = _csv_value(item[column])
        rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return rows


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return one aggregate row per config/method across flights."""

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    group_columns = [
        "method",
        "config",
        "radar_catprob_threshold",
        "radar_range_gate_m",
        "stable_segment_min_frames",
        "stable_segment_max_transition_speed_mps",
    ]
    aggregate_rows: list[dict[str, object]] = []
    for group_key, group in frame.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, group_key))
        row["flight"] = "aggregate"
        row["flight_count"] = int(group["flight"].nunique())
        for column in SUM_COLUMNS:
            row[column] = int(pd.to_numeric(group[column], errors="coerce").fillna(0).sum())
        row["coverage"] = _safe_ratio(row["matched_count"], row["candidate_count"])
        for column in MEAN_COLUMNS:
            row[column] = _mean_column(group, column)
        row["table_path"] = ""
        aggregate_rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return aggregate_rows


def _ranking_rows(
    aggregate_rows: list[dict[str, object]],
    *,
    min_coverage: float = 0.95,
) -> list[dict[str, object]]:
    """Return aggregate rows sorted by paper-facing stable-segment quality."""

    ranking = sorted(
        aggregate_rows,
        key=lambda row: (
            not _coverage_eligible(row, min_coverage=min_coverage),
            _sort_value(row.get("error_3d_mean_m")),
            _sort_value(row.get("error_3d_p95_m")),
            -float(row.get("coverage") or 0.0),
        ),
    )
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(ranking, start=1):
        ranked = {
            "rank": rank,
            "eligible_for_recommendation": _coverage_eligible(
                row,
                min_coverage=min_coverage,
            ),
            "ranking_min_coverage": float(min_coverage),
            **row,
        }
        rows.append({column: ranked.get(column, "") for column in RANKING_COLUMNS})
    return rows


def _mean_column(frame: pd.DataFrame, column: str) -> object:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return round(float(values.mean()), 3)


def _safe_ratio(numerator: object, denominator: object) -> object:
    denominator_value = float(denominator or 0.0)
    if denominator_value <= 0.0:
        return ""
    return round(float(numerator or 0.0) / denominator_value, 3)


def _sort_value(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("inf")


def _coverage_eligible(row: dict[str, object], *, min_coverage: float) -> bool:
    coverage = pd.to_numeric(pd.Series([row.get("coverage")]), errors="coerce").iloc[0]
    return bool(pd.notna(coverage) and float(coverage) >= float(min_coverage))


def _csv_value(value: Any) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return round(float(value), 3)
    return value


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No stable radar segment ablation rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).to_csv(path, index=False)


def _write_ranking(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No stable radar segment ranking rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(RANKING_COLUMNS)).to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
