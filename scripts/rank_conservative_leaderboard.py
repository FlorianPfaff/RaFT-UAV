#!/usr/bin/env python3
"""Apply conservative constraints and accuracy/runtime Pareto marking to a leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import accuracy_runtime_pareto, conservative_leaderboard_rank


def _parse_constraint(text: str) -> tuple[str, tuple[str, float]]:
    for op in ("ge", "gt", "le", "lt", "eq"):
        token = f":{op}:"
        if token in text:
            column, value = text.split(token, 1)
            return column, (op, float(value))
    raise argparse.ArgumentTypeError("constraint must look like column:ge:0.95")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leaderboard_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--objective", default="p95_3d_error_m")
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint in the form column:ge:0.95, column:le:10, etc.",
    )
    parser.add_argument("--runtime-column", default="wall_time_s")
    args = parser.parse_args()

    constraints = dict(_parse_constraint(text) for text in args.constraint)
    rows = pd.read_csv(args.leaderboard_csv)
    ranked = conservative_leaderboard_rank(rows, objective=args.objective, constraints=constraints)
    if args.runtime_column in ranked.columns:
        ranked = accuracy_runtime_pareto(ranked, error_column=args.objective, runtime_column=args.runtime_column)
    output = args.output_csv or args.leaderboard_csv.with_name(f"{args.leaderboard_csv.stem}_conservative.csv")
    ranked.to_csv(output, index=False)
    print(f"conservative_leaderboard_csv={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
