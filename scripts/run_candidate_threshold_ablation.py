"""Run radar candidate class-probability threshold ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ablation_common as common


@dataclass(frozen=True)
class _Config:
    name: str
    threshold: float
    fallback_top_k: int


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/catprob_threshold_ablation"),
        default_summary_output=Path("outputs/catprob_threshold_ablation.csv"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.4, 0.5])
    parser.add_argument(
        "--fallback-top-k",
        nargs="*",
        type=int,
        default=[0],
        help="top-K class-probability fallback sizes to sweep; 0 disables fallback",
    )
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if any(value < 0 for value in args.fallback_top_k):
        parser.error("--fallback-top-k values must be nonnegative")

    configs = [
        _Config(_threshold_name(threshold, fallback_top_k), threshold, fallback_top_k)
        for threshold in args.thresholds
        for fallback_top_k in args.fallback_top_k
    ]
    rows = common.run_named_config_experiments(args, configs, _run_one, _candidate_row)
    return _finish(args.summary_output, rows)


def _finish(summary_output: Path, rows: list[dict[str, object]]) -> int:
    common.write_summary_csv(summary_output, rows)
    print(f"wrote {len(rows)} rows to {summary_output}")
    return 0


def _candidate_row(
    config: _Config, metrics_path: Path, metrics: dict[str, object]
) -> dict[str, object]:
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
            "radar_catprob_fallback_top_k": metrics.get("radar_catprob_fallback_top_k", ""),
            "radar_catprob_fallback_rows": metrics.get("radar_catprob_fallback_rows", ""),
        },
        include_selected_track_ids=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    options: list[object] = ["--radar-catprob-threshold", config.threshold]
    if config.fallback_top_k > 0:
        options.extend(["--radar-catprob-fallback-top-k", config.fallback_top_k])
    options.extend(common.robust_update_options(args))
    options.extend(common.smoother_options("fixed-lag", args.fixed_lag_s))
    common.run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association="prediction-nis",
        extra_options=options,
    )


def _threshold_name(threshold: float, fallback_top_k: int) -> str:
    name = f"prediction_nis_t{common.slug(threshold, precision=2)}"
    if fallback_top_k > 0:
        return f"{name}_fbtop{int(fallback_top_k)}"
    return name


if __name__ == "__main__":
    raise SystemExit(main())
