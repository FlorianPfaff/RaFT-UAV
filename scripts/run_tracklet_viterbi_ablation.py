"""Run controlled sequence-level tracklet-Viterbi radar association ablations."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402
from raft_uav.tracklet_viterbi_cli import (  # noqa: E402
    _BELOW_CATPROB_PENALTY_ENV,
    _CATPROB_MODE_ENV,
    _MAX_CANDIDATE_POOL_ENV,
    _MAX_CANDIDATES_PER_FRAME_ENV,
    _MAX_CANDIDATES_PER_TRACK_ENV,
    _MAX_TRACK_SUPPORT_REWARD_ENV,
    _TRACK_SUPPORT_WEIGHT_ENV,
    _TRACKLET_VARIANT_ENV,
    main as tracklet_viterbi_main,
)

_VARIANTS = ("base", "retention-hard", "soft-catprob", "support", "range-covariance")
_SUPPORT_VARIANTS = {"support", "range-covariance"}
_SOFT_CATPROB_VARIANTS = {"soft-catprob", "support", "range-covariance"}
_RETENTION_RUNNER_VARIANTS = {"retention-hard", "soft-catprob", "support"}


@dataclass(frozen=True)
class _Config:
    name: str
    threshold: float
    variant: str
    runner_variant: str
    catprob_retention_mode: str
    max_candidates_per_frame: int
    max_candidate_pool_per_frame: int
    max_candidates_per_track_id: int
    below_catprob_threshold_penalty: float
    track_support_weight: float
    max_track_support_reward: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/tracklet_viterbi_ablation"),
        default_summary_output=Path("outputs/tracklet_viterbi_ablation.csv"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--variants", nargs="*", choices=_VARIANTS, default=list(_VARIANTS))
    parser.add_argument("--tracklet-max-candidates-per-frame", nargs="*", type=int, default=[8])
    parser.add_argument(
        "--tracklet-max-candidate-pool-per-frame",
        nargs="*",
        type=int,
        default=[16],
    )
    parser.add_argument("--tracklet-max-candidates-per-track-id", nargs="*", type=int, default=[1])
    parser.add_argument(
        "--tracklet-below-catprob-threshold-penalties",
        nargs="*",
        type=float,
        default=[3.0],
    )
    parser.add_argument(
        "--tracklet-track-support-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.45, 0.75],
    )
    parser.add_argument(
        "--tracklet-max-track-support-rewards",
        nargs="*",
        type=float,
        default=[4.0],
    )
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    configs = _configs(args)
    rows = common.run_named_config_experiments(args, configs, _run_one, _candidate_row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _configs(args: argparse.Namespace) -> list[_Config]:
    configs: list[_Config] = []
    variants = list(getattr(args, "variants", ["support"]))
    legacy_names = not hasattr(args, "variants")
    for values in itertools.product(
        args.thresholds,
        variants,
        args.tracklet_max_candidates_per_frame,
        args.tracklet_max_candidate_pool_per_frame,
        args.tracklet_max_candidates_per_track_id,
        args.tracklet_below_catprob_threshold_penalties,
        args.tracklet_track_support_weights,
        args.tracklet_max_track_support_rewards,
    ):
        variant = str(values[1])
        support_weight = float(values[6]) if variant in _SUPPORT_VARIANTS else 0.0
        support_reward = float(values[7]) if variant in _SUPPORT_VARIANTS else 0.0
        if variant not in _SUPPORT_VARIANTS and float(values[6]) != 0.0:
            continue
        config = _Config(
            name="",
            threshold=float(values[0]),
            variant=variant,
            runner_variant=_runner_variant(variant),
            catprob_retention_mode=_catprob_retention_mode(variant),
            max_candidates_per_frame=int(values[2]),
            max_candidate_pool_per_frame=int(values[3]),
            max_candidates_per_track_id=int(values[4]),
            below_catprob_threshold_penalty=float(values[5]),
            track_support_weight=support_weight,
            max_track_support_reward=support_reward,
        )
        name = _legacy_config_name(config) if legacy_names else _config_name(config)
        configs.append(_Config(name=name, **_config_values(config)))
    return configs


def _runner_variant(variant: str) -> str:
    if variant == "base":
        return "base"
    if variant in _RETENTION_RUNNER_VARIANTS:
        return "retention"
    if variant == "range-covariance":
        return "range-covariance"
    raise ValueError(f"unknown variant {variant!r}")


def _catprob_retention_mode(variant: str) -> str:
    return "soft" if variant in _SOFT_CATPROB_VARIANTS else "hard"


def _config_values(config: _Config) -> dict[str, object]:
    return {
        "threshold": config.threshold,
        "variant": config.variant,
        "runner_variant": config.runner_variant,
        "catprob_retention_mode": config.catprob_retention_mode,
        "max_candidates_per_frame": config.max_candidates_per_frame,
        "max_candidate_pool_per_frame": config.max_candidate_pool_per_frame,
        "max_candidates_per_track_id": config.max_candidates_per_track_id,
        "below_catprob_threshold_penalty": config.below_catprob_threshold_penalty,
        "track_support_weight": config.track_support_weight,
        "max_track_support_reward": config.max_track_support_reward,
    }


def _candidate_row(
    config: _Config,
    metrics_path: Path,
    metrics: dict[str, object],
) -> dict[str, object]:
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "variant": config.variant,
            "runner_variant": config.runner_variant,
            "tracklet_catprob_retention_mode": config.catprob_retention_mode,
            "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
            "tracklet_max_candidates_per_frame": config.max_candidates_per_frame,
            "tracklet_max_candidate_pool_per_frame": config.max_candidate_pool_per_frame,
            "tracklet_max_candidates_per_track_id": config.max_candidates_per_track_id,
            "tracklet_below_catprob_threshold_penalty": config.below_catprob_threshold_penalty,
            "tracklet_track_support_weight": config.track_support_weight,
            "tracklet_max_track_support_reward": config.max_track_support_reward,
        },
        include_selected_track_ids=True,
        include_reweighted=True,
        include_inflation=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    cli_args = [
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        "tracklet-viterbi",
        "--radar-catprob-threshold",
        str(config.threshold),
        *[str(option) for option in common.robust_update_options(args)],
        *[str(option) for option in common.smoother_options("fixed-lag", args.fixed_lag_s)],
    ]
    env_overrides = _tracklet_environment(config)
    print(
        " ".join(
            [
                *[f"{key}={value}" for key, value in env_overrides.items()],
                "raft-uav-tracklet-viterbi",
                *cli_args,
            ]
        ),
        flush=True,
    )
    previous_env = _set_environment(env_overrides)
    try:
        status = tracklet_viterbi_main(cli_args)
    finally:
        _restore_environment(previous_env)
    if status != 0:
        raise RuntimeError(f"tracklet-viterbi run failed with status {status}")


def _tracklet_environment(config: _Config) -> dict[str, str]:
    return {
        _TRACKLET_VARIANT_ENV: config.runner_variant,
        _CATPROB_MODE_ENV: config.catprob_retention_mode,
        _MAX_CANDIDATES_PER_FRAME_ENV: str(config.max_candidates_per_frame),
        _MAX_CANDIDATE_POOL_ENV: str(config.max_candidate_pool_per_frame),
        _MAX_CANDIDATES_PER_TRACK_ENV: str(config.max_candidates_per_track_id),
        _BELOW_CATPROB_PENALTY_ENV: str(config.below_catprob_threshold_penalty),
        _TRACK_SUPPORT_WEIGHT_ENV: str(config.track_support_weight),
        _MAX_TRACK_SUPPORT_REWARD_ENV: str(config.max_track_support_reward),
    }


def _set_environment(overrides: dict[str, str]) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _config_name(config: _Config) -> str:
    return "_".join(
        [
            f"tracklet_{config.variant.replace('-', '_')}",
            f"t{common.slug(config.threshold, precision=2)}",
            f"k{config.max_candidates_per_frame}",
            f"pool{config.max_candidate_pool_per_frame}",
            f"perid{config.max_candidates_per_track_id}",
            f"cat{config.catprob_retention_mode}",
            f"catpen{common.slug(config.below_catprob_threshold_penalty, precision=1)}",
            f"sw{common.slug(config.track_support_weight, precision=2)}",
            f"sr{common.slug(config.max_track_support_reward, precision=1)}",
        ]
    )


def _legacy_config_name(config: _Config) -> str:
    return "_".join(
        [
            f"tracklet_t{common.slug(config.threshold, precision=2)}",
            f"k{config.max_candidates_per_frame}",
            f"pool{config.max_candidate_pool_per_frame}",
            f"perid{config.max_candidates_per_track_id}",
            f"catpen{common.slug(config.below_catprob_threshold_penalty, precision=1)}",
            f"sw{common.slug(config.track_support_weight, precision=2)}",
            f"sr{common.slug(config.max_track_support_reward, precision=1)}",
        ]
    )


def _validate_args(args: argparse.Namespace) -> None:
    _require_nonempty("--thresholds", args.thresholds)
    if hasattr(args, "variants"):
        _require_nonempty("--variants", args.variants)
    _require_positive_ints(
        "--tracklet-max-candidates-per-frame",
        args.tracklet_max_candidates_per_frame,
    )
    _require_positive_ints(
        "--tracklet-max-candidate-pool-per-frame",
        args.tracklet_max_candidate_pool_per_frame,
    )
    _require_positive_ints(
        "--tracklet-max-candidates-per-track-id",
        args.tracklet_max_candidates_per_track_id,
    )
    _require_nonnegative_floats(
        "--tracklet-below-catprob-threshold-penalties",
        args.tracklet_below_catprob_threshold_penalties,
    )
    _require_nonnegative_floats(
        "--tracklet-track-support-weights",
        args.tracklet_track_support_weights,
    )
    _require_nonnegative_floats(
        "--tracklet-max-track-support-rewards",
        args.tracklet_max_track_support_rewards,
    )


def _require_nonempty(name: str, values: list[object]) -> None:
    if not values:
        raise SystemExit(f"{name} must contain at least one value")


def _require_positive_ints(name: str, values: list[int]) -> None:
    _require_nonempty(name, values)
    if any(int(value) < 1 for value in values):
        raise SystemExit(f"{name} values must be >= 1")


def _require_nonnegative_floats(name: str, values: list[float]) -> None:
    _require_nonempty(name, values)
    if any(float(value) < 0.0 for value in values):
        raise SystemExit(f"{name} values must be >= 0")


if __name__ == "__main__":
    raise SystemExit(main())
