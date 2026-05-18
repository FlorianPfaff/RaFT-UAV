"""CLI shim exposing range-adaptive radar covariance controls for tracklet Viterbi.

The canonical :mod:`raft_uav.tracklet_viterbi_cli` wrapper already defaults to
the range-covariance-aware implementation.  This module adds explicit knobs for
its covariance model without changing the shared base parser.  It is intended
for quick Opt1/Opt2/Opt3 sensitivity runs via::

    python -m raft_uav.tracklet_viterbi_range_cli run-baseline ...
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import os
import sys

from raft_uav import tracklet_viterbi_cli as _tracklet_cli

_RANGE_VARIANT = "range-covariance"
_RANGE_VARIANT_ENV = "RAFT_UAV_TRACKLET_VARIANT"


class _RangeConfigOverlay:
    """Expose the wrapped tracklet config plus range-covariance overrides."""

    def __init__(self, base: object, **overrides: object) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def main(argv: list[str] | None = None) -> int:
    """Run the tracklet CLI with explicit range-covariance config overrides."""

    forwarded_argv, overrides = _extract_range_args(argv)
    with _temporary_range_configuration(overrides):
        return _tracklet_cli.main(forwarded_argv)


def _extract_range_args(argv: list[str] | None) -> tuple[list[str], dict[str, object]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    namespace, forwarded = _range_parser().parse_known_args(raw_argv)
    return forwarded, _overrides_from_namespace(namespace)


def _range_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tracklet-use-range-adaptive-radar-covariance", type=_bool_value)
    parser.add_argument("--tracklet-radar-range-xy-floor-std-m", type=_nonnegative_float)
    parser.add_argument("--tracklet-radar-range-z-floor-std-m", type=_nonnegative_float)
    parser.add_argument("--tracklet-radar-range-xy-scale", type=_nonnegative_float)
    parser.add_argument("--tracklet-radar-range-z-scale", type=_nonnegative_float)
    return parser


def _overrides_from_namespace(namespace: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    _maybe_add(
        overrides,
        "use_range_adaptive_radar_covariance",
        namespace.tracklet_use_range_adaptive_radar_covariance,
    )
    _maybe_add(overrides, "radar_range_xy_floor_std_m", namespace.tracklet_radar_range_xy_floor_std_m)
    _maybe_add(overrides, "radar_range_z_floor_std_m", namespace.tracklet_radar_range_z_floor_std_m)
    _maybe_add(overrides, "radar_range_xy_scale", namespace.tracklet_radar_range_xy_scale)
    _maybe_add(overrides, "radar_range_z_scale", namespace.tracklet_radar_range_z_scale)
    return overrides


def _maybe_add(overrides: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        overrides[key] = value


@contextmanager
def _temporary_range_configuration(overrides: Mapping[str, object]):
    original_config_factory = _tracklet_cli._tracklet_config_from_environment
    previous_variant = os.environ.get(_RANGE_VARIANT_ENV)

    def config_factory() -> _RangeConfigOverlay:
        return _RangeConfigOverlay(original_config_factory(), **dict(overrides))

    _tracklet_cli._tracklet_config_from_environment = config_factory
    os.environ[_RANGE_VARIANT_ENV] = _RANGE_VARIANT
    try:
        yield
    finally:
        _tracklet_cli._tracklet_config_from_environment = original_config_factory
        if previous_variant is None:
            os.environ.pop(_RANGE_VARIANT_ENV, None)
        else:
            os.environ[_RANGE_VARIANT_ENV] = previous_variant


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
