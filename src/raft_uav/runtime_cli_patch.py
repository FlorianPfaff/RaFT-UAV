"""Runtime CLI shim for reproducible experimental settings.

The main CLI predates the runtime-backed radar-covariance and tracklet-Viterbi
settings.  This module keeps the existing parser stable by stripping the new
runtime flags before the original parser sees them, applying the resolved
configuration to the existing runtime layer, and injecting the resolved settings
into baseline metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raft_uav.runtime_cli_config import (
    apply_runtime_environment,
    parse_runtime_config,
    runtime_environment_names_from_argv,
)

_INSTALLED = False
_ORIGINAL_MAIN: Any = None
_ORIGINAL_BASELINE_METRICS: Any = None
_CURRENT_RUNTIME_CONFIG: dict[str, Any] | None = None


def install() -> None:
    """Install the CLI shim once."""

    global _INSTALLED, _ORIGINAL_MAIN, _ORIGINAL_BASELINE_METRICS
    if _INSTALLED:
        return

    from raft_uav import cli

    _ORIGINAL_MAIN = cli.main
    _ORIGINAL_BASELINE_METRICS = cli._baseline_metrics
    cli.main = _main_with_runtime_config
    cli._baseline_metrics = _baseline_metrics_with_runtime_config
    _INSTALLED = True


def _main_with_runtime_config(argv: list[str] | None = None) -> int:
    """Parse runtime flags before delegating to the original CLI."""

    import sys

    global _CURRENT_RUNTIME_CONFIG
    original_argv = list(sys.argv[1:] if argv is None else argv)
    if original_argv[:1] != ["run-baseline"]:
        return _ORIGINAL_MAIN(argv)

    runtime_config, remaining = parse_runtime_config(original_argv)
    explicit_env_names = runtime_environment_names_from_argv(original_argv)
    apply_runtime_environment(
        runtime_config,
        overwrite_existing_env_names=explicit_env_names,
    )
    _CURRENT_RUNTIME_CONFIG = runtime_config
    try:
        return _ORIGINAL_MAIN(remaining if argv is not None else remaining)
    finally:
        _CURRENT_RUNTIME_CONFIG = None


def _baseline_metrics_with_runtime_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Add resolved runtime settings to metrics.json."""

    metrics = _ORIGINAL_BASELINE_METRICS(*args, **kwargs)
    runtime_config = _CURRENT_RUNTIME_CONFIG
    if runtime_config is None:
        runtime_config, _ = parse_runtime_config([])
    metrics["runtime_configuration"] = runtime_config
    metrics["radar_covariance"] = _radar_covariance_description(runtime_config)
    return metrics


def _radar_covariance_description(runtime_config: dict[str, Any]) -> str:
    radar = runtime_config.get("radar_covariance", {})
    mode = str(radar.get("mode", "fixed"))
    if mode == "fixed":
        xy_std = float(radar.get("xy_std_m", 25.0))
        z_std = float(radar.get("z_std_m", 35.0))
        return f"fixed diag({xy_std:.6g}^2, {xy_std:.6g}^2, {z_std:.6g}^2) m^2"
    return (
        "range-angle covariance with "
        f"range_std={float(radar.get('range_std_m', 5.0)):.6g} m, "
        f"azimuth_std={float(radar.get('azimuth_std_deg', 2.0)):.6g} deg, "
        f"elevation_std={float(radar.get('elevation_std_deg', 2.0)):.6g} deg"
    )


def read_metrics_runtime_config(path: Path) -> dict[str, Any]:
    """Small helper for tests and downstream scripts."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("runtime_configuration", {}))
