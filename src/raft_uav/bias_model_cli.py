"""CLI wrapper that enables runtime RF/radar bias correction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from raft_uav.calibration import bias_runtime
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV


def main(argv: list[str] | None = None) -> int:
    """Run the tracklet-enabled CLI with optional runtime bias correction."""

    args = list(sys.argv[1:] if argv is None else argv)
    bias_model, remaining = _extract_bias_model(args)
    if bias_model is not None:
        os.environ[BIAS_MODEL_ENV] = str(bias_model)
        bias_runtime.install()
    _refresh_cli_normalizers()
    from raft_uav import tracklet_viterbi_cli

    return tracklet_viterbi_cli.main(remaining)


def _refresh_cli_normalizers() -> None:
    from raft_uav import cli as base_cli
    from raft_uav.io import aerpaw

    base_cli.normalize_rf = aerpaw.normalize_rf
    base_cli.normalize_radar = aerpaw.normalize_radar


def _extract_bias_model(argv: list[str]) -> tuple[Path | None, list[str]]:
    remaining: list[str] = []
    bias_model: Path | None = None
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--bias-model":
            if index + 1 >= len(argv):
                raise SystemExit("--bias-model requires a path")
            bias_model = Path(argv[index + 1])
            index += 2
            continue
        if value.startswith("--bias-model="):
            bias_model = Path(value.split("=", 1)[1])
            index += 1
            continue
        remaining.append(value)
        index += 1
    return bias_model, remaining


if __name__ == "__main__":
    raise SystemExit(main())