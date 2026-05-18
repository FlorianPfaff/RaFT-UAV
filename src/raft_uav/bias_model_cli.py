"""CLI wrapper that enables runtime RF/radar bias correction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from raft_uav import cli as _base_cli
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV


def main(argv: list[str] | None = None) -> int:
    """Run the standard CLI with optional runtime bias correction enabled."""

    args = list(sys.argv[1:] if argv is None else argv)
    bias_model, remaining = _extract_bias_model(args)
    if bias_model is not None:
        os.environ[BIAS_MODEL_ENV] = str(bias_model)
    return _base_cli.main(remaining)


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
