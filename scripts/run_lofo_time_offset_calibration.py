"""Run LOFO-safe RF/radar time-offset calibration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.lofo_time_offset_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
