"""Radar-RF Fusion Tracking for UAVs."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from raft_uav.baselines.radar_covariance_runtime import install as _install_radar_covariance
from raft_uav.baselines.tracklet_viterbi_runtime import install as _install_tracklet_viterbi
from raft_uav.runtime_cli_patch import install as _install_runtime_cli_patch

__all__ = ["__version__"]

__version__ = "0.1.0"

_install_radar_covariance()
_install_tracklet_viterbi()
_install_runtime_cli_patch()
