"""Radar-RF Fusion Tracking for UAVs."""

from raft_uav.baselines.radar_covariance_runtime import install as _install_radar_covariance
from raft_uav.baselines.tracklet_viterbi_runtime import install as _install_tracklet_viterbi

__all__ = ["__version__"]

__version__ = "0.1.0"

_install_radar_covariance()
_install_tracklet_viterbi()
