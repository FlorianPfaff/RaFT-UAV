"""IMM replay hooks for tracklet-Viterbi association.

The tracklet-Viterbi association variants select radar rows with richer
sequence-level logic, but their replay code historically instantiated the CV
Kalman tracker directly.  This module keeps the association code unchanged and
switches only the replay tracker class to the existing IMM implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pandas as pd

from raft_uav.baselines import tracklet_viterbi as _base_tracklet_viterbi
from raft_uav.baselines import tracklet_viterbi_result as _tracklet_viterbi_result
from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker

TrackletRunner = Callable[..., tuple[list[dict[str, object]], pd.DataFrame]]


def with_imm_tracklet_tracker(runner: TrackletRunner) -> TrackletRunner:
    """Return ``runner`` with IMM replay instead of CV replay.

    The hook covers both ordinary tracklet-Viterbi replay and the replay helper
    used by fixed-lag/result-producing variants.  Selection logic, candidate
    costs, and range-covariance hooks remain delegated to the wrapped runner.
    """

    def run_with_imm_tracklet_tracker(
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], pd.DataFrame]:
        with _imm_tracker_hooks():
            records, selected = runner(**kwargs)
        for record in records:
            record["tracker"] = "imm"
            record["motion_model"] = "imm"
        return records, selected

    run_with_imm_tracklet_tracker.__name__ = (
        f"imm_{getattr(runner, '__name__', 'tracklet_runner')}"
    )
    run_with_imm_tracklet_tracker.__doc__ = runner.__doc__
    return run_with_imm_tracklet_tracker


@contextmanager
def _imm_tracker_hooks() -> Iterator[None]:
    """Temporarily use IMM where tracklet replay imports the CV tracker."""

    originals = [
        (
            _base_tracklet_viterbi,
            _base_tracklet_viterbi.AsyncConstantVelocityKalmanTracker,
        ),
        (
            _tracklet_viterbi_result,
            _tracklet_viterbi_result.AsyncConstantVelocityKalmanTracker,
        ),
    ]
    try:
        for module, _ in originals:
            module.AsyncConstantVelocityKalmanTracker = AsyncInteractingMultipleModelTracker
        yield
    finally:
        for module, tracker_class in originals:
            module.AsyncConstantVelocityKalmanTracker = tracker_class
