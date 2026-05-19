"""Range-adaptive radar covariance for tracklet-Viterbi association.

The base tracklet-Viterbi implementation uses one Cartesian covariance for all
selected radar rows.  This wrapper keeps the existing retention-aware Viterbi
path but patches the replay and RF-anchor scoring hooks so long-range radar
rows are down-weighted according to their ``range_m`` field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi_retention import (
    run_async_cv_baseline_with_tracklet_viterbi_association as _run_retention_association,
)

TrackletViterbiAssociationConfig = _base.TrackletViterbiAssociationConfig
DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE = True
DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M = 20.0
DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M = 30.0
DEFAULT_RADAR_RANGE_XY_SCALE = 0.035
DEFAULT_RADAR_RANGE_Z_SCALE = 0.050


def run_async_cv_baseline_with_tracklet_viterbi_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
    replay_tracker_kind: str = "cv",
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run retention-aware Viterbi with range-adaptive radar covariance."""

    cfg = config or TrackletViterbiAssociationConfig()
    with _range_adaptive_covariance_hooks(cfg):
        return _run_retention_association(
            rf_measurements=rf_measurements,
            radar=radar,
            acceleration_std_mps2=acceleration_std_mps2,
            radar_xy_std_m=radar_xy_std_m,
            radar_z_std_m=radar_z_std_m,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
            replay_tracker_kind=replay_tracker_kind,
        )


@contextmanager
def _range_adaptive_covariance_hooks(config: TrackletViterbiAssociationConfig):
    original_candidate_cost_terms = _base._candidate_cost_terms
    original_radar_row_to_measurement = _radar_association._radar_row_to_measurement

    def candidate_cost_terms_with_adaptive_covariance(
        *,
        row: pd.Series,
        position: np.ndarray,
        anchor: _base._AnchorState | None,
        covariance: np.ndarray,
        config: TrackletViterbiAssociationConfig,
    ) -> tuple[float, float, float]:
        row_covariance = _radar_row_covariance(row, covariance, config)
        return original_candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=row_covariance,
            config=config,
        )

    def radar_row_to_measurement_with_adaptive_covariance(
        row: pd.Series,
        covariance: np.ndarray,
    ) -> TrackingMeasurement:
        row_covariance = _radar_row_covariance(row, covariance, config)
        _write_radar_covariance_diagnostics(row, row_covariance, covariance)
        return original_radar_row_to_measurement(row, row_covariance)

    _base._candidate_cost_terms = candidate_cost_terms_with_adaptive_covariance
    _radar_association._radar_row_to_measurement = radar_row_to_measurement_with_adaptive_covariance
    try:
        yield
    finally:
        _base._candidate_cost_terms = original_candidate_cost_terms
        _radar_association._radar_row_to_measurement = original_radar_row_to_measurement


def _radar_row_covariance(
    row: pd.Series,
    default_covariance: np.ndarray,
    config: Any,
) -> np.ndarray:
    """Return range-adaptive ENU radar covariance for one radar row.

    The fixed covariance remains a lower bound.  Long-range radar rows are
    down-weighted by inflating horizontal and vertical standard deviations from
    ``range_m``.  This approximates angular-error growth without requiring a
    native polar radar update.
    """

    default_covariance = np.asarray(default_covariance, dtype=float)
    if not bool(
        getattr(
            config,
            "use_range_adaptive_radar_covariance",
            DEFAULT_USE_RANGE_ADAPTIVE_RADAR_COVARIANCE,
        )
    ):
        return default_covariance

    range_m = _base._optional_float(row.get("range_m"))
    if range_m is None or range_m <= 0.0:
        return default_covariance

    default_xy_std_m = float(
        np.sqrt(max(default_covariance[0, 0], default_covariance[1, 1], 0.0))
    )
    default_z_std_m = float(np.sqrt(max(default_covariance[2, 2], 0.0)))
    xy_std_m = max(
        default_xy_std_m,
        float(
            getattr(
                config,
                "radar_range_xy_floor_std_m",
                DEFAULT_RADAR_RANGE_XY_FLOOR_STD_M,
            )
        ),
        float(getattr(config, "radar_range_xy_scale", DEFAULT_RADAR_RANGE_XY_SCALE))
        * float(range_m),
    )
    z_std_m = max(
        default_z_std_m,
        float(getattr(config, "radar_range_z_floor_std_m", DEFAULT_RADAR_RANGE_Z_FLOOR_STD_M)),
        float(getattr(config, "radar_range_z_scale", DEFAULT_RADAR_RANGE_Z_SCALE))
        * float(range_m),
    )
    return np.diag([xy_std_m**2, xy_std_m**2, z_std_m**2])


def _write_radar_covariance_diagnostics(
    row: pd.Series,
    row_covariance: np.ndarray,
    default_covariance: np.ndarray,
) -> None:
    """Attach selected-row covariance diagnostics for ablation analysis."""

    row_covariance = np.asarray(row_covariance, dtype=float)
    default_covariance = np.asarray(default_covariance, dtype=float)
    row["association_radar_xy_std_m"] = float(np.sqrt(max(row_covariance[0, 0], 0.0)))
    row["association_radar_z_std_m"] = float(np.sqrt(max(row_covariance[2, 2], 0.0)))
    row["association_radar_covariance_adaptive"] = bool(
        not np.allclose(row_covariance, default_covariance)
    )
