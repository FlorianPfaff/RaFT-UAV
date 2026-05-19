from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as _radar_association
from raft_uav.baselines import tracklet_viterbi as _base
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_range_covariance import (
    _radar_row_covariance,
    _range_adaptive_covariance_hooks,
    _write_radar_covariance_diagnostics,
)


def test_range_adaptive_radar_covariance_projects_spherical_noise_to_enu() -> None:
    default_covariance = np.diag([1.0, 1.0, 1.0])
    config = _Config(
        use_range_adaptive_radar_covariance=True,
        radar_range_std_m=5.0,
        radar_azimuth_std_rad=0.020,
        radar_elevation_std_rad=0.030,
        radar_range_xy_floor_std_m=0.0,
        radar_range_z_floor_std_m=0.0,
    )
    row = pd.Series(
        {
            "east_m": 1000.0,
            "north_m": 0.0,
            "up_m": 0.0,
            "range_m": 1000.0,
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.allclose(np.sqrt(np.diag(covariance)), [5.0, 20.0, 30.0])
    assert np.allclose(covariance - np.diag(np.diag(covariance)), 0.0)


def test_range_adaptive_radar_covariance_keeps_cartesian_lower_bound() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    row = pd.Series(
        {
            "east_m": 100.0,
            "north_m": 0.0,
            "up_m": 0.0,
            "range_m": 100.0,
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.all(np.diag(covariance) >= np.diag(default_covariance))
    assert np.allclose(covariance, default_covariance)


def test_range_adaptive_radar_covariance_preserves_enu_correlations() -> None:
    default_covariance = np.diag([1.0, 1.0, 1.0])
    config = _Config(
        use_range_adaptive_radar_covariance=True,
        radar_range_std_m=1.0,
        radar_azimuth_std_rad=0.020,
        radar_elevation_std_rad=0.001,
        radar_range_xy_floor_std_m=0.0,
        radar_range_z_floor_std_m=0.0,
    )
    row = pd.Series(
        {
            "east_m": 1000.0,
            "north_m": 1000.0,
            "up_m": 0.0,
            "range_m": float(np.sqrt(2.0) * 1000.0),
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert covariance[0, 1] < 0.0
    assert np.isclose(covariance[0, 1], covariance[1, 0])
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_range_adaptive_radar_covariance_can_be_disabled() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = _Config(use_range_adaptive_radar_covariance=False)
    row = pd.Series(
        {
            "east_m": 1200.0,
            "north_m": 0.0,
            "up_m": 0.0,
            "range_m": 1200.0,
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.allclose(covariance, default_covariance)


def test_range_adaptive_radar_covariance_falls_back_without_valid_geometry() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    row = pd.Series({"cat_prob_uav": 0.9})

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.allclose(covariance, default_covariance)


def test_range_adaptive_radar_covariance_supports_custom_scales_and_floors() -> None:
    default_covariance = np.diag([10.0**2, 10.0**2, 10.0**2])
    config = _Config(
        use_range_adaptive_radar_covariance=True,
        radar_range_xy_floor_std_m=50.0,
        radar_range_z_floor_std_m=40.0,
        radar_range_xy_scale=0.020,
        radar_range_z_scale=0.070,
    )
    row = pd.Series(
        {
            "east_m": 1000.0,
            "north_m": 0.0,
            "up_m": 0.0,
            "range_m": 1000.0,
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.isclose(np.sqrt(covariance[0, 0]), 50.0)
    assert np.isclose(np.sqrt(covariance[1, 1]), 50.0)
    assert np.isclose(np.sqrt(covariance[2, 2]), 70.0)


def test_radar_covariance_diagnostics_mark_adaptive_rows() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    row_covariance = np.array(
        [
            [40.0**2, 12.0, 13.0],
            [12.0, 45.0**2, 14.0],
            [13.0, 14.0, 55.0**2],
        ]
    )
    row = pd.Series({"range_m": 1200.0})

    _write_radar_covariance_diagnostics(row, row_covariance, default_covariance)

    assert float(row["association_radar_std_east_m"]) == 40.0
    assert float(row["association_radar_std_north_m"]) == 45.0
    assert float(row["association_radar_std_up_m"]) == 55.0
    assert float(row["association_radar_xy_std_m"]) == 45.0
    assert float(row["association_radar_z_std_m"]) == 55.0
    assert float(row["association_radar_cov_en_m2"]) == 12.0
    assert float(row["association_radar_cov_eu_m2"]) == 13.0
    assert float(row["association_radar_cov_nu_m2"]) == 14.0
    assert row["association_radar_covariance_model"] == "range-angle"
    assert bool(row["association_radar_covariance_adaptive"])


def test_range_adaptive_covariance_hooks_restore_patched_functions() -> None:
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    original_candidate_cost_terms = _base._candidate_cost_terms
    original_radar_row_to_measurement = _radar_association._radar_row_to_measurement

    with _range_adaptive_covariance_hooks(config):
        assert _base._candidate_cost_terms is not original_candidate_cost_terms
        assert _radar_association._radar_row_to_measurement is not original_radar_row_to_measurement

    assert _base._candidate_cost_terms is original_candidate_cost_terms
    assert _radar_association._radar_row_to_measurement is original_radar_row_to_measurement


class _Config:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)
