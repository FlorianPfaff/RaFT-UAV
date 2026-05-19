from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.nis import NisCalibrationSettings, estimate_nis_covariance_scales


def test_estimate_nis_covariance_scales_recovers_median_multiplier() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["rf"] * 20 + ["radar"] * 20,
            "measurement_dim": [2] * 20 + [3] * 20,
            "accepted": [True] * 40,
            "nis": [2.0] * 20 + [7.5] * 20,
        }
    )

    groups = estimate_nis_covariance_scales(
        diagnostics,
        settings=NisCalibrationSettings(min_samples=5, calibration_quantile=0.5),
    )

    scales = {group.key["source"]: group.covariance_scale for group in groups}
    assert np.isclose(scales["rf"], 2.0 / 1.386294361119891, rtol=1e-6)
    assert np.isclose(scales["radar"], 7.5 / 2.3659738843753377, rtol=1e-6)


def test_estimate_nis_covariance_scales_ignores_rejected_updates() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["radar"] * 10,
            "measurement_dim": [3] * 10,
            "accepted": [True] * 8 + [False] * 2,
            "nis": [3.0] * 8 + [300.0] * 2,
        }
    )

    groups = estimate_nis_covariance_scales(
        diagnostics,
        settings=NisCalibrationSettings(min_samples=8, calibration_quantile=0.5),
    )

    assert len(groups) == 1
    assert groups[0].samples == 8
    assert groups[0].observed_quantile == 3.0


def test_estimate_nis_covariance_scales_respects_clipping() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["rf"] * 8,
            "measurement_dim": [2] * 8,
            "accepted": [True] * 8,
            "nis": [1_000.0] * 8,
        }
    )

    groups = estimate_nis_covariance_scales(
        diagnostics,
        settings=NisCalibrationSettings(min_samples=8, max_scale=10.0),
    )

    assert groups[0].covariance_scale == 10.0
