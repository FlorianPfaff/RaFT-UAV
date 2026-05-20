"""Online time-offset state helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OnlineTimeOffsetEstimator:
    """Scalar online timestamp-offset estimator with a Gaussian state."""

    offset_s: float = 0.0
    variance_s2: float = 1.0
    process_variance_s2: float = 1.0e-4
    min_speed_mps: float = 1.0

    def predict(self, dt_s: float = 0.0) -> None:
        """Apply random-walk process noise."""

        del dt_s
        self.variance_s2 = float(max(self.variance_s2 + self.process_variance_s2, 1e-12))

    def update_from_position_residual(
        self,
        *,
        residual_m: np.ndarray,
        velocity_mps: np.ndarray,
        measurement_variance_m2: float,
    ) -> float:
        """Update the offset from a position residual and return innovation NIS."""

        residual = np.asarray(residual_m, dtype=float).reshape(-1)
        velocity = np.asarray(velocity_mps, dtype=float).reshape(-1)
        if residual.size != velocity.size:
            raise ValueError("residual and velocity must have the same dimension")
        speed2 = float(velocity @ velocity)
        if speed2 < float(self.min_speed_mps) ** 2:
            return float("nan")
        measured_offset = float((residual @ velocity) / speed2)
        variance = max(float(measurement_variance_m2) / speed2, 1e-12)
        innovation = measured_offset - float(self.offset_s)
        innovation_variance = float(self.variance_s2 + variance)
        gain = float(self.variance_s2 / innovation_variance)
        self.offset_s = float(self.offset_s + gain * innovation)
        self.variance_s2 = float(max((1.0 - gain) * self.variance_s2, 1e-12))
        return float((innovation**2) / max(innovation_variance, 1e-12))

    @property
    def std_s(self) -> float:
        """Return the posterior offset standard deviation."""

        return float(np.sqrt(max(self.variance_s2, 0.0)))


def apply_time_offset(frame: pd.DataFrame, *, offset_s: float, column: str = "time_s") -> pd.DataFrame:
    """Return a copy of ``frame`` with ``column`` shifted by ``offset_s``."""

    out = frame.copy()
    if column not in out.columns:
        raise KeyError(column)
    out[column] = pd.to_numeric(out[column], errors="coerce") + float(offset_s)
    return out
