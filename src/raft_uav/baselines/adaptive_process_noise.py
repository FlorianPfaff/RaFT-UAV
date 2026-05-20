"""Adaptive process-noise heuristics from innovation statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AdaptiveProcessNoiseConfig:
    """Parameters for NIS-driven acceleration-noise adaptation."""

    base_acceleration_std_mps2: float = 4.0
    min_scale: float = 0.35
    max_scale: float = 4.0
    ewma_alpha: float = 0.05
    high_nis_ratio: float = 1.5
    low_nis_ratio: float = 0.6
    scale_gain: float = 0.5

    def __post_init__(self) -> None:
        if self.base_acceleration_std_mps2 <= 0.0:
            raise ValueError("base_acceleration_std_mps2 must be positive")
        if self.min_scale <= 0.0 or self.max_scale < self.min_scale:
            raise ValueError("scale bounds must be positive and ordered")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")


@dataclass
class RollingNISAdaptiveAcceleration:
    """Maintain per-source EWMA NIS ratios and return an acceleration scale."""

    config: AdaptiveProcessNoiseConfig = field(default_factory=AdaptiveProcessNoiseConfig)
    ratios_by_source: dict[str, float] = field(default_factory=dict)
    updates_by_source: dict[str, int] = field(default_factory=dict)

    def observe(self, *, source: str, measurement_dim: int, nis: float, accepted: bool = True) -> float:
        """Ingest one innovation and return the updated source ratio."""

        if not accepted or measurement_dim <= 0 or not np.isfinite(nis):
            return self.ratios_by_source.get(source, 1.0)
        ratio = max(float(nis) / float(measurement_dim), 0.0)
        previous = self.ratios_by_source.get(source, 1.0)
        alpha = float(self.config.ewma_alpha)
        updated = (1.0 - alpha) * previous + alpha * ratio
        self.ratios_by_source[source] = updated
        self.updates_by_source[source] = self.updates_by_source.get(source, 0) + 1
        return updated

    def acceleration_std_mps2(self, source_weights: Mapping[str, float] | None = None) -> float:
        """Return the adapted acceleration standard deviation."""

        if not self.ratios_by_source:
            return float(self.config.base_acceleration_std_mps2)
        if source_weights:
            numerator = 0.0
            denominator = 0.0
            for source, ratio in self.ratios_by_source.items():
                weight = float(source_weights.get(source, 0.0))
                numerator += weight * ratio
                denominator += weight
            ratio = numerator / denominator if denominator > 0.0 else np.mean(list(self.ratios_by_source.values()))
        else:
            ratio = float(np.mean(list(self.ratios_by_source.values())))
        return float(self.config.base_acceleration_std_mps2 * adaptive_scale_from_ratio(ratio, self.config))


def adaptive_scale_from_ratio(ratio: float, config: AdaptiveProcessNoiseConfig) -> float:
    """Map a normalized NIS ratio to a bounded process-noise scale."""

    ratio = float(ratio)
    if ratio > float(config.high_nis_ratio):
        scale = 1.0 + float(config.scale_gain) * (ratio - float(config.high_nis_ratio))
    elif ratio < float(config.low_nis_ratio):
        scale = 1.0 - float(config.scale_gain) * (float(config.low_nis_ratio) - ratio)
    else:
        scale = 1.0
    return float(np.clip(scale, float(config.min_scale), float(config.max_scale)))
