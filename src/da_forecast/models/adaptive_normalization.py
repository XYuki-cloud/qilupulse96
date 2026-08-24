"""Causal per-window price normalization for non-stationary forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriceNormalizationStats:
    """Statistics calculated only from a visible historical price window."""

    center: float
    scale: float


class RevIN:
    """Reversible instance normalization for one price history window."""

    def __init__(self, *, eps: float = 1e-4) -> None:
        self.eps = float(eps)

    def statistics(self, history: np.ndarray) -> PriceNormalizationStats:
        values = _validate_history(history)
        center = float(values.mean())
        scale = float(values.std())
        return PriceNormalizationStats(center=center, scale=max(scale, self.eps))

    def normalize(self, values: np.ndarray, stats: PriceNormalizationStats) -> np.ndarray:
        return (np.asarray(values, dtype=float) - stats.center) / stats.scale

    def denormalize(self, values: np.ndarray, stats: PriceNormalizationStats) -> np.ndarray:
        return np.asarray(values, dtype=float) * stats.scale + stats.center


class RobustRecentNormalizer(RevIN):
    """Median/MAD normalization resistant to negative prices and spikes."""

    def statistics(self, history: np.ndarray) -> PriceNormalizationStats:
        values = _validate_history(history)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = 1.4826 * mad
        if scale < self.eps:
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
        if scale < self.eps:
            scale = float(values.std())
        return PriceNormalizationStats(center=center, scale=max(scale, self.eps))


def recent_state_features(history: np.ndarray) -> np.ndarray:
    """Return causal state descriptors in stable physical ranges."""
    values = _validate_history(history)
    center = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - center)))
    if mad_scale < 1e-4:
        mad_scale = max(float(values.std()), 1e-4)
    return np.asarray(
        [
            center,
            mad_scale,
            float((values < 0).mean()),
            float(values.max() - values.min()),
            float(values.std()),
        ],
        dtype=float,
    )


def _validate_history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("price history must be non-empty and finite")
    return values
