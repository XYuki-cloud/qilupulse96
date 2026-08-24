"""Explicit, pickle-free preprocessing state for production inference."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArrayScalerState:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]} features, got {array.shape[-1]}")
        return (array - self.mean) / np.where(self.scale > 0, self.scale, 1.0)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array * self.scale + self.mean

    @classmethod
    def constant(cls, dimension: int) -> "ArrayScalerState":
        if dimension < 1:
            raise ValueError("dimension must be positive")
        return cls(np.zeros(dimension, dtype=np.float32), np.ones(dimension, dtype=np.float32))


@dataclass(frozen=True)
class PreprocessingStateV1:
    price: ArrayScalerState
    history_extra: ArrayScalerState
    target_extra: ArrayScalerState
    station_weather: ArrayScalerState
    state_features: ArrayScalerState
    robust_normalizer: dict[str, float]

    def save_npz(self, path) -> None:
        np.savez_compressed(
            path,
            price_mean=self.price.mean, price_scale=self.price.scale,
            history_extra_mean=self.history_extra.mean, history_extra_scale=self.history_extra.scale,
            target_extra_mean=self.target_extra.mean, target_extra_scale=self.target_extra.scale,
            station_weather_mean=self.station_weather.mean, station_weather_scale=self.station_weather.scale,
            state_features_mean=self.state_features.mean, state_features_scale=self.state_features.scale,
        )

    @classmethod
    def load_npz(cls, path, *, robust_normalizer: dict[str, float] | None = None) -> "PreprocessingStateV1":
        with np.load(path) as data:
            return cls(
                price=ArrayScalerState(data["price_mean"].astype(np.float32), data["price_scale"].astype(np.float32)),
                history_extra=ArrayScalerState(data["history_extra_mean"].astype(np.float32), data["history_extra_scale"].astype(np.float32)),
                target_extra=ArrayScalerState(data["target_extra_mean"].astype(np.float32), data["target_extra_scale"].astype(np.float32)),
                station_weather=ArrayScalerState(data["station_weather_mean"].astype(np.float32), data["station_weather_scale"].astype(np.float32)),
                state_features=ArrayScalerState(data["state_features_mean"].astype(np.float32), data["state_features_scale"].astype(np.float32)),
                robust_normalizer=dict(robust_normalizer or {"eps": 1e-4, "mad_multiplier": 1.4826}),
            )

    @classmethod
    def identity(cls, *, history_extra_dim: int = 16, target_extra_dim: int = 14, station_dim: int = 25, state_dim: int = 5) -> "PreprocessingStateV1":
        return cls(
            price=ArrayScalerState.constant(1),
            history_extra=ArrayScalerState.constant(history_extra_dim),
            target_extra=ArrayScalerState.constant(target_extra_dim),
            station_weather=ArrayScalerState.constant(station_dim),
            state_features=ArrayScalerState.constant(state_dim),
            robust_normalizer={"eps": 1e-4, "mad_multiplier": 1.4826},
        )
