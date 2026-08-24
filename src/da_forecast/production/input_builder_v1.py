"""Causal one-day QiluPulse-96 tensor construction without backtest imports."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from da_forecast.config import TIMEZONE
from da_forecast.features.calendar_v01 import build_calendar_v01
from da_forecast.models.adaptive_normalization import RobustRecentNormalizer, recent_state_features
from da_forecast.sources.spatial_weather_v01 import validate_station_weather
from .bundle_v1 import QiluPulse96ProductionBundle
from .feature_schema_v1 import HISTORY_EXTRA_COLUMNS, STATION_COLUMNS, TARGET_EXTRA_COLUMNS


@dataclass(frozen=True)
class CausalInputBundle:
    target_date: pd.Timestamp
    realtime_cutoff: pd.Timestamp
    day_ahead_cutoff: pd.Timestamp
    history_price: np.ndarray
    history_extra: np.ndarray
    history_station_weather: np.ndarray
    target_extra: np.ndarray
    target_station_weather: np.ndarray
    state_features: np.ndarray
    normalization_center: float
    normalization_scale: float


class CausalInputBuilderV1:
    context_slots = 90 * 96

    def __init__(self, bundle: QiluPulse96ProductionBundle, *, calendar_reference_dir: str | None = None) -> None:
        self.bundle = bundle
        self.calendar_reference_dir = calendar_reference_dir
        self.normalizer = RobustRecentNormalizer(eps=float(bundle.preprocessing.robust_normalizer.get("eps", 1e-4)))

    def build(self, *, target_date: str | pd.Timestamp, realtime: pd.Series, day_ahead: pd.Series | None, history_weather: dict[str, pd.DataFrame], target_weather: dict[str, pd.DataFrame]) -> CausalInputBundle:
        target = _local_day(target_date)
        realtime_cutoff = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
        day_ahead_cutoff = target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
        history_index = pd.date_range(realtime_cutoff - pd.Timedelta(minutes=15 * (self.context_slots - 1)), realtime_cutoff, freq="15min", tz=TIMEZONE)
        target_index = pd.date_range(target, periods=96, freq="15min", tz=TIMEZONE)
        realtime = _series(realtime)
        price = realtime.reindex(history_index)
        missing_realtime = int(price.isna().sum())
        calendar_history = build_calendar_v01(history_index, reference_dir=self.calendar_reference_dir)
        calendar_target = build_calendar_v01(target_index, reference_dir=self.calendar_reference_dir)
        missing_messages = []
        if missing_realtime:
            missing_messages.append(f"Missing {missing_realtime} required realtime history slots")
        realtime_only = self.bundle.spec.history_extra_dim == len(calendar_history.columns)
        if realtime_only:
            da_available = None
            da_values = None
        else:
            if day_ahead is None:
                raise ValueError("This production bundle requires day-ahead history; use the realtime-only retrained bundle")
            day_ahead = _series(day_ahead)
            da_available = history_index <= day_ahead_cutoff
            da_values = day_ahead.reindex(history_index)
            missing_day_ahead = int(da_values.loc[da_available].isna().sum())
            if missing_day_ahead:
                missing_messages.append(f"Missing {missing_day_ahead} required day-ahead history slots")
        if missing_messages:
            raise ValueError("; ".join(missing_messages))
        history_base = calendar_history.copy()
        if realtime_only:
            history_scaled = self.bundle.preprocessing.history_extra.transform(calendar_history.to_numpy(dtype=np.float32))
            history_model = history_scaled.astype(np.float32)
        else:
            raw_da = da_values.fillna(0.0).to_numpy(dtype=np.float32)
            raw_spread = price.to_numpy(dtype=np.float32) - raw_da
            history_base["history_day_ahead"] = raw_da
            history_base["history_rt_da_spread"] = raw_spread
            history_scaled = self.bundle.preprocessing.history_extra.transform(history_base[list(HISTORY_EXTRA_COLUMNS)].to_numpy(dtype=np.float32))
            history_scaled[~da_available, -2:] = 0.0
            history_model = np.column_stack([history_scaled, da_available.astype(np.float32), da_available.astype(np.float32)]).astype(np.float32)
        state_raw = recent_state_features(price.to_numpy(dtype=float))
        state = self.bundle.preprocessing.state_features.transform(state_raw[None, :])[0].astype(np.float32)
        target_base = self.bundle.preprocessing.target_extra.transform(calendar_target[list(TARGET_EXTRA_COLUMNS)].to_numpy(dtype=np.float32))
        target_model = np.column_stack([target_base, np.repeat(state[None, :], 96, axis=0)]).astype(np.float32)
        history_station = self._station_tensor(history_weather, history_index)
        target_station = self._station_tensor(target_weather, target_index)
        stats = self.normalizer.statistics(price.to_numpy(dtype=float))
        return CausalInputBundle(target, realtime_cutoff, day_ahead_cutoff, self.normalizer.normalize(price.to_numpy(dtype=float), stats).astype(np.float32)[:, None], history_model, history_station, target_model, target_station, state, stats.center, stats.scale)

    def _station_tensor(self, panel: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> np.ndarray:
        validate_station_weather(panel)
        arrays = []
        for code in sorted(panel):
            values = panel[code].reindex(index)
            if any(column not in values for column in STATION_COLUMNS) or values[list(STATION_COLUMNS)].isna().any().any():
                raise ValueError(f"Incomplete weather for station {code}")
            arrays.append(values[list(STATION_COLUMNS)].to_numpy(dtype=np.float32))
        raw = np.stack(arrays, axis=1)
        return self.bundle.preprocessing.station_weather.transform(raw).astype(np.float32)


def _series(values: pd.Series) -> pd.Series:
    result = values.copy().astype(float)
    index = pd.DatetimeIndex(result.index)
    result.index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    return result.sort_index()


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)
    return stamp.normalize()
