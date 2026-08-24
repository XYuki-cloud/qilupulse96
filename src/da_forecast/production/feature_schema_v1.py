"""Frozen feature ordering for the QiluPulse-96 production contract."""

from __future__ import annotations

from da_forecast.sources.spatial_weather_v01 import DETAIL_WEATHER_COLUMNS

SOLAR_COLUMNS = (
    "solar_elevation", "solar_azimuth_sin", "solar_azimuth_cos", "is_daylight",
    "clear_sky_ghi", "shortwave_clear_sky_index", "shortwave_radiation_ramp_15m",
)
CALENDAR_COLUMNS = (
    "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "day_of_year_progress",
    "annual_sin", "annual_cos", "month_of_year", "quarter", "month_progress",
    "is_public_holiday", "is_adjusted_workday", "is_weekend_effective", "is_regular_workday",
)
HISTORY_EXTRA_COLUMNS = CALENDAR_COLUMNS + ("history_day_ahead", "history_rt_da_spread")
TARGET_EXTRA_COLUMNS = CALENDAR_COLUMNS
STATE_COLUMNS = ("recent_price_median", "robust_scale", "negative_share", "peak_to_valley", "std")
STATION_COLUMNS = DETAIL_WEATHER_COLUMNS + SOLAR_COLUMNS

FEATURE_SCHEMA_VERSION = "qilupulse96_feature_schema_v1"

def feature_schema(*, realtime_only: bool = False) -> dict[str, object]:
    history_base = list(CALENDAR_COLUMNS) if realtime_only else list(HISTORY_EXTRA_COLUMNS)
    history_model = list(history_base) if realtime_only else list(HISTORY_EXTRA_COLUMNS) + ["day_ahead_available", "rt_da_spread_available"]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "history_extra_base": history_base,
        "target_extra_base": list(TARGET_EXTRA_COLUMNS),
        "history_extra_model": history_model,
        "target_extra_model": list(TARGET_EXTRA_COLUMNS) + list(STATE_COLUMNS),
        "station_weather": list(STATION_COLUMNS),
        "state_features": list(STATE_COLUMNS),
        "station_count": 16,
        "history_extra_base_dim": len(history_base),
        "history_extra_model_dim": len(history_model),
        "price_features": "realtime_only" if realtime_only else "realtime_plus_day_ahead",
        "target_extra_base_dim": 14,
        "target_extra_model_dim": 19,
        "station_variable_dim": 25,
    }
