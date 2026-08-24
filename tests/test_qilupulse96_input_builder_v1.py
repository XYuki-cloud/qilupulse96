import numpy as np
import pandas as pd
import pytest

from da_forecast.config import SHANDONG_SPATIAL_STATIONS
from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Spec
from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.input_builder_v1 import CausalInputBuilderV1
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1


def _fixture():
    index = pd.date_range("2026-01-01", periods=10000, freq="15min", tz="Asia/Shanghai")
    price = pd.Series(np.sin(np.arange(len(index)) / 20) * 50 + 300, index=index)
    da = pd.Series(280.0, index=index)
    weather = {}
    cols = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "rain", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m", "solar_elevation", "solar_azimuth_sin", "solar_azimuth_cos", "is_daylight", "clear_sky_ghi", "shortwave_clear_sky_index", "shortwave_radiation_ramp_15m",
    ]
    for station in SHANDONG_SPATIAL_STATIONS:
        weather[station.code] = pd.DataFrame(np.ones((len(index), len(cols))), index=index, columns=cols)
    spec = QiluPulse96V1Spec(station_variable_dim=25, history_extra_dim=18, target_extra_dim=19, n_stations=16)
    bundle = QiluPulse96ProductionBundle(spec, spec.build_model(), PreprocessingStateV1.identity(), {"feature_schema": {}, "station_order": []})
    return bundle, price, da, weather


def test_input_builder_enforces_8640_and_96_shapes():
    bundle, price, da, weather = _fixture()
    result = CausalInputBuilderV1(bundle).build(target_date="2026-04-14", realtime=price, day_ahead=da, history_weather=weather, target_weather=weather)
    assert result.history_price.shape == (8640, 1)
    assert result.history_extra.shape == (8640, 18)
    assert result.history_station_weather.shape == (8640, 16, 25)
    assert result.target_extra.shape == (96, 19)
    assert result.target_station_weather.shape == (96, 16, 25)
    assert result.realtime_cutoff == pd.Timestamp("2026-04-13 10:45", tz="Asia/Shanghai")


def test_input_builder_reports_realtime_and_day_ahead_gaps_together():
    bundle, price, da, weather = _fixture()
    realtime = price.drop(pd.Timestamp("2026-04-13 10:45", tz="Asia/Shanghai"))
    day_ahead = da.drop(pd.Timestamp("2026-04-12 23:45", tz="Asia/Shanghai"))

    with pytest.raises(ValueError, match="Missing 1 required realtime history slots; Missing 1 required day-ahead history slots"):
        CausalInputBuilderV1(bundle).build(
            target_date="2026-04-14",
            realtime=realtime,
            day_ahead=day_ahead,
            history_weather=weather,
            target_weather=weather,
        )


def test_input_builder_realtime_only_bundle_does_not_require_day_ahead_prices():
    bundle, price, _day_ahead, weather = _fixture()
    spec = QiluPulse96V1Spec(station_variable_dim=25, history_extra_dim=14, target_extra_dim=19, n_stations=16)
    realtime_only = QiluPulse96ProductionBundle(
        spec,
        spec.build_model(),
        PreprocessingStateV1.identity(history_extra_dim=14, target_extra_dim=14, station_dim=25),
        {"feature_schema": {}, "station_order": [], "price_features": "realtime_only"},
    )

    result = CausalInputBuilderV1(realtime_only).build(
        target_date="2026-04-14",
        realtime=price,
        day_ahead=None,
        history_weather=weather,
        target_weather=weather,
    )

    assert result.history_extra.shape == (8640, 14)
