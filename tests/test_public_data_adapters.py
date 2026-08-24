from __future__ import annotations

import pandas as pd
import pytest

from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.openmeteo import ZONE_WEATHER_COORDS, _date_chunks, fetch_weather_at


def test_public_weather_coordinates_are_shandong_only() -> None:
    assert ZONE_WEATHER_COORDS["SD"] == ZONE_WEATHER_COORDS["SD_JINAN"]
    assert ZONE_WEATHER_COORDS
    assert all(code == "SD" or code.startswith("SD_") for code in ZONE_WEATHER_COORDS)


def test_parquet_cache_merge_replaces_overlapping_rows(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    first_index = pd.date_range("2026-08-01", periods=2, freq="h", tz="Asia/Shanghai")
    second_index = pd.date_range("2026-08-01 01:00", periods=2, freq="h", tz="Asia/Shanghai")
    cache.save("demo", "SD_JINAN", "weather", pd.DataFrame({"value": [1.0, 2.0]}, index=first_index))
    cache.merge("demo", "SD_JINAN", "weather", pd.DataFrame({"value": [20.0, 3.0]}, index=second_index))

    restored = cache.load("demo", "SD_JINAN", "weather")
    assert restored is not None
    assert restored["value"].tolist() == [1.0, 20.0, 3.0]
    assert cache.get_cached_range("demo", "SD_JINAN", "weather") == (first_index[0], second_index[-1])


def test_date_chunks_rejects_empty_interval() -> None:
    start = pd.Timestamp("2026-08-01", tz="Asia/Shanghai")
    with pytest.raises(ValueError, match="end must be later than start"):
        _date_chunks(start, start)


def test_weather_fetch_does_not_treat_an_internal_cache_gap_as_complete(tmp_path, monkeypatch) -> None:
    cached_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-01 00:00", tz="Asia/Shanghai"),
            pd.Timestamp("2026-08-02 23:00", tz="Asia/Shanghai"),
        ]
    )
    cache = ParquetCache(tmp_path)
    cache.save("openmeteo", "SD_JINAN", "weather", pd.DataFrame({"temperature_2m": [1.0, 2.0]}, index=cached_index))
    requested = pd.date_range("2026-08-01", "2026-08-02 23:00", freq="h", tz="Asia/Shanghai")
    response = {
        "hourly": {
            "time": requested.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [3.0] * len(requested),
        }
    }
    calls: list[dict] = []

    def request(params):
        calls.append(params)
        return response

    monkeypatch.setattr("da_forecast.sources.openmeteo._request_with_retry", request)

    restored = fetch_weather_at(
        36.65,
        117.0,
        pd.Timestamp("2026-08-01", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-03", tz="Asia/Shanghai"),
        tmp_path,
        zone="SD_JINAN",
        variables=("temperature_2m",),
    )

    assert calls
    assert len(restored) == len(requested)
    assert restored.index.equals(requested.tz_convert("UTC"))
