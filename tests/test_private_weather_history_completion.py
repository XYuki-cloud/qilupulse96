from __future__ import annotations

import pandas as pd

from da_forecast.config import SHANDONG_SPATIAL_STATIONS
from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
from da_forecast.sources.cache import ParquetCache
from complete_private_weather_history import complete_private_history


TZ = "Asia/Shanghai"


def _panel(index: pd.DatetimeIndex, value: float) -> dict[str, pd.DataFrame]:
    return {
        station.code: pd.DataFrame(
            {column: value for column in STATION_COLUMNS},
            index=index,
        )
        for station in SHANDONG_SPATIAL_STATIONS
    }


def test_complete_private_history_merges_fetched_quarters_into_runtime(tmp_path, monkeypatch):
    existing_index = pd.date_range("2026-08-21 00:00", periods=4, freq="15min", tz=TZ)
    fetched_index = pd.date_range("2026-08-21 01:00", periods=4, freq="15min", tz=TZ)
    cache = ParquetCache(tmp_path / "data/raw")
    for code, frame in _panel(existing_index, 1.0).items():
        cache.save("weather_history_v1", code, "weather", frame)

    calls: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

    def fake_fetch(start, end, *, cache_dir):
        calls.append((start, end, str(cache_dir)))
        return _panel(fetched_index, 2.0)

    monkeypatch.setattr("complete_private_weather_history.disaggregate_station_panel_to_quarters", lambda panel: panel)

    result = complete_private_history(
        tmp_path,
        start_date="2026-08-21",
        end_date="2026-08-23",
        fetcher=fake_fetch,
    )

    assert result["station_count"] == 16
    assert calls[0][0] == pd.Timestamp("2026-08-21", tz=TZ)
    assert calls[0][1] == pd.Timestamp("2026-08-23", tz=TZ)
    restored = cache.load("weather_history_v1", "SD_JINAN", "weather")
    assert restored.loc[pd.Timestamp("2026-08-21 00:00", tz=TZ), "temperature_2m"] == 1.0
    assert restored.loc[pd.Timestamp("2026-08-21 01:00", tz=TZ), "temperature_2m"] == 2.0
