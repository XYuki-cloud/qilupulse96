from __future__ import annotations

import json

import pandas as pd

from da_forecast.config import SHANDONG_SPATIAL_STATIONS
from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
from da_forecast.sources.openmeteo import ForecastWeatherSnapshot
from fetch_qilupulse96_weather_snapshot import fetch_and_store_snapshot


TZ = "Asia/Shanghai"


def test_fetch_and_store_snapshot_archives_all_stations_with_run_identity(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-08-23", tz=TZ)
    issued = pd.Timestamp("2026-08-22T12:00:00+08:00")
    model_run = pd.Timestamp("2026-08-21T18:00:00Z")
    hourly_index = pd.date_range(target, periods=24, freq="h", tz=TZ).tz_convert("UTC")

    def fake_fetch(lat, lon, **kwargs):
        weather = pd.DataFrame(
            {column: 1.0 for column in STATION_COLUMNS},
            index=hourly_index,
        )
        return ForecastWeatherSnapshot(
            weather=weather,
            payloads=[
                {
                    "request": {
                        "run": model_run.strftime("%Y-%m-%dT%H:%M"),
                        "models": kwargs["model"],
                    },
                    "response": {"fake": True},
                }
            ],
            issued_at=kwargs["issued_at"],
        )

    monkeypatch.setattr("fetch_qilupulse96_weather_snapshot.fetch_weather_single_run_snapshot_at", fake_fetch)

    result = fetch_and_store_snapshot(
        tmp_path,
        target_date=target,
        issued_at=issued,
        model_run=model_run,
        model="ecmwf_ifs",
        sleep=lambda _seconds: None,
    )

    assert result["station_count"] == len(SHANDONG_SPATIAL_STATIONS)
    assert result["model_run"] == "2026-08-21T18:00:00+00:00"
    assert len(result["snapshot_paths"]) == 16
    for path_text in result["snapshot_paths"]:
        payload = json.loads((tmp_path / path_text).read_text(encoding="utf-8"))
        assert payload["forecast_issued_at"] == "2026-08-22T12:00:00+08:00"
        assert len(payload["weather"]) == 24
        assert payload["payloads"][0]["request"]["run"] == "2026-08-21T18:00"
        assert payload["payloads"][0]["request"]["models"] == "ecmwf_ifs"
