from __future__ import annotations

import json

import pandas as pd
import pytest

from da_forecast.sources.openmeteo import (
    SINGLE_RUNS_URL,
    fetch_weather_forecast_snapshot_at,
    fetch_weather_single_run_snapshot_at,
)
from da_forecast.sources.weather_provenance import ForecastSnapshotArchive


TZ = "Asia/Shanghai"


def _response():
    return {
        "hourly": {
            "time": ["2026-08-16T00:00", "2026-08-16T01:00"],
            "temperature_2m": [25.0, 24.0],
            "wind_speed_10m": [2.0, 3.0],
            "wind_speed_100m": [6.0, 7.0],
            "direct_radiation": [0.0, 0.0],
            "diffuse_radiation": [0.0, 0.0],
        }
    }


def test_forecast_snapshot_keeps_raw_response_and_parsed_weather(monkeypatch):
    monkeypatch.setattr("da_forecast.sources.openmeteo._request_with_retry", lambda params, url: _response())

    snapshot = fetch_weather_forecast_snapshot_at(
        36.65, 117.0, pd.Timestamp("2026-08-16", tz=TZ), pd.Timestamp("2026-08-17", tz=TZ)
    )

    assert snapshot.weather_kind == "forecast"
    assert snapshot.weather.index[0] == pd.Timestamp("2026-08-15 16:00", tz="UTC")
    assert snapshot.payloads[0]["response"] == _response()
    assert snapshot.payloads[0]["request"]["timezone"] == TZ


def test_forecast_snapshot_archive_is_replayable(tmp_path):
    archive = ForecastSnapshotArchive(tmp_path)
    weather = pd.DataFrame(
        {"temperature_2m": [25.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 16:00", tz="UTC")])
    )
    raw = [{"request": {"latitude": 36.65}, "response": _response()}]

    path = archive.store("SD_JINAN", weather, raw, issued_at="2026-08-15 12:00+08:00")
    restored = archive.load(path)

    assert path.suffix == ".json"
    assert restored["weather_kind"] == "forecast"
    assert restored["station_code"] == "SD_JINAN"
    assert restored["weather"][0]["temperature_2m"] == 25.0
    assert restored["payloads"] == raw


def _single_run_response(hours: int = 72):
    times = pd.date_range("2026-08-22 02:00", periods=hours, freq="h")
    return {
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": [25.0] * hours,
            "wind_speed_10m": [2.0] * hours,
        }
    }


def test_single_run_snapshot_keeps_model_run_and_target_day_only(monkeypatch):
    calls: list[tuple[dict, str]] = []
    response = _single_run_response()

    def request(params, url):
        calls.append((params, url))
        return response

    monkeypatch.setattr("da_forecast.sources.openmeteo._request_with_retry", request)

    snapshot = fetch_weather_single_run_snapshot_at(
        36.65,
        117.0,
        target_date=pd.Timestamp("2026-08-23", tz=TZ),
        model_run=pd.Timestamp("2026-08-21T18:00:00Z"),
        issued_at=pd.Timestamp("2026-08-22T12:00:00+08:00"),
        model="ecmwf_ifs",
        variables=("temperature_2m", "wind_speed_10m"),
    )

    assert calls[0][1] == SINGLE_RUNS_URL
    assert calls[0][0]["run"] == "2026-08-21T18:00"
    assert calls[0][0]["models"] == "ecmwf_ifs"
    assert "start_date" not in calls[0][0]
    assert "end_date" not in calls[0][0]
    assert len(snapshot.weather) == 24
    assert snapshot.weather.index[0] == pd.Timestamp("2026-08-22 16:00", tz="UTC")
    assert snapshot.weather.index[-1] == pd.Timestamp("2026-08-23 15:00", tz="UTC")
    assert snapshot.issued_at == pd.Timestamp("2026-08-22T12:00:00+08:00")
    assert snapshot.payloads[0]["request"]["run"] == "2026-08-21T18:00"
    assert snapshot.payloads[0]["response"] == response


def test_single_run_snapshot_rejects_incomplete_target_day(monkeypatch):
    monkeypatch.setattr(
        "da_forecast.sources.openmeteo._request_with_retry",
        lambda params, url: _single_run_response(hours=30),
    )

    with pytest.raises(ValueError, match="complete 24-hour target-day coverage"):
        fetch_weather_single_run_snapshot_at(
            36.65,
            117.0,
            target_date=pd.Timestamp("2026-08-23", tz=TZ),
            model_run=pd.Timestamp("2026-08-21T18:00:00Z"),
            issued_at=pd.Timestamp("2026-08-22T12:00:00+08:00"),
            model="ecmwf_ifs",
            variables=("temperature_2m", "wind_speed_10m"),
        )
