from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from da_forecast.config import SHANDONG_SPATIAL_STATIONS
from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
from da_forecast.production.weather_runtime_v1 import WeatherRuntimeV1
from da_forecast.sources.cache import ParquetCache


TZ = "Asia/Shanghai"


def _panel(index: pd.DatetimeIndex, value: float) -> dict[str, pd.DataFrame]:
    return {
        station.code: pd.DataFrame(
            {column: value for column in STATION_COLUMNS},
            index=index,
        )
        for station in SHANDONG_SPATIAL_STATIONS
    }


def test_ensure_weather_to_target_merges_observed_gap_and_writes_manifest(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    observed_index = pd.date_range(required_start, pd.Timestamp("2026-04-12 23:45", tz=TZ), freq="15min")
    fetched_index = pd.date_range(pd.Timestamp("2026-04-13 00:00", tz=TZ), required_end, freq="15min")
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    runtime = WeatherRuntimeV1(tmp_path)
    monkeypatch.setattr(runtime, "load_history_observed", lambda: _panel(observed_index, 1.0))
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_observed_spatial_weather",
        lambda *_args, **_kwargs: _panel(fetched_index, 2.0),
    )
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.disaggregate_station_panel_to_quarters",
        lambda panel: panel,
    )
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:00:00+08:00",
        ),
    )

    result = runtime.ensure_weather_to_target(
        target_date="2026-04-14",
        as_of="2026-04-13T12:00:00+08:00",
    )

    assert result.history_panel["SD_JINAN"].index.equals(pd.date_range(required_start, required_end, freq="15min", tz=TZ))
    assert result.history_panel["SD_JINAN"].loc[pd.Timestamp("2026-04-12 23:45", tz=TZ), "temperature_2m"] == 1.0
    assert result.history_panel["SD_JINAN"].loc[pd.Timestamp("2026-04-13 00:00", tz=TZ), "temperature_2m"] == 2.0
    assert result.target_panel["SD_JINAN"].index.equals(target_index)
    assert result.used_forecast_backfill is False
    assert result.manifest_path.is_file()
    assert result.source_hash


def test_weather_completion_preserves_a_wider_existing_history_cache(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    history_index = pd.date_range(required_start, required_end, freq="15min", tz=TZ)
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    older_timestamp = required_start - pd.Timedelta(minutes=15)
    cache = ParquetCache(tmp_path / "data" / "raw")
    for station in SHANDONG_SPATIAL_STATIONS:
        cache.save(
            "weather_history_v1",
            station.code,
            "weather",
            pd.DataFrame({column: [9.0] for column in STATION_COLUMNS}, index=pd.DatetimeIndex([older_timestamp])),
        )

    runtime = WeatherRuntimeV1(tmp_path)
    monkeypatch.setattr(runtime, "load_history_observed", lambda: _panel(history_index, 1.0))
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.disaggregate_station_panel_to_quarters",
        lambda panel: panel,
    )
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:00:00+08:00",
        ),
    )

    runtime.ensure_weather_to_target(
        target_date="2026-04-14",
        as_of="2026-04-13T12:00:00+08:00",
    )

    restored = cache.load("weather_history_v1", "SD_JINAN", "weather")
    assert restored is not None
    assert restored.loc[older_timestamp, "temperature_2m"] == 9.0


def test_existing_weather_source_blocks_without_exact_target_snapshot(tmp_path, monkeypatch):
    runtime = WeatherRuntimeV1(tmp_path, weather_source="existing")
    monkeypatch.setattr(runtime, "load_history_observed", lambda: {})
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_observed_spatial_weather",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network must not be used")),
    )

    with pytest.raises(ValueError, match="target forecast snapshot"):
        runtime.ensure_weather_to_target(
            target_date="2026-08-23",
            as_of="2026-08-22T12:00:00+08:00",
        )


def test_existing_weather_source_rejects_snapshot_issue_time_mismatch(tmp_path):
    runtime = WeatherRuntimeV1(tmp_path, weather_source="existing")
    issued = pd.Timestamp("2026-08-22 12:00", tz=TZ)
    for path in runtime._existing_snapshot_paths(issued).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "forecast_issued_at": "2026-08-21T12:00:00+08:00",
                    "weather": [],
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="issued_at mismatch"):
        runtime.fetch_target_forecast(
            target_date="2026-08-23",
            as_of="2026-08-22T12:00:00+08:00",
        )


def test_ensure_weather_to_target_uses_forecast_for_remaining_history_gap(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    observed_index = pd.date_range(required_start, pd.Timestamp("2026-04-12 23:45", tz=TZ), freq="15min")
    missing_index = pd.date_range(pd.Timestamp("2026-04-13 00:00", tz=TZ), required_end, freq="15min")
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    runtime = WeatherRuntimeV1(tmp_path)
    monkeypatch.setattr(runtime, "load_history_observed", lambda: _panel(observed_index, 1.0))
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_observed_spatial_weather",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("archive unavailable")),
    )
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_forecast_spatial_weather",
        lambda *_args, **_kwargs: (_panel(missing_index, 4.0), ["history-forecast.json"]),
    )
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.disaggregate_station_panel_to_quarters",
        lambda panel: panel,
    )
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:00:00+08:00",
        ),
    )

    result = runtime.ensure_weather_to_target(
        target_date="2026-04-14",
        as_of="2026-04-13T12:00:00+08:00",
    )

    assert result.used_forecast_backfill is True
    assert result.history_panel["SD_JINAN"].loc[pd.Timestamp("2026-04-13 00:00", tz=TZ), "temperature_2m"] == 4.0
    assert any(item["kind"] == "forecast_backfill" for item in result.source_ranges["SD_JINAN"])


def test_weather_manifest_keeps_fetch_and_raw_response_provenance(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    history_index = pd.date_range(required_start, required_end, freq="15min", tz=TZ)
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    raw_path = tmp_path / "raw-target.json"
    raw_path.write_text('{"raw": true}', encoding="utf-8")
    runtime = WeatherRuntimeV1(tmp_path)
    monkeypatch.setattr(runtime, "load_history_observed", lambda: _panel(history_index, 1.0))
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(raw_path,),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:01:00+08:00",
        ),
    )

    result = runtime.ensure_weather_to_target(
        target_date="2026-04-14",
        as_of="2026-04-13T12:00:00+08:00",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    target_record = next(item for item in manifest["source_ranges"]["SD_JINAN"] if item["kind"] == "target_forecast")
    assert target_record["fetched_at"] == "2026-04-13T12:01:00+08:00"
    assert target_record["raw_response_paths"] == [str(raw_path)]
    assert len(target_record["raw_response_hashes"]) == 1
    assert manifest["merged_panel_hash"]


def test_ensure_weather_to_target_fetches_archive_when_local_history_is_absent(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    history_index = pd.date_range(required_start, required_end, freq="15min", tz=TZ)
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    runtime = WeatherRuntimeV1(tmp_path)
    monkeypatch.setattr(
        runtime,
        "load_history_observed",
        lambda: (_ for _ in ()).throw(FileNotFoundError("no local history")),
    )
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch_archive(start, end, **_kwargs):
        calls.append((start, end))
        return _panel(history_index, 2.0)

    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_observed_spatial_weather",
        fetch_archive,
    )
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.disaggregate_station_panel_to_quarters",
        lambda panel: panel,
    )
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:00:00+08:00",
        ),
    )

    result = runtime.ensure_weather_to_target(
        target_date="2026-04-14",
        as_of="2026-04-13T12:00:00+08:00",
    )

    assert calls
    assert result.history_panel["SD_JINAN"].index.equals(history_index)
    assert result.history_panel["SD_JINAN"]["temperature_2m"].eq(2.0).all()


def test_load_history_observed_expands_hourly_cache_to_quarters(tmp_path, monkeypatch):
    runtime = WeatherRuntimeV1(tmp_path)
    hourly_index = pd.date_range("2026-04-01", periods=4, freq="h", tz=TZ)
    hourly = {
        station.code: pd.DataFrame({column: 1.0 for column in STATION_COLUMNS[:18]}, index=hourly_index)
        for station in SHANDONG_SPATIAL_STATIONS
    }
    source = tmp_path / "data" / "raw" / "openmeteo_detailed"
    for code, frame in hourly.items():
        path = source / code / "weather.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    result = runtime.load_history_observed()

    assert result["SD_JINAN"].index.equals(pd.date_range("2026-04-01", periods=16, freq="15min", tz=TZ))
    assert set(STATION_COLUMNS).issubset(result["SD_JINAN"].columns)


def test_source_records_preserve_discontinuous_ranges():
    from da_forecast.production.weather_runtime_v1 import _source_records

    index = pd.DatetimeIndex([
        pd.Timestamp("2026-04-01 00:00", tz=TZ),
        pd.Timestamp("2026-04-01 00:15", tz=TZ),
        pd.Timestamp("2026-04-01 01:00", tz=TZ),
    ])
    frame = pd.DataFrame({"temperature_2m": [1.0, 1.0, 1.0]}, index=index)

    records = _source_records("observed", frame)

    assert [(item["start"], item["end"], item["rows"]) for item in records] == [
        ("2026-04-01T00:00:00+08:00", "2026-04-01T00:15:00+08:00", 2),
        ("2026-04-01T01:00:00+08:00", "2026-04-01T01:00:00+08:00", 1),
    ]


def test_incomplete_history_writes_failure_manifest_with_city_ranges(tmp_path, monkeypatch):
    target = pd.Timestamp("2026-04-14", tz=TZ)
    required_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    required_start = required_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
    history_index = pd.date_range(required_start, required_end, freq="15min", tz=TZ)
    target_index = pd.date_range(target, periods=96, freq="15min", tz=TZ)
    runtime = WeatherRuntimeV1(tmp_path)
    incomplete = _panel(history_index, 1.0)
    incomplete["SD_JINAN"] = incomplete["SD_JINAN"].iloc[:-1]
    monkeypatch.setattr(runtime, "load_history_observed", lambda: incomplete)
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_observed_spatial_weather",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "da_forecast.production.weather_runtime_v1.fetch_forecast_spatial_weather",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        runtime,
        "fetch_target_forecast",
        lambda **_kwargs: SimpleNamespace(
            panel=_panel(target_index, 3.0),
            snapshot_paths=(),
            snapshot_hash="target-hash",
            fetched_at="2026-04-13T12:00:00+08:00",
        ),
    )

    with pytest.raises(ValueError, match="SD_JINAN"):
        runtime.ensure_weather_to_target(
            target_date="2026-04-14",
            as_of="2026-04-13T12:00:00+08:00",
        )

    manifests = list((tmp_path / "data" / "raw" / "weather_completion").glob("*.json"))
    assert manifests
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "SD_JINAN" in manifest["incomplete_stations"]
