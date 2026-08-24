"""Forecast acquisition and snapshot provenance for production runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
from da_forecast.sources.spatial_weather_v01 import (
    disaggregate_hourly_weather_to_quarters,
    disaggregate_station_panel_to_quarters,
    fetch_forecast_spatial_weather,
    fetch_observed_spatial_weather,
    validate_station_weather,
)
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.weather_provenance import ForecastSnapshotArchive


@dataclass(frozen=True)
class ForecastWeatherRun:
    panel: dict[str, pd.DataFrame]
    snapshot_paths: tuple[Path, ...]
    snapshot_hash: str
    fetched_at: str


@dataclass(frozen=True)
class WeatherCompletionResult:
    history_panel: dict[str, pd.DataFrame]
    target_panel: dict[str, pd.DataFrame]
    history_start: str
    history_end: str
    target_start: str
    target_end: str
    source_ranges: dict[str, list[dict[str, object]]]
    source_counts: dict[str, int]
    used_forecast_backfill: bool
    source_hash: str
    merged_panel_hash: str
    manifest_path: Path
    target_forecast: ForecastWeatherRun


class WeatherRuntimeV1:
    """Fetch target-day 16-city forecasts and archive every raw response."""

    def __init__(self, root: str | Path, *, weather_source: str = "fetch") -> None:
        if weather_source not in {"fetch", "existing"}:
            raise ValueError("weather_source must be either 'fetch' or 'existing'")
        self.root = Path(root)
        self.weather_source = weather_source
        self.archive = ForecastSnapshotArchive(self.root / "data" / "raw" / "weather_forecasts")
        self.history_cache = ParquetCache(self.root / "data" / "raw")
        self.manifest_root = self.root / "data" / "raw" / "weather_completion"

    def fetch_target_forecast(self, *, target_date: str | pd.Timestamp, as_of: str | pd.Timestamp) -> ForecastWeatherRun:
        target = _local_day(target_date)
        issued = _local_timestamp(as_of)
        expected_as_of = target - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        if issued != expected_as_of:
            raise ValueError("Target forecast must be acquired exactly at the T-1 12:00 contract time")
        if self.weather_source == "existing":
            hourly, paths = self._load_existing_target_forecast(issued)
        else:
            hourly, paths = fetch_forecast_spatial_weather(
                target,
                target + pd.Timedelta(days=1),
                archive=self.archive,
                issued_at=issued,
            )
        quarters = disaggregate_station_panel_to_quarters(hourly)
        expected = pd.date_range(target, periods=96, freq="15min", tz=TIMEZONE)
        validate_station_weather(quarters)
        for code in (station.code for station in SHANDONG_SPATIAL_STATIONS):
            if not quarters[code].index.equals(expected):
                raise ValueError(f"Target forecast lacks a complete 96-slot panel for {code}")
        resolved = tuple(Path(path) for path in paths)
        digest = hashlib.sha256()
        for path in sorted(resolved):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return ForecastWeatherRun(quarters, resolved, digest.hexdigest(), issued.isoformat())

    def _existing_snapshot_paths(self, issued: pd.Timestamp) -> dict[str, Path]:
        safe_issued = issued.strftime("%Y%m%dT%H%M%S%z")
        return {
            station.code: self.archive.root / station.code / f"{safe_issued}.json"
            for station in SHANDONG_SPATIAL_STATIONS
        }

    def _load_existing_target_forecast(
        self, issued: pd.Timestamp
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Load one exact archived issue without making any network request."""
        paths = self._existing_snapshot_paths(issued)
        missing = [code for code, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError(
                "target forecast snapshot missing for the exact as-of contract "
                f"{issued.isoformat()}; missing_stations={','.join(missing)}"
            )

        result: dict[str, pd.DataFrame] = {}
        for code, path in paths.items():
            payload = ForecastSnapshotArchive.load(path)
            recorded = payload.get("forecast_issued_at")
            if recorded is None or _local_timestamp(recorded) != issued:
                raise ValueError(
                    "Target forecast snapshot issued_at mismatch for "
                    f"{code}: expected {issued.isoformat()}, got {recorded!r}"
                )
            records = payload.get("weather")
            if not isinstance(records, list) or not records:
                raise ValueError(f"Target forecast snapshot has no weather rows for {code}: {path}")
            frame = pd.DataFrame(records)
            if "timestamp" not in frame.columns:
                raise ValueError(f"Target forecast snapshot has no timestamp column for {code}: {path}")
            index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp")))
            index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
            frame.index = index
            frame.index.name = "timestamp"
            result[code] = frame.sort_index()
        return result, [str(path) for path in paths.values()]

    def load_history_observed(self) -> dict[str, pd.DataFrame]:
        """Load the bootstrap observed panel; it is valid only for already past history."""
        candidates = [
            self.root / "data" / "raw" / "weather_history_v1",
            self.root / "data" / "bootstrap" / "raw" / "openmeteo_spatial_v01_quarter",
            self.root / "data" / "bootstrap" / "raw" / "openmeteo_detailed",
            self.root / "data" / "raw" / "openmeteo_spatial_v01_quarter",
            self.root / "data" / "raw" / "openmeteo_detailed",
        ]
        source = next((path for path in candidates if path.is_dir()), None)
        if source is None:
            # An empty seed is valid here: ensure_weather_to_target will request
            # the complete required range from the Archive API.
            return {}
        result: dict[str, pd.DataFrame] = {}
        for station in SHANDONG_SPATIAL_STATIONS:
            path = source / station.code / "weather.parquet"
            if not path.is_file():
                # Keep a partial cache usable.  The missing cities are fetched
                # below, while existing local observations retain priority.
                continue
            result[station.code] = pd.read_parquet(path)
        if not result:
            return {}
        result = _as_quarter_panel(result)
        if len(result) == len(SHANDONG_SPATIAL_STATIONS):
            validate_station_weather(result)
        return result

    def ensure_weather_to_target(
        self,
        *,
        target_date: str | pd.Timestamp,
        as_of: str | pd.Timestamp,
        progress: Callable[[str], None] | None = None,
    ) -> WeatherCompletionResult:
        """Complete history and target weather independently of market prices."""
        target = _local_day(target_date)
        issued = _local_timestamp(as_of)
        expected_as_of = target - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        if issued != expected_as_of:
            raise ValueError("Weather completion requires the T-1 12:00 Asia/Shanghai contract time")

        history_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
        history_start = history_end - pd.Timedelta(minutes=15 * (90 * 96 - 1))
        history_index = pd.date_range(history_start, history_end, freq="15min", tz=TIMEZONE)
        target_index = pd.date_range(target, periods=96, freq="15min", tz=TIMEZONE)
        station_codes = [station.code for station in SHANDONG_SPATIAL_STATIONS]
        source_ranges: dict[str, list[dict[str, object]]] = {code: [] for code in station_codes}

        if self.weather_source == "existing":
            missing_snapshots = [
                code for code, path in self._existing_snapshot_paths(issued).items() if not path.is_file()
            ]
            if missing_snapshots:
                error = (
                    "target forecast snapshot missing for the exact as-of contract "
                    f"{issued.isoformat()}; missing_stations={','.join(missing_snapshots)}"
                )
                manifest_path = self._write_failure_manifest(
                    target=target,
                    issued=issued,
                    history_start=history_start,
                    history_end=history_end,
                    target_index=target_index,
                    source_ranges=source_ranges,
                    incomplete={code: target_index for code in missing_snapshots},
                    error=error,
                )
                raise ValueError(f"{error}; manifest={manifest_path}")

        _report(progress, "检查历史天气覆盖")
        try:
            seed = _normalise_panel(self.load_history_observed())
        except (FileNotFoundError, ValueError) as exc:
            seed = {}
            _report(progress, f"本地历史天气不可用，改为请求 Archive：{type(exc).__name__}: {exc}")
        history = {code: seed[code].copy() for code in seed if code in station_codes}
        observed_panel: dict[str, pd.DataFrame] = {}
        forecast_panel: dict[str, pd.DataFrame] = {}
        for code, frame in history.items():
            source_ranges[code].extend(_source_records("observed", frame.reindex(history_index).dropna(how="all")))

        missing = _missing_panel_index(history, history_index)
        observed_error: str | None = None
        if self.weather_source == "existing" and any(len(index) for index in missing.values()):
            detail = "; ".join(
                f"{code}: {len(index)} slots from {index.min().isoformat()} to {index.max().isoformat()}"
                for code, index in missing.items() if len(index)
            )
            error = f"Existing weather history is incomplete: {detail}"
            manifest_path = self._write_failure_manifest(
                target=target,
                issued=issued,
                history_start=history_start,
                history_end=history_end,
                target_index=target_index,
                source_ranges=source_ranges,
                incomplete=missing,
                error=error,
            )
            raise ValueError(f"{error}; manifest={manifest_path}")
        if self.weather_source != "existing" and any(len(index) for index in missing.values()):
            observed_start = min(index.min() for index in missing.values() if len(index)).normalize()
            observed_end = max(index.max() for index in missing.values() if len(index)).normalize() + pd.Timedelta(days=1)
            _report(progress, f"补取历史真实天气：{observed_start.date()} 至 {observed_end.date()}")
            try:
                observed_hourly = fetch_observed_spatial_weather(
                    observed_start,
                    observed_end,
                    cache_dir=self.root / "data" / "raw",
                )
                observed_panel = _as_quarter_panel(observed_hourly)
            except Exception as exc:
                observed_error = f"{type(exc).__name__}: {exc}"
                observed_panel = {}
                _report(progress, f"历史真实天气补取失败，继续尝试 Forecast：{observed_error}")
            for code, frame in observed_panel.items():
                existing = history.get(code, pd.DataFrame())
                history[code] = existing.combine_first(frame).sort_index()

        missing = _missing_panel_index(history, history_index)
        forecast_paths: list[str] = []
        forecast_fetched_at: str | None = None
        used_forecast_backfill = False
        forecast_error: str | None = None
        if self.weather_source != "existing" and any(len(index) for index in missing.values()):
            forecast_start = min(index.min() for index in missing.values() if len(index)).normalize()
            forecast_end = max(index.max() for index in missing.values() if len(index)).normalize() + pd.Timedelta(days=1)
            _report(progress, f"Forecast 回填历史天气：{forecast_start.date()} 至 {forecast_end.date()}")
            try:
                forecast_hourly, forecast_paths = fetch_forecast_spatial_weather(
                    forecast_start,
                    forecast_end,
                    archive=self.archive,
                    issued_at=issued,
                )
                forecast_panel = _as_quarter_panel(forecast_hourly)
                forecast_fetched_at = pd.Timestamp.now(tz=TIMEZONE).isoformat()
            except Exception as exc:
                forecast_error = f"{type(exc).__name__}: {exc}"
                forecast_panel = {}
                _report(progress, f"Forecast 回填失败：{forecast_error}")
            for code, frame in forecast_panel.items():
                existing = history.get(code, pd.DataFrame())
                history[code] = existing.combine_first(frame).sort_index()
            used_forecast_backfill = bool(forecast_panel)

        remaining = _missing_panel_index(history, history_index)
        if any(len(index) for index in remaining.values()):
            detail = "; ".join(
                f"{code}: {len(index)} slots from {index.min().isoformat()} to {index.max().isoformat()}"
                for code, index in remaining.items() if len(index)
            )
            errors = "; ".join(error for error in (observed_error, forecast_error) if error)
            suffix = f"; source_errors={errors}" if errors else ""
            manifest_path = self._write_failure_manifest(
                target=target,
                issued=issued,
                history_start=history_start,
                history_end=history_end,
                target_index=target_index,
                source_ranges=source_ranges,
                incomplete=remaining,
                error=f"Unable to complete historical weather panel: {detail}{suffix}",
            )
            raise ValueError(f"Unable to complete historical weather panel: {detail}{suffix}; manifest={manifest_path}")

        history = {code: frame.reindex(history_index).copy() for code, frame in history.items()}
        # Record only the rows actually used by the required history panel.  A
        # cache may contain years of data, but those rows are not part of this
        # run's completion counts or audit range.
        for code, frame in history.items():
            source_ranges[code].clear()
            original = seed.get(code, pd.DataFrame(index=history_index)).reindex(history_index)
            observed_rows = _complete_rows(original, history_index)
            source_ranges[code].extend(_source_records("observed", frame.loc[observed_rows], source="local_observed_cache"))
            if code in observed_panel:
                archive_rows = observed_panel[code].reindex(history_index)
                archive_rows = _complete_rows(archive_rows, history_index) & ~observed_rows
                source_ranges[code].extend(_source_records("observed", frame.loc[archive_rows], source="openmeteo_archive"))
            if code in forecast_panel:
                forecast_rows = forecast_panel[code].reindex(history_index)
                forecast_rows = _complete_rows(forecast_rows, history_index) & ~observed_rows
                if code in observed_panel:
                    archive_rows = observed_panel[code].reindex(history_index)
                    forecast_rows &= ~_complete_rows(archive_rows, history_index)
                source_ranges[code].extend(_source_records(
                    "forecast_backfill",
                    frame.loc[forecast_rows],
                    source="forecast_api",
                    issued_at=issued,
                    fetched_at=forecast_fetched_at,
                    paths=forecast_paths,
                ))
        validate_station_weather(history)
        _report(progress, "保存完整历史天气缓存")
        history_paths: list[Path] = []
        for code, frame in history.items():
            path = self.history_cache._path("weather_history_v1", code, "weather")
            # Keep any wider private cache already staged for calibration or
            # later replay.  Saving only this run's causal window would
            # silently trim valid history and make the next calibration run
            # fail at an earlier replay date.
            self.history_cache.merge("weather_history_v1", code, "weather", frame)
            history_paths.append(path)

        _report(progress, "抓取目标日 Forecast")
        try:
            target_forecast = self.fetch_target_forecast(target_date=target, as_of=issued)
        except Exception as exc:
            incomplete = {code: target_index for code in station_codes}
            manifest_path = self._write_failure_manifest(
                target=target,
                issued=issued,
                history_start=history_start,
                history_end=history_end,
                target_index=target_index,
                source_ranges=source_ranges,
                incomplete=incomplete,
                error=f"Target forecast incomplete: {type(exc).__name__}: {exc}",
            )
            raise type(exc)(f"{exc}; manifest={manifest_path}") from exc
        target_panel = {code: frame.reindex(target_index).copy() for code, frame in target_forecast.panel.items()}
        try:
            validate_station_weather(target_panel)
        except Exception as exc:
            incomplete = {
                code: target_index
                for code in station_codes
                if code not in target_panel or not target_panel[code].index.equals(target_index)
            }
            if not incomplete:
                incomplete = {code: target_index for code in station_codes}
            manifest_path = self._write_failure_manifest(
                target=target,
                issued=issued,
                history_start=history_start,
                history_end=history_end,
                target_index=target_index,
                source_ranges=source_ranges,
                incomplete=incomplete,
                error=f"Target forecast incomplete: {type(exc).__name__}: {exc}",
            )
            raise type(exc)(f"{exc}; manifest={manifest_path}") from exc
        for code, frame in target_panel.items():
            if frame[list(STATION_COLUMNS)].isna().any().any() or not frame.index.equals(target_index):
                incomplete = {code: target_index}
                manifest_path = self._write_failure_manifest(
                    target=target,
                    issued=issued,
                    history_start=history_start,
                    history_end=history_end,
                    target_index=target_index,
                    source_ranges=source_ranges,
                    incomplete=incomplete,
                    error=f"Target weather is incomplete for station {code}",
                )
                raise ValueError(f"Target weather is incomplete for station {code}; manifest={manifest_path}")
            source_ranges[code].extend(_source_records(
                "target_forecast",
                frame,
                issued_at=issued,
                fetched_at=target_forecast.fetched_at,
                paths=[str(path) for path in target_forecast.snapshot_paths],
            ))

        merged_panel_hash = _panel_hash(
            {f"history/{code}": frame for code, frame in history.items()}
            | {f"target/{code}": frame for code, frame in target_panel.items()}
        )

        source_counts = {
            kind: sum(int(item.get("rows", 0)) for records in source_ranges.values() for item in records if item.get("kind") == kind)
            for kind in ("observed", "forecast_backfill", "target_forecast")
        }
        manifest = {
            "schema_version": "weather_completion_v1",
            "status": "complete",
            "target_date": target.strftime("%Y-%m-%d"),
            "as_of": issued.isoformat(),
            "history": {"start": history_start.isoformat(), "end": history_end.isoformat()},
            "target": {"start": target.isoformat(), "end": target_index[-1].isoformat()},
            "fetch_time": target_forecast.fetched_at,
            "source_ranges": source_ranges,
            "source_counts": source_counts,
            "used_forecast_backfill": used_forecast_backfill,
            "target_forecast_hash": target_forecast.snapshot_hash,
            "target_forecast_snapshots": [str(path) for path in target_forecast.snapshot_paths],
            "history_cache_paths": [str(path) for path in history_paths],
            "forecast_backfill_snapshots": forecast_paths,
            "merged_panel_hash": merged_panel_hash,
        }
        source_hash = _source_hash(manifest, history_paths, target_forecast.snapshot_paths)
        manifest["source_hash"] = source_hash
        manifest_path = self._manifest_path(target, issued)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
        _report(progress, f"天气已补齐：历史至 {history_end.isoformat()}，目标日至 {target_index[-1].isoformat()}，manifest={manifest_path}")
        return WeatherCompletionResult(
            history_panel=history,
            target_panel=target_panel,
            history_start=history_start.isoformat(),
            history_end=history_end.isoformat(),
            target_start=target.isoformat(),
            target_end=target_index[-1].isoformat(),
            source_ranges=source_ranges,
            source_counts=source_counts,
            used_forecast_backfill=used_forecast_backfill,
            source_hash=source_hash,
            merged_panel_hash=merged_panel_hash,
            manifest_path=manifest_path,
            target_forecast=target_forecast,
        )

    def _manifest_path(self, target: pd.Timestamp, issued: pd.Timestamp) -> Path:
        return self.manifest_root / f"weather_completion_{target:%Y%m%d}_{issued:%Y%m%dT%H%M%S%z}.json"

    def _write_failure_manifest(
        self,
        *,
        target: pd.Timestamp,
        issued: pd.Timestamp,
        history_start: pd.Timestamp,
        history_end: pd.Timestamp,
        target_index: pd.DatetimeIndex,
        source_ranges: dict[str, list[dict[str, object]]],
        incomplete: dict[str, pd.DatetimeIndex],
        error: str,
    ) -> Path:
        payload: dict[str, object] = {
            "schema_version": "weather_completion_v1",
            "status": "error",
            "target_date": target.strftime("%Y-%m-%d"),
            "as_of": issued.isoformat(),
            "history": {"start": history_start.isoformat(), "end": history_end.isoformat()},
            "target": {"start": target.isoformat(), "end": target_index[-1].isoformat()},
            "source_ranges": source_ranges,
            "incomplete_stations": _coverage_details(incomplete),
            "error": error,
        }
        payload["source_hash"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        path = self._manifest_path(target, issued)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    @staticmethod
    def save_runtime_manifest(path: str | Path, run: ForecastWeatherRun) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "weather_kind": "forecast", "fetched_at": run.fetched_at,
            "snapshot_hash": run.snapshot_hash, "snapshots": [str(item) for item in run.snapshot_paths],
        }, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    return _local_timestamp(value).normalize()


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _normalise_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for code, frame in panel.items():
        copy = frame.copy()
        index = pd.DatetimeIndex(copy.index)
        copy.index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
        result[code] = copy.sort_index()
    return result


def _as_quarter_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalised = _normalise_panel(panel)
    if not normalised:
        return normalised
    quarter_flags = {code: _is_quarter_index(frame.index) for code, frame in normalised.items()}
    if all(quarter_flags.values()):
        return normalised
    # The common-panel helper validates all 16 cities.  Partial local caches
    # need per-city expansion so that the present cities remain usable while
    # Archive fills only the missing ranges/cities.
    if len(normalised) == len(SHANDONG_SPATIAL_STATIONS) and not any(quarter_flags.values()):
        return disaggregate_station_panel_to_quarters(normalised)
    stations = {station.code: station for station in SHANDONG_SPATIAL_STATIONS}
    return {
        code: frame if quarter_flags[code] else disaggregate_hourly_weather_to_quarters(
            frame,
            latitude=stations[code].latitude,
            longitude=stations[code].longitude,
            altitude=stations[code].altitude_m,
        )
        for code, frame in normalised.items()
    }


def _is_quarter_index(index: pd.DatetimeIndex) -> bool:
    return len(index) < 2 or bool((index[1] - index[0]) == pd.Timedelta(minutes=15))


def _missing_panel_index(panel: dict[str, pd.DataFrame], required: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    result: dict[str, pd.DatetimeIndex] = {}
    for station in SHANDONG_SPATIAL_STATIONS:
        code = station.code
        frame = panel.get(code, pd.DataFrame(index=required))
        values = frame.reindex(required)
        if set(STATION_COLUMNS).issubset(values.columns):
            missing = values[list(STATION_COLUMNS)].isna().any(axis=1)
        else:
            missing = pd.Series(True, index=required)
        result[code] = required[missing.to_numpy()]
    return result


def _complete_rows(frame: pd.DataFrame, required: pd.DatetimeIndex) -> pd.Series:
    values = frame.reindex(required)
    if not set(STATION_COLUMNS).issubset(values.columns):
        return pd.Series(False, index=required)
    return values[list(STATION_COLUMNS)].notna().all(axis=1)


def _coverage_details(incomplete: dict[str, pd.DatetimeIndex]) -> dict[str, dict[str, object]]:
    return {
        code: {
            "slots": int(len(index)),
            "start": index.min().isoformat() if len(index) else None,
            "end": index.max().isoformat() if len(index) else None,
        }
        for code, index in incomplete.items()
        if len(index)
    }


def _source_records(
    kind: str,
    frame: pd.DataFrame,
    *,
    source: str = "local_observed_cache",
    issued_at: pd.Timestamp | None = None,
    fetched_at: str | pd.Timestamp | None = None,
    paths: list[str] | None = None,
) -> list[dict[str, object]]:
    if frame.empty:
        return []
    index = pd.DatetimeIndex(frame.index).sort_values()
    # Weather panels are quarter-hourly.  Keep separate audit ranges whenever
    # a cache has an internal gap instead of hiding it inside one min/max span.
    breaks = index.to_series().diff().fillna(pd.Timedelta(minutes=15)).ne(pd.Timedelta(minutes=15))
    group_ids = breaks.cumsum().to_numpy()
    records: list[dict[str, object]] = []
    for group_id in pd.unique(group_ids):
        run_index = index[group_ids == group_id]
        record: dict[str, object] = {
            "kind": kind,
            "source": source,
            "start": run_index.min().isoformat(),
            "end": run_index.max().isoformat(),
            "rows": int(len(run_index)),
            "raw_response_paths": [str(path) for path in (paths or [])],
            "raw_response_hashes": [_file_hash(Path(path)) for path in (paths or [])],
        }
        if issued_at is not None:
            record["issued_at"] = issued_at.isoformat()
        if fetched_at is not None:
            record["fetched_at"] = pd.Timestamp(fetched_at).isoformat()
        records.append(record)
    return records


def _source_hash(manifest: dict[str, object], history_paths: list[Path], target_paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    for path in [*history_paths, *target_paths]:
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _panel_hash(panel: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for key in sorted(panel):
        frame = panel[key].sort_index()
        digest.update(key.encode("utf-8"))
        digest.update("\x1f".join(str(column) for column in frame.columns).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
