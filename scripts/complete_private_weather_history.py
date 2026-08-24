"""Complete an ignored runtime's historical weather cache from Open-Meteo Archive.

This is for observed/history rows needed by the causal context.  It is not a
target-day forecast collector and never writes to the public source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.spatial_weather_v01 import (
    disaggregate_station_panel_to_quarters,
    fetch_observed_spatial_weather,
    validate_station_weather,
)


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)
    return stamp.normalize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_private_history(
    runtime_root: Path | str,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    fetcher=fetch_observed_spatial_weather,
) -> dict[str, object]:
    """Fetch observed hourly weather, expand it, and merge it into history cache."""
    runtime = Path(runtime_root).expanduser().resolve()
    start = _local_day(start_date)
    end = _local_day(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")

    print(f"Fetching observed weather: {start.date()} to {end.date()} (exclusive)", flush=True)
    hourly = fetcher(start, end, cache_dir=runtime / "data" / "raw")
    quarters = disaggregate_station_panel_to_quarters(hourly)
    validate_station_weather(quarters)

    cache = ParquetCache(runtime / "data" / "raw")
    paths: list[Path] = []
    for station in SHANDONG_SPATIAL_STATIONS:
        frame = quarters[station.code]
        cache.merge("weather_history_v1", station.code, "weather", frame)
        paths.append(cache._path("weather_history_v1", station.code, "weather"))

    manifest_root = runtime / "data" / "raw" / "weather_history_completion"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"weather_history_{start:%Y%m%d}_{end:%Y%m%d}.json"
    relative_paths = [path.relative_to(runtime).as_posix() for path in paths]
    manifest = {
        "status": "ready",
        "source": "openmeteo_archive_api",
        "start_date": start.date().isoformat(),
        "end_date_exclusive": end.date().isoformat(),
        "station_count": len(paths),
        "cache_paths": relative_paths,
        "cache_sha256": {path: _sha256(runtime / path) for path in relative_paths},
        "fetched_at": pd.Timestamp.now(tz=TIMEZONE).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ready",
        "station_count": len(paths),
        "start_date": start.date().isoformat(),
        "end_date_exclusive": end.date().isoformat(),
        "manifest_path": manifest_path.relative_to(runtime).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True, help="Exclusive Shanghai date")
    args = parser.parse_args()
    result = complete_private_history(
        args.runtime_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
