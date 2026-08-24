"""Acquire one auditable historical weather model run into private runtime.

This command is deliberately separate from the production workflow.  It uses
Open-Meteo's Single Runs API to retrieve a historical model initialisation and
then stores only the requested target day under the ignored runtime root.  The
business ``as-of`` contract and the UTC model-run timestamp are both retained
in the raw request and acquisition manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import pandas as pd

from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
from da_forecast.sources.openmeteo import (
    SINGLE_RUNS_URL,
    fetch_weather_single_run_snapshot_at,
)
from da_forecast.sources.spatial_weather_v01 import DETAIL_WEATHER_COLUMNS, validate_station_weather
from da_forecast.sources.weather_provenance import ForecastSnapshotArchive


MIN_MODEL_OUTPUT_LAG = pd.Timedelta(hours=4)


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(TIMEZONE) if timestamp.tz is None else timestamp.tz_convert(TIMEZONE)


def _model_run_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tz is None else timestamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path(archive: ForecastSnapshotArchive, station_code: str, issued_at: pd.Timestamp) -> Path:
    safe_issued = issued_at.strftime("%Y%m%dT%H%M%S%z")
    return archive.root / station_code / f"{safe_issued}.json"


def _assert_contract(target_date: pd.Timestamp, issued_at: pd.Timestamp, model_run: pd.Timestamp) -> None:
    expected = target_date - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    if issued_at != expected:
        raise ValueError(
            "Weather acquisition requires the T-1 12:00 Asia/Shanghai contract: "
            f"expected={expected.isoformat()}, got={issued_at.isoformat()}"
        )
    if model_run + MIN_MODEL_OUTPUT_LAG > issued_at.tz_convert("UTC"):
        raise ValueError(
            "model_run is too close to the as-of contract for the conservative "
            f"{MIN_MODEL_OUTPUT_LAG.total_seconds() / 3600:.0f}-hour availability assumption"
        )


def _existing_snapshot_matches(
    path: Path,
    *,
    issued_at: pd.Timestamp,
    model_run: pd.Timestamp,
    model: str,
) -> bool:
    if not path.is_file():
        return False
    payload = ForecastSnapshotArchive.load(path)
    if payload.get("forecast_issued_at") != issued_at.isoformat():
        return False
    weather = payload.get("weather")
    payloads = payload.get("payloads")
    request = payloads[0].get("request", {}) if isinstance(payloads, list) and payloads else {}
    return (
        isinstance(weather, list)
        and len(weather) == 24
        and request.get("run") == model_run.strftime("%Y-%m-%dT%H:%M")
        and request.get("models") == model
    )


def fetch_and_store_snapshot(
    runtime_root: Path | str,
    *,
    target_date: str | pd.Timestamp,
    issued_at: str | pd.Timestamp,
    model_run: str | pd.Timestamp,
    model: str = "ecmwf_ifs",
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Fetch and archive a complete 16-city target-day model-run panel."""
    runtime = Path(runtime_root).expanduser().resolve()
    target = _local_timestamp(target_date).normalize()
    issued = _local_timestamp(issued_at)
    run = _model_run_timestamp(model_run)
    _assert_contract(target, issued, run)

    expected_index = pd.date_range(target, periods=24, freq="h", tz=TIMEZONE).tz_convert("UTC")
    archive = ForecastSnapshotArchive(runtime / "data" / "raw" / "weather_forecasts")
    panel: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []

    for position, station in enumerate(SHANDONG_SPATIAL_STATIONS, start=1):
        print(f"[{position}/{len(SHANDONG_SPATIAL_STATIONS)}] {station.code}: historical single run", flush=True)
        path = _snapshot_path(archive, station.code, issued)
        if _existing_snapshot_matches(path, issued_at=issued, model_run=run, model=model):
            payload = ForecastSnapshotArchive.load(path)
            frame = pd.DataFrame(payload["weather"])
            frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
            panel[station.code] = frame.sort_index()
            paths.append(path)
            continue
        if path.exists():
            raise FileExistsError(
                f"Existing snapshot has a different model run or contract; review before replacing: {path}"
            )

        snapshot = fetch_weather_single_run_snapshot_at(
            station.latitude,
            station.longitude,
            target_date=target,
            model_run=run,
            issued_at=issued,
            model=model,
            variables=DETAIL_WEATHER_COLUMNS,
        )
        if not snapshot.weather.index.equals(expected_index):
            raise ValueError(f"{station.code} did not return the exact 24-hour target-day index")
        panel[station.code] = snapshot.weather
        stored = archive.store(
            station.code,
            snapshot.weather,
            snapshot.payloads,
            issued_at=issued,
        )
        if stored != path:
            raise RuntimeError(f"Unexpected snapshot path generated: {stored}")
        paths.append(stored)
        sleep(0.3)

    validate_station_weather(panel)
    relative_paths = [path.relative_to(runtime).as_posix() for path in paths]
    acquisition_root = runtime / "data" / "raw" / "weather_acquisition"
    acquisition_root.mkdir(parents=True, exist_ok=True)
    manifest_path = acquisition_root / (
        f"weather_acquisition_{target.strftime('%Y%m%d')}_{issued.strftime('%Y%m%dT%H%M%S%z')}.json"
    )
    manifest = {
        "status": "ready",
        "target_date": target.date().isoformat(),
        "as_of_contract": issued.isoformat(),
        "model": model,
        "model_run_initialization_utc": run.isoformat(),
        "model_run_parameter": run.strftime("%Y-%m-%dT%H:%M"),
        "endpoint": SINGLE_RUNS_URL,
        "availability_assumption": {
            "minimum_hours_after_initialization": MIN_MODEL_OUTPUT_LAG.total_seconds() / 3600,
            "run_is_before_contract": True,
            "note": "The model run is retained separately from the business as-of timestamp.",
        },
        "station_count": len(paths),
        "snapshot_paths": relative_paths,
        "snapshot_sha256": {path: _sha256(runtime / path) for path in relative_paths},
        "fetched_at": pd.Timestamp.now(tz=TIMEZONE).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ready",
        "target_date": target.date().isoformat(),
        "as_of_contract": issued.isoformat(),
        "model": model,
        "model_run": run.isoformat(),
        "station_count": len(paths),
        "snapshot_paths": relative_paths,
        "manifest_path": manifest_path.relative_to(runtime).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True, dest="issued_at")
    parser.add_argument("--model-run", required=True, help="UTC model initialisation, for example 2026-08-21T18:00Z")
    parser.add_argument("--model", default="ecmwf_ifs")
    args = parser.parse_args()
    result = fetch_and_store_snapshot(
        args.runtime_root,
        target_date=args.target_date,
        issued_at=args.issued_at,
        model_run=args.model_run,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
