"""Open-Meteo weather adapter for the public Shandong workflow.

The adapter keeps the network boundary explicit: callers provide coordinates,
dates, and an optional cache directory; the module returns timezone-aware
frames and, for forecasts, the request/response evidence used to build them.
Open-Meteo's public archive and forecast APIs do not require an API key.

API documentation: https://open-meteo.com/en/docs
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from da_forecast.config import API_BACKOFF_SECONDS, API_MAX_RETRIES, SHANDONG_SPATIAL_STATIONS
from da_forecast.sources.cache import ParquetCache

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "direct_radiation",
    "diffuse_radiation",
]

# Local timezone used when requesting/parsing weather data (China has no DST).
WEATHER_TIMEZONE = "Asia/Shanghai"

# Coordinates are defined once in the public Shandong station contract.
ZONE_WEATHER_COORDS: dict[str, tuple[float, float]] = {
    station.code: (station.latitude, station.longitude)
    for station in SHANDONG_SPATIAL_STATIONS
}
ZONE_WEATHER_COORDS["SD"] = ZONE_WEATHER_COORDS["SD_JINAN"]

# Open-Meteo hourly archive caps at roughly 1 year per request.
MAX_DAYS_PER_REQUEST = 365


@dataclass(frozen=True)
class ForecastWeatherSnapshot:
    """Parsed forecast weather plus the exact public API responses that produced it."""

    weather: pd.DataFrame
    payloads: list[dict]
    issued_at: pd.Timestamp
    weather_kind: str = "forecast"


def _request_with_retry(params: dict, url: str = ARCHIVE_URL) -> dict:
    """Send GET request to Open-Meteo with exponential back-off."""
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and data["error"]:
                raise ValueError(data.get("reason", "Unknown Open-Meteo error"))
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt < API_MAX_RETRIES - 1:
                wait = API_BACKOFF_SECONDS[attempt]
                logger.warning(
                    "Open-Meteo request failed (attempt %d): %s. Retrying in %ds...",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


def _parse_response(data: dict, variables: list[str] | tuple[str, ...] = HOURLY_VARIABLES) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a DataFrame with UTC DatetimeIndex."""
    hourly = data["hourly"]
    idx = pd.to_datetime(hourly["time"])
    # Open-Meteo returns timestamps in the requested timezone; convert to UTC.
    idx = idx.tz_localize(WEATHER_TIMEZONE, ambiguous="infer").tz_convert("UTC")
    df = pd.DataFrame(
        {var: hourly[var] for var in variables if var in hourly},
        index=idx,
    )
    df.index.name = "utc_timestamp"
    return df


def _date_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[str, str]]:
    """Split a date range into chunks of at most MAX_DAYS_PER_REQUEST days.

    Returns pairs of (start_date, end_date) formatted as YYYY-MM-DD strings.
    The *end* date is inclusive in the Open-Meteo API, so the last chunk's end
    is clamped to ``end - 1 day`` (we don't want to include the boundary day
    of the next chunk twice).
    """
    if end <= start:
        raise ValueError("end must be later than start")

    chunks: list[tuple[str, str]] = []
    current = start.normalize()
    final = (end - pd.Timedelta(days=1)).normalize()
    while current <= final:
        chunk_end = current + pd.Timedelta(days=MAX_DAYS_PER_REQUEST - 1)
        if chunk_end > final:
            chunk_end = final
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_weather(
    zone: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch historical weather for a configured zone (single coordinate)."""
    if zone not in ZONE_WEATHER_COORDS:
        raise ValueError(f"Unknown zone '{zone}'. Must be one of {list(ZONE_WEATHER_COORDS)}")
    lat, lon = ZONE_WEATHER_COORDS[zone]
    return fetch_weather_at(lat, lon, start_date, end_date, cache_dir, zone=zone)


def fetch_weather_at(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    zone: str = "weather",
    source: str = "openmeteo",
    variables: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Fetch historical weather at (lat, lon) between *start_date* and *end_date*.

    Checks the parquet cache first.  If the cached data already covers the
    requested range, the cached DataFrame is returned without hitting the API.
    New data is merged into the cache after fetching.

    Parameters
    ----------
    lat, lon : float
        Weather station coordinates.
    start_date, end_date : pd.Timestamp
        Half-open interval ``[start_date, end_date)``.
    cache_dir : Path
        Root directory for parquet cache files (typically ``data/raw``).
    zone : str
        Cache zone key (e.g. a city code such as ``SD_QINGDAO``).
    source : str
        Cache source key (default ``openmeteo``).

    Returns
    -------
    pd.DataFrame
        Hourly weather data with UTC DatetimeIndex.
    """
    cache = ParquetCache(cache_dir)
    requested_variables = tuple(variables or HOURLY_VARIABLES)
    datatype = "weather"

    # Check if cache already covers the requested range.
    cached = cache.load(source, zone, datatype)
    if cached is not None and not cached.empty:
        start_utc = start_date.tz_convert("UTC") if start_date.tzinfo else start_date.tz_localize("UTC")
        end_utc = end_date.tz_convert("UTC") if end_date.tzinfo else end_date.tz_localize("UTC")
        expected_index = pd.date_range(
            start_utc,
            end_utc - pd.Timedelta(hours=1),
            freq="h",
            tz="UTC",
        )
        cached_index = cached.index
        cached_index = (
            cached_index.tz_localize("UTC")
            if cached_index.tz is None
            else cached_index.tz_convert("UTC")
        )
        if (
            set(requested_variables).issubset(cached.columns)
            and cached.index.min() <= start_utc
            and cached.index.max() >= end_utc - pd.Timedelta(hours=1)
            and expected_index.isin(cached_index).all()
        ):
            return cached.loc[start_utc:end_utc]

    chunks = _date_chunks(start_date, end_date)

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in chunks:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(requested_variables),
            "timezone": WEATHER_TIMEZONE,
        }
        data = _request_with_retry(params)
        df_chunk = _parse_response(data, requested_variables)
        if not df_chunk.empty:
            frames.append(df_chunk)
        time.sleep(0.3)  # polite rate-limiting

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    # Remove any duplicate timestamps from overlapping chunks.
    df = df[~df.index.duplicated(keep="first")]

    cache.merge(source, zone, datatype, df)
    return df


def fetch_weather_forecast_at(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch Open-Meteo *forecast* weather for a future date range.

    The forecast API needs no API key and covers roughly today +/- a few
    days (up to 16 days ahead). Used by the prediction script for target
    dates beyond the archive cache.

    Returns
    -------
    pd.DataFrame
        Hourly forecast with UTC DatetimeIndex and the same columns as the
        archive fetch (temperature_2m, wind_speed_*, *_radiation).
    """
    return fetch_weather_forecast_snapshot_at(lat, lon, start_date, end_date).weather


def fetch_weather_forecast_snapshot_at(
    lat: float,
    lon: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    issued_at: pd.Timestamp | None = None,
    variables: list[str] | tuple[str, ...] | None = None,
) -> ForecastWeatherSnapshot:
    """Fetch a forecast and retain request/response evidence for later replay."""
    requested_variables = tuple(variables or HOURLY_VARIABLES)
    chunks = _date_chunks(start_date, end_date)
    frames: list[pd.DataFrame] = []
    payloads: list[dict] = []
    for chunk_start, chunk_end in chunks:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(requested_variables),
            "timezone": WEATHER_TIMEZONE,
        }
        data = _request_with_retry(params, url=FORECAST_URL)
        payloads.append({"request": params, "response": data})
        df_chunk = _parse_response(data, requested_variables)
        if not df_chunk.empty:
            frames.append(df_chunk)
        time.sleep(0.3)
    weather = pd.DataFrame() if not frames else pd.concat(frames).sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    observed = issued_at or pd.Timestamp.now(tz=WEATHER_TIMEZONE)
    if observed.tz is None:
        observed = observed.tz_localize(WEATHER_TIMEZONE)
    return ForecastWeatherSnapshot(
        weather=weather,
        payloads=payloads,
        issued_at=observed.tz_convert(WEATHER_TIMEZONE),
    )


def fetch_weather_single_run_snapshot_at(
    lat: float,
    lon: float,
    *,
    target_date: pd.Timestamp,
    model_run: pd.Timestamp,
    issued_at: pd.Timestamp,
    model: str = "ecmwf_ifs",
    variables: list[str] | tuple[str, ...] | None = None,
) -> ForecastWeatherSnapshot:
    """Fetch one archived model run and retain only one target local day.

    ``issued_at`` is the project's business decision-time contract.  The
    Open-Meteo ``run`` parameter is a separate UTC model initialisation time;
    both are retained in the request payload so a later reviewer can tell
    exactly which historical run supplied the target-day values.
    """
    requested_variables = tuple(variables or HOURLY_VARIABLES)
    target = pd.Timestamp(target_date)
    target = target.tz_localize(WEATHER_TIMEZONE) if target.tz is None else target.tz_convert(WEATHER_TIMEZONE)
    target = target.normalize()
    run = pd.Timestamp(model_run)
    run = run.tz_localize("UTC") if run.tz is None else run.tz_convert("UTC")
    issued = pd.Timestamp(issued_at)
    issued = issued.tz_localize(WEATHER_TIMEZONE) if issued.tz is None else issued.tz_convert(WEATHER_TIMEZONE)
    if run >= issued.tz_convert("UTC"):
        raise ValueError("model_run must be earlier than the business as-of contract")

    run_local = run.tz_convert(WEATHER_TIMEZONE).normalize()
    forecast_days = max(2, min(16, int((target - run_local).days) + 2))
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(requested_variables),
        "timezone": WEATHER_TIMEZONE,
        "forecast_days": forecast_days,
        "models": model,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
    }
    data = _request_with_retry(params, url=SINGLE_RUNS_URL)
    full_weather = _parse_response(data, requested_variables)
    expected_local = pd.date_range(target, periods=24, freq="h", tz=WEATHER_TIMEZONE)
    expected_utc = expected_local.tz_convert("UTC")
    if not expected_utc.isin(full_weather.index).all():
        raise ValueError("single model run lacks complete 24-hour target-day coverage")
    weather = full_weather.reindex(expected_utc)
    missing_columns = [column for column in requested_variables if column not in weather.columns]
    if missing_columns or weather[list(requested_variables)].isna().any().any():
        detail = f"missing_columns={missing_columns}" if missing_columns else "null target-day values"
        raise ValueError(f"single model run lacks complete 24-hour target-day coverage: {detail}")
    return ForecastWeatherSnapshot(
        weather=weather,
        payloads=[{"request": params, "response": data}],
        issued_at=issued,
    )
