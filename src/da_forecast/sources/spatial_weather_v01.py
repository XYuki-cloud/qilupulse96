"""16-city spatial weather tensors and physically constrained solar features."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.openmeteo import fetch_weather_at, fetch_weather_forecast_snapshot_at
from da_forecast.sources.weather_provenance import ForecastSnapshotArchive


RADIATION_COLUMNS = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
)
DETAIL_WEATHER_COLUMNS = (
    "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "rain",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation",
    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "wind_speed_10m",
    "wind_direction_10m", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m",
)
SPATIAL_WEATHER_SOURCE = "openmeteo_spatial_v01"
_EAST_CODES = ("SD_QINGDAO", "SD_YANTAI", "SD_WEIHAI", "SD_RIZHAO")
_WEST_CODES = ("SD_LIAOCHENG", "SD_DEZHOU", "SD_HEZE", "SD_JINAN")


def _market_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)


def validate_station_weather(station_weather: Mapping[str, pd.DataFrame]) -> None:
    """Require the complete 16-city panel on one common hourly index."""
    expected = {station.code for station in SHANDONG_SPATIAL_STATIONS}
    received = set(station_weather)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        raise ValueError(f"station panel mismatch; missing={missing}, extra={extra}")
    reference: pd.DatetimeIndex | None = None
    for code in sorted(expected):
        frame = station_weather[code]
        if frame.index.has_duplicates:
            raise ValueError(f"duplicate timestamps for {code}")
        index = _market_index(frame.index)
        if reference is None:
            reference = index
        elif not index.equals(reference):
            raise ValueError(f"incomplete hourly coverage for {code}")


def _sun_times(index: pd.DatetimeIndex, location: Location) -> pd.DataFrame:
    dates = index.normalize().unique() + pd.Timedelta(hours=12)
    sun = location.get_sun_rise_set_transit(dates, method="spa")
    sun.index = sun.index.normalize()
    per_day = pd.DataFrame(index=index.normalize())
    per_day["sunrise"] = sun["sunrise"].reindex(per_day.index).to_numpy()
    per_day["sunset"] = sun["sunset"].reindex(per_day.index).to_numpy()
    per_day.index = index
    return per_day


def disaggregate_hourly_weather_to_quarters(
    hourly: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    altitude: float,
) -> pd.DataFrame:
    """Map hourly weather to quarters while preserving each radiation-hour mean."""
    if hourly.empty or hourly.index.has_duplicates:
        raise ValueError("Hourly weather must be non-empty with unique timestamps")
    local = hourly.copy()
    local.index = _market_index(local.index)
    local = local.sort_index()
    quarter_index = pd.DatetimeIndex(
        np.concatenate(
            [pd.date_range(timestamp, periods=4, freq="15min", tz=TIMEZONE).to_numpy() for timestamp in local.index]
        )
    )
    if len(quarter_index) >= 3 and pd.infer_freq(quarter_index) == "15min":
        quarter_index.freq = pd.tseries.frequencies.to_offset("15min")
    result = local.reindex(quarter_index.floor("h")).copy()
    result.index = quarter_index
    location = Location(latitude=latitude, longitude=longitude, altitude=altitude, tz=TIMEZONE)
    solar = location.get_solarposition(quarter_index)
    clearsky = location.get_clearsky(quarter_index)
    result["solar_elevation"] = solar["apparent_elevation"].to_numpy()
    result["solar_azimuth_sin"] = np.sin(np.deg2rad(solar["azimuth"].to_numpy()))
    result["solar_azimuth_cos"] = np.cos(np.deg2rad(solar["azimuth"].to_numpy()))
    result["is_daylight"] = result["solar_elevation"] > 0.0
    result["clear_sky_ghi"] = clearsky["ghi"].to_numpy()
    sun = _sun_times(quarter_index, location)
    result["minutes_from_sunrise"] = (quarter_index - pd.DatetimeIndex(sun["sunrise"])).total_seconds() / 60.0
    result["minutes_to_sunset"] = (pd.DatetimeIndex(sun["sunset"]) - quarter_index).total_seconds() / 60.0

    # Reprofile only irradiance. State variables remain the source hour's
    # value, which makes their hourly provenance explicit and reproducible.
    source_hours = result.index.floor("h")
    clear_shape = result["clear_sky_ghi"].to_numpy(dtype=float)
    clear_means = pd.Series(clear_shape, index=result.index).groupby(source_hours).transform("mean").to_numpy()
    for column in set(RADIATION_COLUMNS).intersection(local.columns):
        source_values = local[column].reindex(source_hours).to_numpy(dtype=float)
        valid_profile = np.isfinite(source_values) & (source_values > 0) & (clear_means > 0)
        values = source_values.copy()
        np.divide(source_values * clear_shape, clear_means, out=values, where=valid_profile)
        result[column] = values
    if "shortwave_radiation" in result:
        result["shortwave_clear_sky_index"] = result["shortwave_radiation"].divide(
            result["clear_sky_ghi"].where(result["clear_sky_ghi"] > 1.0)
        ).fillna(0.0)
        result["shortwave_radiation_ramp_15m"] = result["shortwave_radiation"].diff().fillna(0.0)
    result["weather_source_resolution_minutes"] = 60
    return result


def disaggregate_station_panel_to_quarters(
    station_weather: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Expand one validated 16-city hourly panel to a shared 15-minute index."""
    validate_station_weather(station_weather)
    def convert(station):
        return station.code, disaggregate_hourly_weather_to_quarters(
            station_weather[station.code],
            latitude=station.latitude,
            longitude=station.longitude,
            altitude=station.altitude_m,
        )

    with ThreadPoolExecutor(max_workers=min(8, len(SHANDONG_SPATIAL_STATIONS))) as executor:
        result = dict(executor.map(convert, SHANDONG_SPATIAL_STATIONS))
    validate_station_weather(result)
    return result


def _station_matrix(station_weather: Mapping[str, pd.DataFrame], column: str) -> pd.DataFrame:
    validate_station_weather(station_weather)
    index = next(iter(station_weather.values())).index
    return pd.DataFrame(
        {code: station_weather[code][column].to_numpy() for code in station_weather if column in station_weather[code]},
        index=index,
    )


def build_spatial_aggregate_features(station_weather: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact, interpretable spatial summaries for the XGBoost candidate."""
    validate_station_weather(station_weather)
    index = next(iter(station_weather.values())).index
    result = pd.DataFrame(index=index)
    candidates = ("shortwave_radiation", "direct_radiation", "cloud_cover", "wind_speed_100m", "wind_gusts_10m")
    for column in candidates:
        matrix = _station_matrix(station_weather, column)
        if matrix.empty:
            continue
        result[f"{column}_prov_mean"] = matrix.mean(axis=1)
        result[f"{column}_prov_std"] = matrix.std(axis=1)
        result[f"{column}_prov_p10"] = matrix.quantile(0.1, axis=1)
        result[f"{column}_prov_p90"] = matrix.quantile(0.9, axis=1)
        east = matrix.reindex(columns=[code for code in _EAST_CODES if code in matrix], copy=False).mean(axis=1)
        west = matrix.reindex(columns=[code for code in _WEST_CODES if code in matrix], copy=False).mean(axis=1)
        result[f"east_west_{column}"] = east - west
    if "shortwave_radiation" in result:
        result["shortwave_radiation_ramp_15m"] = result["shortwave_radiation_prov_mean"].diff().fillna(0.0)
    return result


class SpatialPCAReducer:
    """Train-fold-only PCA over station-level weather fields."""

    def __init__(self, *, columns: tuple[str, ...], n_components: int = 3) -> None:
        self.columns = columns
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def _matrix(self, station_weather: Mapping[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, np.ndarray]:
        validate_station_weather(station_weather)
        index = next(iter(station_weather.values())).index
        fields = []
        for column in self.columns:
            values = _station_matrix(station_weather, column)
            if values.shape[1] != len(SHANDONG_SPATIAL_STATIONS):
                raise ValueError(f"Every station must provide '{column}' for PCA")
            fields.append(values.to_numpy(dtype=float))
        return index, np.concatenate(fields, axis=1)

    def fit(self, station_weather: Mapping[str, pd.DataFrame]) -> "SpatialPCAReducer":
        _index, matrix = self._matrix(station_weather)
        components = min(self.n_components, matrix.shape[0], matrix.shape[1])
        if components < 1:
            raise ValueError("PCA requires at least one sample")
        scaled = self.scaler.fit_transform(matrix)
        self.pca = PCA(n_components=components, random_state=7).fit(scaled)
        return self

    def transform(self, station_weather: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        if self.pca is None:
            raise RuntimeError("SpatialPCAReducer must be fitted on the training fold first")
        index, matrix = self._matrix(station_weather)
        scores = self.pca.transform(self.scaler.transform(matrix))
        return pd.DataFrame(scores, index=index, columns=[f"spatial_pca_{idx + 1}" for idx in range(scores.shape[1])])

    def fit_transform(self, station_weather: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        return self.fit(station_weather).transform(station_weather)


def fetch_observed_spatial_weather(
    start_date: pd.Timestamp, end_date: pd.Timestamp, *, cache_dir: Path | str
) -> dict[str, pd.DataFrame]:
    """Incrementally cache all 18 Archive variables for the complete city panel."""
    result = {
        station.code: fetch_weather_at(
            station.latitude,
            station.longitude,
            start_date,
            end_date,
            Path(cache_dir),
            zone=station.code,
            source=SPATIAL_WEATHER_SOURCE,
            variables=DETAIL_WEATHER_COLUMNS,
        )
        for station in SHANDONG_SPATIAL_STATIONS
    }
    validate_station_weather(result)
    return result


def load_cached_observed_spatial_weather(*, cache_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load a complete 16-city Archive panel without accepting gaps."""
    cache = ParquetCache(Path(cache_dir))
    result: dict[str, pd.DataFrame] = {}
    for station in SHANDONG_SPATIAL_STATIONS:
        frame = cache.load(SPATIAL_WEATHER_SOURCE, station.code, "weather")
        if frame is None or frame.empty:
            raise FileNotFoundError(f"No spatial weather cache for {station.code}")
        missing = sorted(set(DETAIL_WEATHER_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Spatial weather cache for {station.code} misses {missing}")
        result[station.code] = frame[list(DETAIL_WEATHER_COLUMNS)]
    validate_station_weather(result)
    return result


def load_or_build_observed_spatial_quarters(*, cache_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Cache the expensive 16-city solar-geometry expansion for reuse in backtests."""
    root = Path(cache_dir)
    cache = ParquetCache(root)
    source = "openmeteo_spatial_v01_quarter"
    hourly = load_cached_observed_spatial_weather(cache_dir=root)
    hourly_index = next(iter(hourly.values())).index
    expected_start = hourly_index.min()
    expected_end = hourly_index.max() + pd.Timedelta(minutes=45)
    cached: dict[str, pd.DataFrame] = {}
    for station in SHANDONG_SPATIAL_STATIONS:
        frame = cache.load(source, station.code, "weather")
        if frame is None or frame.empty or not {"shortwave_clear_sky_index", "shortwave_radiation_ramp_15m"}.issubset(frame.columns):
            cached = {}
            break
        cached[station.code] = frame
    if cached and all(frame.index.min() <= expected_start and frame.index.max() >= expected_end for frame in cached.values()):
        validate_station_weather(cached)
        return cached
    quarters = disaggregate_station_panel_to_quarters(hourly)
    for code, frame in quarters.items():
        cache.save(source, code, "weather", frame)
    return quarters


def fetch_forecast_spatial_weather(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    archive: ForecastSnapshotArchive,
    issued_at: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch and append-only archive one production Forecast issue for all cities."""
    result: dict[str, pd.DataFrame] = {}
    paths: list[str] = []
    for station in SHANDONG_SPATIAL_STATIONS:
        snapshot = fetch_weather_forecast_snapshot_at(
            station.latitude,
            station.longitude,
            start_date,
            end_date,
            issued_at=issued_at,
            variables=DETAIL_WEATHER_COLUMNS,
        )
        result[station.code] = snapshot.weather
        paths.append(str(archive.store(station.code, snapshot.weather, snapshot.payloads, issued_at=snapshot.issued_at)))
    validate_station_weather(result)
    return result, paths
