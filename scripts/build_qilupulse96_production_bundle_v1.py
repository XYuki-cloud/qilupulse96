#!/usr/bin/env python3
"""Build the auditable QiluPulse-96 v1.0 production bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.feature_schema_v1 import feature_schema
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1
from da_forecast.models.qilupulse96_v1 import CONTRACT_VERSION
from da_forecast.models.adaptive_normalization import recent_state_features
from da_forecast.features.calendar_v01 import build_calendar_v01
from da_forecast.sources.spatial_weather_v01 import load_or_build_observed_spatial_quarters, DETAIL_WEATHER_COLUMNS
from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
from da_forecast.sources.shandong_market_xlsx import DAY_AHEAD_PRICE_COL

SOLAR_COLUMNS = ("solar_elevation", "solar_azimuth_sin", "solar_azimuth_cos", "is_daylight", "clear_sky_ghi", "shortwave_clear_sky_index", "shortwave_radiation_ramp_15m")


def _fit_scalers(data_root: Path, weather_root: Path, *, realtime_only: bool = False) -> PreprocessingStateV1:
    market = data_root / "shandong_all_network" / "SD"
    if not market.exists():
        matches = list(data_root.rglob("realtime_prices_15min.parquet"))
        if matches:
            market = matches[0].parent
    if not market.exists():
        raise FileNotFoundError(f"Cannot locate canonical Shandong market parquet under {data_root}")
    prices = pd.read_parquet(market / "realtime_prices_15min.parquet").sort_index()
    da = None if realtime_only else pd.read_parquet(market / "day_ahead_prices_15min.parquet").sort_index()
    index = prices.index
    index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    prices.index = index
    if da is not None:
        da.index = da.index.tz_localize(TIMEZONE) if da.index.tz is None else da.index.tz_convert(TIMEZONE)
    price_raw = prices["price_cny_mwh"].to_numpy(dtype=float)
    calendar = build_calendar_v01(index).astype(float)
    if realtime_only:
        history_base = calendar.to_numpy(dtype=float)
    else:
        da_values = da[DAY_AHEAD_PRICE_COL].reindex(index).to_numpy(dtype=float)
        history_base = np.column_stack([calendar.to_numpy(dtype=float), da_values, price_raw - da_values])
    target_base = calendar.to_numpy(dtype=float)
    direct = {station.code: pd.read_parquet(weather_root / station.code / "weather.parquet") for station in SHANDONG_SPATIAL_STATIONS if (weather_root / station.code / "weather.parquet").is_file()}
    if len(direct) == len(SHANDONG_SPATIAL_STATIONS):
        panel = direct
    else:
        try:
            panel = load_or_build_observed_spatial_quarters(cache_dir=weather_root)
        except (FileNotFoundError, ValueError):
            sibling = weather_root.parent / "openmeteo_spatial_v01_quarter"
            direct = {station.code: pd.read_parquet(sibling / station.code / "weather.parquet") for station in SHANDONG_SPATIAL_STATIONS}
            panel = direct
    station = np.stack([panel[station.code].reindex(index)[list(DETAIL_WEATHER_COLUMNS + SOLAR_COLUMNS)].to_numpy(dtype=float) for station in SHANDONG_SPATIAL_STATIONS], axis=1)
    context_slots = 90 * 96
    train_dates = pd.date_range(index[0].normalize() + pd.Timedelta(days=91), pd.Timestamp("2025-12-30", tz=TIMEZONE), freq="D", tz=TIMEZONE)
    positions = np.asarray([index.get_loc(day) for day in train_dates], dtype=int)
    fit_positions = positions[:-14]
    fit_end = int(fit_positions[-1]) + 96
    history_fit_end = max(0, fit_end - 97 + 1)
    def state(pos: int) -> np.ndarray:
        end = pos - 53
        return recent_state_features(price_raw[end - context_slots + 1:end + 1])
    state_values = np.stack([state(int(pos)) for pos in fit_positions])
    def scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fitted = StandardScaler().fit(values)
        return fitted.mean_.astype(np.float32), fitted.scale_.astype(np.float32)
    price_mean, price_scale = scaler(price_raw[:fit_end, None])
    history_mean, history_scale = scaler(history_base[:history_fit_end])
    target_mean, target_scale = scaler(target_base[:fit_end])
    station_mean, station_scale = scaler(station[:fit_end].reshape(-1, station.shape[-1]))
    state_mean, state_scale = scaler(state_values)
    from da_forecast.production.preprocessing_v1 import ArrayScalerState
    return PreprocessingStateV1(
        price=ArrayScalerState(price_mean, price_scale), history_extra=ArrayScalerState(history_mean, history_scale),
        target_extra=ArrayScalerState(target_mean, target_scale), station_weather=ArrayScalerState(station_mean, station_scale),
        state_features=ArrayScalerState(state_mean, state_scale), robust_normalizer={"eps": 1e-4, "mad_multiplier": 1.4826},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-data-snapshot-hash", required=True)
    parser.add_argument("--calendar-reference-hash", required=True)
    parser.add_argument("--market-data-root", type=Path, default=None)
    parser.add_argument("--weather-root", type=Path, default=None)
    parser.add_argument(
        "--synthetic-preprocessing",
        action="store_true",
        help="Use identity preprocessing and mark the bundle as synthetic-test-only",
    )
    parser.add_argument("--realtime-only", action="store_true", help="Build a bundle without day-ahead price features")
    args = parser.parse_args(argv)
    has_market = args.market_data_root is not None
    has_weather = args.weather_root is not None
    if args.synthetic_preprocessing and (has_market or has_weather):
        parser.error("--synthetic-preprocessing cannot be combined with market/weather roots")
    if not args.synthetic_preprocessing and not (has_market and has_weather):
        parser.error(
            "provide both --market-data-root and --weather-root, or explicitly use "
            "--synthetic-preprocessing"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    preprocessing = (
        PreprocessingStateV1.identity(
            history_extra_dim=14 if args.realtime_only else 16,
            target_extra_dim=14,
            station_dim=25,
            state_dim=5,
        )
        if args.synthetic_preprocessing
        else _fit_scalers(
            args.market_data_root,
            args.weather_root,
            realtime_only=args.realtime_only,
        )
    )
    bundle = QiluPulse96ProductionBundle.from_artifact(
        args.checkpoint,
        preprocessing=preprocessing,
        training_metadata={
            "train_end": "2025-12-30", "validation_days": 14,
            "training_weather_kind": "observed_proxy", "production_weather_kind": "forecast",
            "training_data_snapshot_hash": args.training_data_snapshot_hash,
            "calendar_reference_hash": args.calendar_reference_hash,
            "price_features": "realtime_only" if args.realtime_only else "realtime_plus_day_ahead",
            "preprocessing_kind": "identity_synthetic" if args.synthetic_preprocessing else "fitted_from_supplied_data",
        },
    )
    bundle.manifest_data.update({
        "contract_version": CONTRACT_VERSION,
        "model_version": "1.1.0-realtime-only" if args.realtime_only else bundle.manifest_data.get("model_version", "1.0.0"),
        "calibration": {
            "enabled": True,
            "required": bool(args.realtime_only),
            "bootstrap_history": bool(args.realtime_only),
            "window_days": 56,
            "min_days": 14,
            "half_life_days": 28.0,
        },
        "feature_schema": feature_schema(realtime_only=args.realtime_only),
        "station_order": [station.code for station in SHANDONG_SPATIAL_STATIONS],
        "intended_use": "synthetic_test_only" if args.synthetic_preprocessing else "operator_supplied_data",
        "calibration_config": {
            "version": "frozen_adaln_bias_interval_v02", "window_days": 56,
            "min_days": 14, "half_life_days": 28.0, "long_weight": 0.65,
            "recent_weight": 0.35, "robust_scale_floor": 20.0, "bias_clip_cny_mwh": 100.0,
        },
    })
    destination = bundle.save(args.output)
    print(json.dumps({"bundle": str(destination), "parameter_checksum": bundle.parameter_checksum, "bundle_sha256": bundle.bundle_sha256}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
