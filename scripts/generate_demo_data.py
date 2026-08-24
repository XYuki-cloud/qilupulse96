#!/usr/bin/env python3
"""Generate deterministic synthetic (demo) data for the Shandong zone (SD).

Writes into the parquet cache under source="shandong" so the pipeline runs
end-to-end offline:

    data/raw/shandong/SD/day_ahead_prices.parquet   (price_cny_mwh)
    data/raw/shandong/SD/load_forecast.parquet       (load_mw)
    data/raw/shandong/SD/wind_solar_forecast.parquet (wind_mw, solar_mw)
    data/raw/shandong/SD/weather.parquet             (temperature_2m, direct_radiation, wind_speed_100m)

NOTE: This overwrites the SD cache files. Keep generated demo data separate
from any authorized runtime inputs.

Usage:
    uv run python scripts/generate_demo_data.py
    uv run python scripts/generate_demo_data.py --start 2024-01-01 --end 2025-12-31
    uv run python scripts/generate_demo_data.py --days 120 --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, PRICE_COL, PRICE_RANGE_MIN, PRICE_RANGE_MAX
from da_forecast.sources.cache import ParquetCache

TZ = "Asia/Shanghai"
SOURCE = "shandong"


def generate(start: str, end: str | None, days: int | None, seed: int, zone: str) -> dict:
    rng = np.random.default_rng(seed)
    if days is not None:
        n_hours = days * 24
        idx = pd.date_range(start, periods=n_hours, freq="h", tz=TZ)
    else:
        idx = pd.date_range(start, end=end, freq="h", tz=TZ, inclusive="left")
        n_hours = len(idx)

    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    weekday = idx.weekday.to_numpy()

    # --- weather ---
    temperature = (
        15
        + 12 * np.sin(2 * np.pi * (doy - 100) / 365)
        + 4 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 1.5, n_hours)
    )
    solar_shape = np.maximum(0, np.sin(np.pi * (hour - 6) / 12.0))
    direct_radiation = np.clip(
        solar_shape * (450 + 150 * np.sin(2 * np.pi * (doy - 100) / 365))
        + rng.normal(0, 30, n_hours),
        0, None,
    )
    wind_speed_100m = np.clip(
        6 + 2.5 * np.sin(2 * np.pi * (doy - 30) / 365) + rng.normal(0, 2, n_hours), 0, None
    )

    # --- wind / solar generation ---
    wind_ar = np.zeros(n_hours)
    w = 9000.0
    for i in range(n_hours):
        w = 0.97 * w + rng.normal(0, 800)
        wind_ar[i] = w
    wind_mw = np.clip(
        wind_ar + 2500 * np.sin(2 * np.pi * (doy - 120) / 365), 0, 22000
    )
    solar_mw = np.clip(
        solar_shape * 40000 * (0.85 + 0.15 * np.sin(2 * np.pi * (doy - 100) / 365))
        + rng.normal(0, 1500, n_hours) * solar_shape,
        0, None,
    )

    # --- load ---
    load_mw = np.clip(
        48000
        + 8000 * np.sin(2 * np.pi * (doy - 100) / 365)
        + 2500 * np.sin(2 * np.pi * (hour - 16) / 24)
        + 500 * (temperature - 15)
        - 4000 * (weekday >= 5)
        + rng.normal(0, 1200, n_hours),
        28000, 90000,
    )

    # --- price (CNY/MWh) ---
    residual = load_mw - wind_mw - solar_mw
    morning_peak = np.exp(-((hour - 9.5) ** 2) / 6.0)
    evening_peak = np.exp(-((hour - 19.0) ** 2) / 8.0)
    price = (
        380
        + 60 * np.sin(2 * np.pi * (doy - 100) / 365)
        + 120 * morning_peak
        + 160 * evening_peak
        - 40 * (weekday >= 5)
        + (residual - residual.mean()) / 8000 * 80
        - 60 * (solar_mw > 25000)
        + rng.normal(0, 25, n_hours)
    )
    # occasional scarcity spikes
    spike_mask = rng.random(n_hours) < 0.004
    price[spike_mask] += rng.uniform(250, 600, int(spike_mask.sum()))
    price = np.clip(price, PRICE_RANGE_MIN, PRICE_RANGE_MAX)

    frames = {
        "day_ahead_prices": pd.DataFrame({PRICE_COL: price.round(2)}, index=idx),
        "load_forecast": pd.DataFrame({"load_mw": load_mw.round(1)}, index=idx),
        "wind_solar_forecast": pd.DataFrame(
            {"wind_mw": wind_mw.round(1), "solar_mw": solar_mw.round(1)}, index=idx
        ),
        "weather": pd.DataFrame(
            {
                "temperature_2m": temperature.round(2),
                "direct_radiation": direct_radiation.round(2),
                "wind_speed_100m": wind_speed_100m.round(2),
            },
            index=idx,
        ),
    }
    return {"zone": zone, "idx": idx, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for Shandong (SD).")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-03-31", help="Default: ~3 months for a fast end-to-end run")
    parser.add_argument("--days", type=int, default=None, help="Override: number of days (periods = days*24)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zone", default="SD")
    args = parser.parse_args()

    out = generate(args.start, args.end, args.days, args.seed, args.zone)
    cache = ParquetCache(RAW_DIR)
    for datatype, df in out["frames"].items():
        cache.save(SOURCE, args.zone, datatype, df)

    prices = out["frames"]["day_ahead_prices"]
    print(f"Demo data written to cache: source='{SOURCE}' zone={args.zone}")
    print(f"  Period: {out['idx'].min()} -> {out['idx'].max()}  ({len(out['idx'])} hours)")
    print(f"  Price mean={prices[PRICE_COL].mean():.1f} CNY/MWh, "
          f"min={prices[PRICE_COL].min():.1f}, max={prices[PRICE_COL].max():.1f}, "
          f"negative hours={(prices[PRICE_COL] < 0).sum()}")
    print(f"  Files: {RAW_DIR / SOURCE / args.zone}")


if __name__ == "__main__":
    main()
