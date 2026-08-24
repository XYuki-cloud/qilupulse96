"""Shared fixtures for the Shandong-localized forecast test suite."""

import numpy as np
import pandas as pd
import pytest


TZ = "Asia/Shanghai"


@pytest.fixture
def hourly_index():
    """168 hours (1 week) of hourly timestamps in Asia/Shanghai."""
    return pd.date_range(
        "2025-01-06", periods=168, freq="h", tz=TZ
    )


@pytest.fixture
def sample_prices(hourly_index):
    """Realistic-looking price DataFrame with 'price_cny_mwh' column."""
    rng = np.random.default_rng(42)
    base = 380 + 120 * np.sin(2 * np.pi * np.arange(168) / 24)
    noise = rng.normal(0, 20, 168)
    prices = base + noise
    return pd.DataFrame(
        {"price_cny_mwh": prices},
        index=hourly_index,
    )


@pytest.fixture
def sample_prices_with_nan(sample_prices):
    """Price DataFrame with a 3-hour NaN gap."""
    df = sample_prices.copy()
    df.iloc[10:13, 0] = np.nan
    return df


@pytest.fixture
def sample_wind(hourly_index):
    """Wind generation DataFrame with 'wind_mw' column."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"wind_mw": rng.uniform(2000, 20000, len(hourly_index))},
        index=hourly_index,
    )


@pytest.fixture
def sample_solar(hourly_index):
    """Solar generation DataFrame with 'solar_mw' column."""
    rng = np.random.default_rng(11)
    hour = hourly_index.hour
    solar = np.where((hour >= 7) & (hour <= 18), rng.uniform(1000, 30000, len(hourly_index)), 0.0)
    return pd.DataFrame(
        {"solar_mw": solar},
        index=hourly_index,
    )


@pytest.fixture
def sample_load(hourly_index):
    """Load forecast DataFrame with 'load_mw' column."""
    rng = np.random.default_rng(13)
    base = 48000 + 5000 * np.sin(2 * np.pi * np.arange(len(hourly_index)) / 24)
    return pd.DataFrame(
        {"load_mw": base + rng.normal(0, 1000, len(hourly_index))},
        index=hourly_index,
    )


@pytest.fixture
def sample_weather(hourly_index):
    """Raw weather DataFrame (Open-Meteo style columns)."""
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        {
            "temperature_2m": 15 + 5 * np.sin(2 * np.pi * np.arange(len(hourly_index)) / 24) + rng.normal(0, 1, len(hourly_index)),
            "direct_radiation": np.where(hourly_index.hour.isin(range(7, 19)), rng.uniform(100, 800, len(hourly_index)), 0.0),
            "wind_speed_100m": rng.uniform(2, 14, len(hourly_index)),
        },
        index=hourly_index,
    )


@pytest.fixture
def empty_price_df():
    """Empty DataFrame with the expected schema."""
    idx = pd.DatetimeIndex([], dtype="datetime64[ns, Asia/Shanghai]", freq="h")
    return pd.DataFrame({"price_cny_mwh": pd.Series(dtype=float)}, index=idx)


@pytest.fixture
def feature_matrix(sample_prices):
    """Minimal feature matrix for model training: price + 2 numeric features."""
    rng = np.random.default_rng(99)
    df = sample_prices.copy()
    df["feature_a"] = rng.normal(0, 1, len(df))
    df["feature_b"] = rng.normal(0, 1, len(df))
    return df
