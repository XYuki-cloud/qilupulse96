"""Versioned China-calendar features for 15-minute D+1 forecasts."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from da_forecast.config import PROJECT_ROOT, TIMEZONE


@lru_cache(maxsize=None)
def _adjusted_workdays(years: tuple[int, ...], reference_root: str | None = None) -> frozenset[pd.Timestamp]:
    """Load versioned State Council make-up workdays for the requested years."""
    dates: set[pd.Timestamp] = set()
    reference_dir = Path(reference_root) if reference_root else Path(os.environ.get("DA_FORECAST_CALENDAR_REFERENCE_DIR", PROJECT_ROOT / "data" / "reference"))
    for year in years:
        candidates = (
            reference_dir / f"china_workday_overrides_{year}_confirmed.json",
            reference_dir / f"china_workday_overrides_{year}.json",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates.update(pd.Timestamp(value).normalize() for value in payload.get("adjusted_workdays", []))
    return frozenset(dates)


def build_calendar_v01(index: pd.DatetimeIndex, *, reference_dir: str | Path | None = None) -> pd.DataFrame:
    """Return strictly deterministic, target-date-known calendar features."""
    local = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    dates = local.normalize().tz_localize(None)
    years = sorted(set(local.year))
    cn_holidays = holidays.country_holidays("CN", years=years)
    holiday = np.asarray([date.date() in cn_holidays for date in dates], dtype=bool)
    adjusted = np.asarray([date in _adjusted_workdays(tuple(years), str(reference_dir) if reference_dir else None) for date in dates], dtype=bool)
    weekend = np.asarray(local.weekday >= 5, dtype=bool)
    effective_weekend = weekend & ~adjusted & ~holiday
    days_in_year = np.where(local.is_leap_year, 366, 365)
    day_progress = (np.asarray(local.dayofyear, dtype=float) - 1.0) / (days_in_year - 1.0)
    month_progress = (np.asarray(local.day, dtype=float) - 1.0) / (np.asarray(local.days_in_month, dtype=float) - 1.0)
    slot = local.hour * 4 + local.minute // 15
    weekday = np.asarray(local.weekday, dtype=float)
    return pd.DataFrame(
        {
            "slot_sin": np.sin(2 * np.pi * slot / 96),
            "slot_cos": np.cos(2 * np.pi * slot / 96),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "day_of_year_progress": day_progress,
            "annual_sin": np.sin(2 * np.pi * day_progress),
            "annual_cos": np.cos(2 * np.pi * day_progress),
            "month_of_year": local.month,
            "quarter": local.quarter,
            "month_progress": month_progress,
            "is_public_holiday": holiday,
            "is_adjusted_workday": adjusted,
            "is_weekend_effective": effective_weekend,
            "is_regular_workday": ~holiday & ~effective_weekend,
        },
        index=index,
    )
