"""Import the monthly Shandong all-network 15-minute market workbooks.

The source timestamps are settlement-period *end* times.  This module stores
them as period starts, so the source's ``24:00`` belongs to 23:45 on the same
market date.  It deliberately has no dependency on the legacy company/PMOS
hourly caches: the two price series have not been reconciled.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from da_forecast.config import PRICE_COL, PRICE_RANGE_MAX, PRICE_RANGE_MIN, TIMEZONE


REALTIME_SHEET = "实时出清数据"
DAY_AHEAD_DISCLOSURE_SHEET = "日前披露数据"
PRICE_COLUMN = "实时出清电价"
DAY_AHEAD_PRICE_SHEET = "日前出清数据"
DAY_AHEAD_PRICE_COLUMN = "日前出清电价"
DAY_AHEAD_PRICE_COL = "day_ahead_price_cny_mwh"

FORECAST_FEATURE_COLUMNS = (
    "load_forecast_mw",
    "wind_forecast_mw",
    "solar_forecast_mw",
    "intertie_forecast_mw",
    "bidding_space_forecast_mw",
    "load_factor_forecast",
    "local_plant_forecast_mw",
    "captive_unit_forecast_mw",
)

_DISCLOSURE_COLUMNS = {
    "负荷信息预测": "load_forecast_mw",
    "日前风电总加（MW）": "wind_forecast_mw",
    "日前光伏总加（MW）": "solar_forecast_mw",
    "联络线信息预测": "intertie_forecast_mw",
    "竞价空间预测(MW)": "bidding_space_forecast_mw",
    "负荷率预测": "load_factor_forecast",
    "日前地方电厂发电总加（MW）": "local_plant_forecast_mw",
    "日前自备机组总加（MW）": "captive_unit_forecast_mw",
}
_TIME_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::\d{2})?$")


def _delivery_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
    """Convert source date/right-end time columns to Shanghai period starts."""
    if "目标日期" not in raw or "时刻" not in raw:
        raise ValueError("Expected columns '目标日期' and '时刻'")
    dates = pd.to_datetime(raw["目标日期"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Invalid target date in workbook")
    starts = []
    for day, value in zip(dates, raw["时刻"]):
        match = _TIME_RE.match(str(value).strip())
        if match is None:
            raise ValueError(f"Invalid quarter-hour label: {value!r}")
        hour, minute = int(match["hour"]), int(match["minute"])
        if hour > 24 or minute not in (0, 15, 30, 45) or (hour == 24 and minute != 0):
            raise ValueError(f"Invalid quarter-hour label: {value!r}")
        starts.append(day + pd.Timedelta(hours=hour, minutes=minute) - pd.Timedelta(minutes=15))
    return pd.DatetimeIndex(starts).tz_localize(TIMEZONE)


def _validate_complete_days(index: pd.DatetimeIndex) -> None:
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate quarter-hour slots: {duplicates[:3]}")
    for day in index.normalize().unique():
        actual = index[index.normalize() == day].sort_values()
        expected = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
        if len(actual) != len(expected) or not (actual == expected).all():
            raise ValueError(f"{day.date()}: expected 96 quarter-hour slots, got {len(actual)}")


def _base_frame(raw: pd.DataFrame, source_file: str | None) -> pd.DataFrame:
    index = _delivery_index(raw)
    _validate_complete_days(index)
    return pd.DataFrame(
        {
            "market_date": index.normalize().tz_localize(None),
            "source_file": source_file or "<in-memory>",
        },
        index=index,
    )


def parse_realtime_prices(raw: pd.DataFrame, source_file: str | None = None) -> pd.DataFrame:
    """Parse and validate all-network real-time clearing prices at 15 minutes."""
    if PRICE_COLUMN not in raw:
        raise ValueError(f"Expected '{PRICE_COLUMN}' in the real-time price sheet")
    price = pd.to_numeric(raw[PRICE_COLUMN], errors="coerce")
    dates = pd.to_datetime(raw["目标日期"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Invalid target date in workbook")
    empty_day_mask = price.groupby(dates).transform("count") == 0
    usable_raw = raw.loc[~empty_day_mask].copy()
    price = price.loc[~empty_day_mask]
    if usable_raw.empty:
        raise ValueError("Real-time price sheet has no settled delivery days")
    result = _base_frame(usable_raw, source_file)
    if price.isna().any():
        raise ValueError("Real-time price sheet contains missing prices within a delivery day")
    if not price.between(PRICE_RANGE_MIN, PRICE_RANGE_MAX).all():
        raise ValueError(f"Real-time prices must be within [{PRICE_RANGE_MIN}, {PRICE_RANGE_MAX}] CNY/MWh")
    result[PRICE_COL] = price.to_numpy(dtype=float)
    return result[[PRICE_COL, "market_date", "source_file"]].sort_index()


def parse_day_ahead_prices(raw: pd.DataFrame, source_file: str | None = None) -> pd.DataFrame:
    """Parse target-day all-network day-ahead prices at 15-minute resolution."""
    if DAY_AHEAD_PRICE_COLUMN not in raw:
        raise ValueError(f"Expected '{DAY_AHEAD_PRICE_COLUMN}' in the day-ahead price sheet")
    price = pd.to_numeric(raw[DAY_AHEAD_PRICE_COLUMN], errors="coerce")
    dates = pd.to_datetime(raw["目标日期"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Invalid target date in workbook")
    empty_day_mask = price.groupby(dates).transform("count") == 0
    usable_raw = raw.loc[~empty_day_mask].copy()
    price = price.loc[~empty_day_mask]
    if usable_raw.empty:
        raise ValueError("Day-ahead price sheet has no published delivery days")
    result = _base_frame(usable_raw, source_file)
    if price.isna().any():
        raise ValueError("Day-ahead price sheet contains missing prices within a delivery day")
    if not price.between(PRICE_RANGE_MIN, PRICE_RANGE_MAX).all():
        raise ValueError(f"Day-ahead prices must be within [{PRICE_RANGE_MIN}, {PRICE_RANGE_MAX}] CNY/MWh")
    result[DAY_AHEAD_PRICE_COL] = price.to_numpy(dtype=float)
    return result[[DAY_AHEAD_PRICE_COL, "market_date", "source_file"]].sort_index()


def parse_day_ahead_disclosure(raw: pd.DataFrame, source_file: str | None = None) -> pd.DataFrame:
    """Parse D+1-available disclosure features and reject a different horizon."""
    required = {"当前日期", "相隔天数", *_DISCLOSURE_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing day-ahead disclosure columns: {missing}")
    result = _base_frame(raw, source_file)
    disclosure_date = pd.to_datetime(raw["当前日期"], errors="coerce").dt.normalize()
    lead_days = pd.to_numeric(raw["相隔天数"], errors="coerce")
    market_date = result["market_date"]
    if disclosure_date.isna().any() or lead_days.isna().any():
        raise ValueError("Invalid current date or lead days in day-ahead disclosure")
    if not (lead_days == 1).all() or not (disclosure_date.to_numpy() == (market_date - pd.Timedelta(days=1)).to_numpy()).all():
        raise ValueError("Only D+1 day-ahead disclosure rows are valid for this forecast contract")
    result["disclosure_date"] = disclosure_date.to_numpy()
    for source, target in _DISCLOSURE_COLUMNS.items():
        values = pd.to_numeric(raw[source], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Day-ahead disclosure column '{source}' contains missing or non-numeric values")
        result[target] = values.to_numpy(dtype=float)
    return result[[*FORECAST_FEATURE_COLUMNS, "market_date", "disclosure_date", "source_file"]].sort_index()


def read_workbook(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two production-safe tables from one monthly source workbook."""
    path = Path(path)
    prices = pd.read_excel(path, sheet_name=REALTIME_SHEET)
    disclosure = pd.read_excel(path, sheet_name=DAY_AHEAD_DISCLOSURE_SHEET)
    return (
        parse_realtime_prices(prices, source_file=path.name),
        parse_day_ahead_disclosure(disclosure, source_file=path.name),
    )


def read_day_ahead_prices(path: str | Path) -> pd.DataFrame:
    """Read target-day 15-minute day-ahead prices from one source workbook."""
    path = Path(path)
    raw = pd.read_excel(path, sheet_name=DAY_AHEAD_PRICE_SHEET)
    return parse_day_ahead_prices(raw, source_file=path.name)


def merge_identical(existing: pd.DataFrame | None, new: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Append new rows while retaining existing provenance for matching overlap."""
    if existing is None or existing.empty:
        return new.sort_index()
    overlap = existing.index.intersection(new.index)
    if not overlap.empty:
        comparison_columns = [column for column in new.columns if column != "source_file"]
        left = existing.loc[overlap, comparison_columns]
        right = new.loc[overlap, comparison_columns]
        if not left.equals(right):
            raise ValueError(f"{label}: source conflict at {overlap[0]}; existing data was preserved")
    return pd.concat([existing, new.loc[~new.index.isin(existing.index)]]).sort_index()
