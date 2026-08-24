"""Read the operator-maintained real-time price workbook.

The historical source uses settlement end-point labels (00:15 ... 24:00),
but operator-maintained sheets in the field are also commonly exported with
an Excel ``24:00`` sentinel as the *first* row followed by 00:15 ... 23:45.
The parser detects that first-row variant and treats it as a period-start
table.  Excel's ``1900-01-01 00:00:00`` sentinel is handled explicitly.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
import re
import sys
import warnings

import pandas as pd

from da_forecast.config import PRICE_COL, PRICE_RANGE_MAX, PRICE_RANGE_MIN, TIMEZONE


DEFAULT_MANUAL_WORKBOOK_NAME = "manual_realtime_prices.xlsx"
MANUAL_SHEET = 0
_TIME_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$")


def _column(raw: pd.DataFrame, names: tuple[str, ...], position: int) -> pd.Series:
    normalized = {str(column).strip().replace(" ", ""): column for column in raw.columns}
    for name in names:
        if name in normalized:
            return raw[normalized[name]]
    if len(raw.columns) > position:
        return raw.iloc[:, position]
    raise ValueError(f"Manual real-time workbook is missing column {names[0]!r}")


def _parse_endpoint(value: object) -> tuple[int, int, bool, pd.Timestamp | None]:
    """Return hour/minute and whether Excel encoded the 24:00 sentinel."""
    if pd.isna(value):
        raise ValueError("Manual real-time workbook contains an empty time")
    if isinstance(value, pd.Timestamp):
        if value.year == 1900 and value.month == 1 and value.day == 1 and value.time() == time(0, 0):
            return 24, 0, True, None
        return value.hour, value.minute, False, pd.Timestamp(value)
    if isinstance(value, datetime):
        if value.year == 1900 and value.month == 1 and value.day == 1 and value.time() == time(0, 0):
            return 24, 0, True, None
        return value.hour, value.minute, False, pd.Timestamp(value)
    if isinstance(value, time):
        return value.hour, value.minute, False, None
    if isinstance(value, timedelta):
        total_seconds = int(round(value.total_seconds()))
        if 0 <= total_seconds <= 24 * 60 * 60:
            if total_seconds == 24 * 60 * 60:
                return 24, 0, True, None
            return (total_seconds // 3600, (total_seconds % 3600) // 60, False, None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel may return a fraction of a day for a time-only cell.  Zero is
        # indistinguishable from 00:00, and is therefore treated as midnight.
        seconds = round(float(value) * 24 * 60 * 60)
        if 0 <= seconds <= 24 * 60 * 60:
            return (seconds // 3600) % 24, (seconds % 3600) // 60, False, None
    match = _TIME_RE.match(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid settlement endpoint time: {value!r}")
    hour, minute = int(match["hour"]), int(match["minute"])
    if hour > 24 or minute not in (0, 15, 30, 45) or (hour == 24 and minute != 0):
        raise ValueError(f"Invalid settlement endpoint time: {value!r}")
    return hour, minute, hour == 24, None


def _local_naive_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(TIMEZONE).tz_localize(None)
    return stamp


def _slot_candidates(
    day: pd.Timestamp,
    *,
    hour: int,
    minute: int,
    sentinel_2400: bool,
    explicit_timestamp: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return endpoint-label and period-start interpretations of one time cell."""
    if explicit_timestamp is not None:
        start = _local_naive_timestamp(explicit_timestamp)
        return start - pd.Timedelta(minutes=15), start
    if sentinel_2400:
        return day + pd.Timedelta(hours=23, minutes=45), day
    start = day + pd.Timedelta(hours=hour, minutes=minute)
    return start - pd.Timedelta(minutes=15), start


def _is_cross_midnight_correction_candidate(hour: int, minute: int) -> bool:
    return (hour, minute) in {(0, 0), (23, 45)}


def _period_start_days_from_layout(dates: pd.Series, times: pd.Series) -> set[pd.Timestamp]:
    """Recognize full period-start days from layout, including blank price rows."""
    layout_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    by_day: dict[pd.Timestamp, list[tuple[int, int, bool]]] = {}
    for day, raw_time in zip(layout_dates, times):
        if pd.isna(day) or pd.isna(raw_time):
            continue
        try:
            hour, minute, sentinel_2400, _ = _parse_endpoint(raw_time)
        except ValueError:
            # A blank-price layout row is not price data and cannot invalidate
            # already entered values.  It simply contributes no layout signal.
            continue
        by_day.setdefault(pd.Timestamp(day), []).append((hour, minute, sentinel_2400))
    result: set[pd.Timestamp] = set()
    for day, labels_with_sentinel in by_day.items():
        if len(labels_with_sentinel) < 96:
            continue
        if labels_with_sentinel[0][2]:
            result.add(day)
            continue
        labels = {(hour, minute) for hour, minute, _ in labels_with_sentinel}
        if len(labels) == 96 and all(
            (hour, minute) in labels
            for hour in range(24)
            for minute in (0, 15, 30, 45)
        ):
            result.add(day)
    return result


def _manual_index(
    dates: pd.Series,
    times: pd.Series,
    *,
    start_mode_days: set[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    market_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if market_dates.isna().any():
        raise ValueError("Manual real-time workbook contains an invalid date")
    rows: list[dict[str, object]] = []
    for (_label, day), raw_time in zip(market_dates.items(), times):
        hour, minute, sentinel_2400, explicit_timestamp = _parse_endpoint(raw_time)
        endpoint, start = _slot_candidates(
            pd.Timestamp(day),
            hour=hour,
            minute=minute,
            sentinel_2400=sentinel_2400,
            explicit_timestamp=explicit_timestamp,
        )
        rows.append(
            {
                "day": pd.Timestamp(day),
                "hour": hour,
                "minute": minute,
                "sentinel_2400": sentinel_2400,
                "endpoint": endpoint,
                "start": start,
            }
        )
    if start_mode_days is None:
        start_mode_days = _period_start_days_from_layout(dates, times)
    slots = [row["start"] if row["day"] in start_mode_days else row["endpoint"] for row in rows]
    for position, row in enumerate(rows):
        hour, minute = int(row["hour"]), int(row["minute"])
        if not _is_cross_midnight_correction_candidate(hour, minute):
            continue
        if 0 < position < len(rows) - 1 and slots[position + 1] - slots[position - 1] == pd.Timedelta(minutes=30):
            slots[position] = slots[position - 1] + pd.Timedelta(minutes=15)
            continue
        if (hour, minute) == (0, 0) and row["day"] not in start_mode_days:
            previous = slots[position - 1].isoformat() if position else "<none>"
            following = slots[position + 1].isoformat() if position < len(rows) - 1 else "<none>"
            raise ValueError(
                "Manual real-time workbook cannot determine cross-midnight slot "
                f"for date={row['day'].date()}, time=00:00; "
                "provide a complete 96-slot period-start layout or filled 15-minute "
                f"records on both sides (previous normalized={previous}, next normalized={following})."
            )
    index = pd.DatetimeIndex(slots)
    return index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)


def parse_manual_realtime_prices(raw: pd.DataFrame, *, source_file: str = "<in-memory>") -> pd.DataFrame:
    """Parse partial or complete operator-entered real-time price rows."""
    dates = _column(raw, ("目标日期", "日期", "交易日", "市场日期"), 0)
    times = _column(raw, ("时刻", "时间", "结算时刻"), 1)
    raw_values = _column(raw, ("实时电价", "实时出清电价", "实时价格", "价格"), 2)
    start_mode_days = _period_start_days_from_layout(dates, times)
    # The operator workbook is intentionally allowed to contain prefilled
    # future rows whose prices are still blank.  They are not manual
    # overrides; leave those slots to the canonical parquet source.
    blank_values = raw_values.isna() | raw_values.astype(str).str.strip().eq("")
    invalid_values = pd.to_numeric(raw_values[~blank_values], errors="coerce").isna()
    if invalid_values.any():
        raise ValueError("Manual real-time workbook contains missing or non-numeric prices")
    keep = ~blank_values
    dates = dates.loc[keep]
    times = times.loc[keep]
    values = pd.to_numeric(raw_values.loc[keep], errors="coerce")
    index = _manual_index(dates, times, start_mode_days=start_mode_days)
    if not values.between(PRICE_RANGE_MIN, PRICE_RANGE_MAX).all():
        raise ValueError(f"Manual real-time prices must be within [{PRICE_RANGE_MIN}, {PRICE_RANGE_MAX}] CNY/MWh")
    result = pd.DataFrame(
        {
            PRICE_COL: values.to_numpy(dtype=float),
            "market_date": index.normalize().tz_localize(None),
            "source_file": source_file,
        },
        index=index,
    )
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated(keep=False)].unique()
        warnings.warn(
            "Manual real-time workbook contains duplicate slot(s) "
            f"{', '.join(value.isoformat() for value in duplicates[:5])}; "
            "keeping the first row for each slot.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = result[~result.index.duplicated(keep="first")]
    return result.sort_index()


def read_manual_realtime_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        raw = pd.read_excel(path, sheet_name=MANUAL_SHEET, engine="openpyxl")
    except ImportError as exc:
        # pandas' native error is technically correct but not actionable in
        # the Streamlit page.  Keep the original exception chained for logs,
        # while exposing the exact interpreter and install commands needed by
        # the local-port launcher.
        if "openpyxl" not in str(exc).lower():
            raise
        interpreter = Path(sys.executable)
        raise ImportError(
            "读取人工实时价格 Excel 需要 openpyxl，但当前 Python 环境未安装。"
            f" 当前解释器：{interpreter}。"
            " 请执行："
            f'uv pip install --python "{interpreter}" openpyxl'
            "（或使用该解释器安装 openpyxl）。"
        ) from exc
    return parse_manual_realtime_prices(raw, source_file=path.name)
