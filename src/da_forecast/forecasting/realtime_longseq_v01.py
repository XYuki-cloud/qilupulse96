"""No-leakage continuous price context for the D-noon -> D+1 contract."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from da_forecast.config import PRICE_COL, TIMEZONE
from da_forecast.features.calendar_v01 import build_calendar_v01
from da_forecast.sources.shandong_market_xlsx import DAY_AHEAD_PRICE_COL


SLOTS_PER_DAY = 96

# Current research profile.  Legacy T-2 contracts remain available below for
# reproducibility and regression comparisons.
MAINLINE_MODEL_ID = "robust_adaln_tminus1_11"
MAINLINE_CONTRACT_VERSION = "realtime_tminus1_1100_endpoint_v1"


@dataclass(frozen=True)
class V01Contract:
    as_of: pd.Timestamp
    target_date: pd.Timestamp
    realtime_cutoff: pd.Timestamp
    day_ahead_cutoff: pd.Timestamp


@dataclass(frozen=True)
class TMinus1NoonPriceVisibilityContract:
    """Price visibility for a T-1 noon decision using the 11:00 source endpoint."""

    as_of: pd.Timestamp
    target_date: pd.Timestamp
    realtime_source_endpoint: pd.Timestamp
    realtime_cutoff: pd.Timestamp
    day_ahead_cutoff: pd.Timestamp
    contract_version: str = "realtime_tminus1_1100_endpoint_v1"


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(TIMEZONE) if timestamp.tz is None else timestamp.tz_convert(TIMEZONE)


def validate_v01_contract(*, target_date: str | pd.Timestamp, as_of: str | pd.Timestamp) -> V01Contract:
    target = _local_timestamp(target_date).normalize()
    decision = _local_timestamp(as_of)
    expected_decision_day = target - pd.Timedelta(days=1)
    if decision.normalize() != expected_decision_day or decision < expected_decision_day + pd.Timedelta(hours=12):
        raise ValueError("v0.1 requires D 12:00-or-later Asia/Shanghai for a D+1 target")
    cutoff = target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
    return V01Contract(
        as_of=decision,
        target_date=target,
        realtime_cutoff=cutoff,
        day_ahead_cutoff=cutoff,
    )


def validate_tminus1_1100_contract(
    *, target_date: str | pd.Timestamp, as_of: str | pd.Timestamp
) -> TMinus1NoonPriceVisibilityContract:
    """Validate the T-1 noon contract with a right-endpoint 11:00 RT source value."""
    target = _local_timestamp(target_date).normalize()
    decision = _local_timestamp(as_of)
    decision_day = target - pd.Timedelta(days=1)
    if decision.normalize() != decision_day or decision < decision_day + pd.Timedelta(hours=12):
        raise ValueError("T-1 11:00 visibility requires T-1 12:00-or-later Asia/Shanghai")
    source_endpoint = decision_day + pd.Timedelta(hours=11)
    return TMinus1NoonPriceVisibilityContract(
        as_of=decision,
        target_date=target,
        realtime_source_endpoint=source_endpoint,
        # The source uses right endpoints, while the local 15-minute index uses starts.
        realtime_cutoff=source_endpoint - pd.Timedelta(minutes=15),
        day_ahead_cutoff=target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45),
    )


def build_target_price_context(
    prices: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
    context_days: int,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, V01Contract]:
    """Return a continuous history ending exactly at the contract cutoff."""
    if PRICE_COL not in prices:
        raise ValueError(f"Expected '{PRICE_COL}' in prices")
    if context_days < 1:
        raise ValueError("context_days must be positive")
    contract = validate_v01_contract(target_date=target_date, as_of=as_of)
    history_end = contract.realtime_cutoff
    history_index = pd.date_range(
        history_end - pd.Timedelta(days=context_days) + pd.Timedelta(minutes=15),
        periods=context_days * SLOTS_PER_DAY,
        freq="15min",
        tz=TIMEZONE,
    )
    history = prices.sort_index().reindex(history_index)
    if history[PRICE_COL].isna().any():
        raise ValueError(f"Incomplete continuous price history through {history_end}")
    target_index = pd.date_range(contract.target_date, periods=SLOTS_PER_DAY, freq="15min", tz=TIMEZONE)
    return history, target_index, contract


def build_target_price_context_tminus1_1100(
    prices: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
    context_days: int,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, TMinus1NoonPriceVisibilityContract]:
    """Return the visible continuous RT history for the T-1 noon experiment."""
    if PRICE_COL not in prices:
        raise ValueError(f"Expected '{PRICE_COL}' in prices")
    if context_days < 1:
        raise ValueError("context_days must be positive")
    contract = validate_tminus1_1100_contract(target_date=target_date, as_of=as_of)
    history_index = pd.date_range(
        contract.realtime_cutoff - pd.Timedelta(days=context_days) + pd.Timedelta(minutes=15),
        periods=context_days * SLOTS_PER_DAY,
        freq="15min",
        tz=TIMEZONE,
    )
    history = prices.sort_index().reindex(history_index)
    if history[PRICE_COL].isna().any():
        raise ValueError(f"Incomplete continuous price history through {contract.realtime_cutoff}")
    target_index = pd.date_range(contract.target_date, periods=SLOTS_PER_DAY, freq="15min", tz=TIMEZONE)
    return history, target_index, contract


def _daily_statistics(series: pd.Series) -> pd.DataFrame:
    local = pd.DataFrame({"value": series})
    grouped = local.groupby(local.index.normalize())["value"]
    return pd.DataFrame(
        {
            "lag2_day_mean": grouped.mean(),
            "lag2_day_min": grouped.min(),
            "lag2_day_max": grouped.max(),
            "lag2_day_negative_share": grouped.apply(lambda values: (values < 0).mean()),
        }
    )


def build_xgb_price_features(
    prices: pd.DataFrame,
    day_ahead: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, V01Contract]:
    """Build 96 target rows without target-day or T-1 day-ahead prices."""
    if PRICE_COL not in prices or DAY_AHEAD_PRICE_COL not in day_ahead:
        raise ValueError("Expected real-time and day-ahead price columns")
    contract = validate_v01_contract(target_date=target_date, as_of=as_of)
    index = pd.date_range(contract.target_date, periods=SLOTS_PER_DAY, freq="15min", tz=TIMEZONE)
    known_rt = prices.sort_index().loc[: contract.realtime_cutoff, PRICE_COL]
    known_da = day_ahead.sort_index().loc[: contract.day_ahead_cutoff, DAY_AHEAD_PRICE_COL]
    result = pd.DataFrame(index=index)
    for days in (2, 3, 7):
        result[f"realtime_lag_{days}d"] = known_rt.reindex(index - pd.Timedelta(days=days)).to_numpy()
        result[f"day_ahead_lag_{days}d"] = known_da.reindex(index - pd.Timedelta(days=days)).to_numpy()
    result["rt_da_spread_lag_2d"] = result["realtime_lag_2d"] - result["day_ahead_lag_2d"]
    source_day = (index - pd.Timedelta(days=2)).normalize()
    daily = _daily_statistics(known_rt)
    for column in daily:
        result[column] = daily[column].reindex(source_day).to_numpy()
    result = result.join(build_calendar_v01(index))
    if result.isna().any().any():
        raise ValueError(f"Incomplete T-2/T-3/T-7 price history for {contract.target_date.date()}")
    return result, contract


def build_xgb_history_price_features(prices: pd.DataFrame, day_ahead: pd.DataFrame) -> pd.DataFrame:
    """Build historical rows under the same T-2 price and day-ahead availability rule."""
    if PRICE_COL not in prices or DAY_AHEAD_PRICE_COL not in day_ahead:
        raise ValueError("Expected real-time and day-ahead price columns")
    price = prices.sort_index()[PRICE_COL]
    da = day_ahead.sort_index()[DAY_AHEAD_PRICE_COL]
    result = pd.DataFrame({PRICE_COL: price})
    for days in (2, 3, 7):
        result[f"realtime_lag_{days}d"] = price.shift(days * SLOTS_PER_DAY)
        result[f"day_ahead_lag_{days}d"] = da.reindex(result.index).shift(days * SLOTS_PER_DAY)
    result["rt_da_spread_lag_2d"] = result["realtime_lag_2d"] - result["day_ahead_lag_2d"]
    daily = _daily_statistics(price)
    source_day = (result.index - pd.Timedelta(days=2)).normalize()
    for column in daily:
        result[column] = daily[column].reindex(source_day).to_numpy()
    return result.join(build_calendar_v01(result.index))
