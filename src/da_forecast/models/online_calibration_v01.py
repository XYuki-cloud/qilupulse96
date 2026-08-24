"""Causal scheduling helpers and stable post-processing for frozen AdaLN output.

The calibrator intentionally operates on already-issued prediction records. It
never fits a statistical learner, updates Transformer weights, or accesses a
target day's label. Point-bias estimation is made at the *day x period-group*
level, while conformal scores use the immutable, bias-adjusted pre-interval
quantiles saved when each earlier prediction was issued.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from da_forecast.config import TIMEZONE


CALIBRATION_VERSION = "frozen_adaln_bias_interval_v02"
RAW_COLUMNS = {
    "predicted_cny_mwh": "raw_predicted_cny_mwh",
    "p10_cny_mwh": "raw_p10_cny_mwh",
    "p50_cny_mwh": "raw_p50_cny_mwh",
    "p90_cny_mwh": "raw_p90_cny_mwh",
}
BIAS_COLUMNS = {
    "predicted_cny_mwh": "bias_predicted_cny_mwh",
    "p10_cny_mwh": "bias_p10_cny_mwh",
    "p50_cny_mwh": "bias_p50_cny_mwh",
    "p90_cny_mwh": "bias_p90_cny_mwh",
}
PERIOD_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 23),
    ("morning", 24, 35),
    ("solar_midday", 36, 59),
    ("evening_peak", 60, 83),
    ("late_night", 84, 95),
)


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize(TIMEZONE) if timestamp.tz is None else timestamp.tz_convert(TIMEZONE)
    return timestamp.normalize()


def checkpoint_date_for_target(
    target_date: str | pd.Timestamp, initial_checkpoint_date: str | pd.Timestamp,
) -> pd.Timestamp:
    """Return the latest causal Monday checkpoint available for one target day."""
    target = _local_day(target_date)
    initial = _local_day(initial_checkpoint_date)
    decision_day = target - pd.Timedelta(days=1)
    monday = decision_day - pd.Timedelta(days=decision_day.weekday())
    return max(initial, monday)


def time_decay_weights(
    dates: pd.DatetimeIndex, cutoff_date: str | pd.Timestamp, *, half_life_days: float,
) -> np.ndarray:
    """Return exponential weights whose newest admissible date has weight one."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    cutoff = _local_day(cutoff_date)
    local_dates = pd.DatetimeIndex([_local_day(value) for value in dates])
    ages = np.maximum(0.0, (cutoff - local_dates).total_seconds() / 86_400.0)
    return np.exp(-np.log(2.0) * ages / float(half_life_days))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a deterministic weighted quantile for finite, positive-weight data."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("values and weights must be equally sized one-dimensional arrays")
    if not 0.0 <= quantile <= 1.0 or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("quantile values and weights must be finite")
    positive = weights > 0.0
    if not positive.any():
        raise ValueError("quantile weights must contain a positive value")
    order = np.argsort(values[positive], kind="mergesort")
    sorted_values = values[positive][order]
    sorted_weights = weights[positive][order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def _slot_numbers(frame: pd.DataFrame) -> np.ndarray:
    value = frame["period_start"].astype(str)
    parsed = pd.to_datetime(value, format="%H:%M", errors="coerce")
    if parsed.isna().any():
        raise ValueError("period_start must contain HH:MM values")
    minutes = parsed.dt.hour.to_numpy(dtype=int) * 60 + parsed.dt.minute.to_numpy(dtype=int)
    if (minutes % 15 != 0).any() or (minutes < 0).any() or (minutes >= 24 * 60).any():
        raise ValueError("period_start must be a 15-minute slot from 00:00 through 23:45")
    return minutes // 15


def _period_group(slots: np.ndarray) -> np.ndarray:
    group = np.empty(len(slots), dtype=object)
    for name, start, end in PERIOD_GROUPS:
        group[(slots >= start) & (slots <= end)] = name
    if pd.isna(group).any():
        raise ValueError("Could not assign every slot to a period group")
    return group


def _require_columns(frame: pd.DataFrame, columns: set[str], *, subject: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{subject} is missing required columns: {sorted(missing)}")


def _with_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    _require_columns(result, set(RAW_COLUMNS), subject="Prediction")
    for final_column, raw_column in RAW_COLUMNS.items():
        if raw_column not in result:
            result[raw_column] = result[final_column].to_numpy(dtype=float)
        if not np.isfinite(result[raw_column].to_numpy(dtype=float)).all():
            raise ValueError(f"{raw_column} must be finite")
    return result


def _complete_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates with exactly one finite record per 15-minute slot."""
    complete: list[pd.DataFrame] = []
    for _, day in frame.groupby("_market_day", sort=True):
        slots = day["_slot"].to_numpy(dtype=int)
        if len(day) != 96 or len(np.unique(slots)) != 96 or set(slots) != set(range(96)):
            continue
        numeric = day[["actual_cny_mwh", "raw_predicted_cny_mwh", "bias_p10_cny_mwh", "bias_p90_cny_mwh"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            continue
        complete.append(day)
    return pd.concat(complete, ignore_index=True) if complete else frame.iloc[0:0].copy()


def _day_group_residuals(frame: pd.DataFrame, name: str, start: int, end: int) -> pd.DataFrame:
    expected_slots = set(range(start, end + 1))
    rows: list[dict[str, object]] = []
    for day, group in frame.loc[frame["_period_group"] == name].groupby("_market_day", sort=True):
        slots = group["_slot"].to_numpy(dtype=int)
        if len(group) != len(expected_slots) or set(slots) != expected_slots:
            continue
        residual = group["actual_cny_mwh"].to_numpy(dtype=float) - group["raw_predicted_cny_mwh"].to_numpy(dtype=float)
        if np.isfinite(residual).all():
            rows.append({"market_day": day, "daily_residual": float(np.mean(residual))})
    return pd.DataFrame(rows, columns=["market_day", "daily_residual"])


@dataclass(frozen=True)
class CalibrationConfig:
    window_days: int = 56
    min_days: int = 14
    half_life_days: float = 28.0
    long_weight: float = 0.65
    recent_weight: float = 0.35
    robust_scale_floor: float = 20.0
    bias_clip_cny_mwh: float = 100.0


class CausalResidualCalibrator:
    """Apply causal day-group bias correction and optional conformal expansion."""

    def __init__(
        self,
        *,
        window_days: int = 56,
        min_days: int = 14,
        half_life_days: float = 28.0,
        long_weight: float = 0.65,
        recent_weight: float = 0.35,
        robust_scale_floor: float = 20.0,
        bias_clip_cny_mwh: float = 100.0,
        enable_interval: bool = True,
    ) -> None:
        if window_days < 1 or min_days < 1 or min_days > window_days:
            raise ValueError("Require 1 <= min_days <= window_days")
        if half_life_days <= 0.0 or robust_scale_floor <= 0.0 or bias_clip_cny_mwh <= 0.0:
            raise ValueError("half_life_days, robust_scale_floor, and bias_clip_cny_mwh must be positive")
        if long_weight < 0.0 or recent_weight < 0.0 or not np.isclose(long_weight + recent_weight, 1.0):
            raise ValueError("long_weight and recent_weight must be non-negative and sum to one")
        self.config = CalibrationConfig(
            window_days=window_days,
            min_days=min_days,
            half_life_days=half_life_days,
            long_weight=long_weight,
            recent_weight=recent_weight,
            robust_scale_floor=robust_scale_floor,
            bias_clip_cny_mwh=bias_clip_cny_mwh,
        )
        self.enable_interval = enable_interval

    def _settled_history(self, history: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
        _require_columns(
            history,
            {"market_date", "period_start", "actual_cny_mwh", "predicted_cny_mwh", "p10_cny_mwh", "p90_cny_mwh"},
            subject="Calibration history",
        )
        frame = _with_raw_columns(history)
        for source, bias_column in BIAS_COLUMNS.items():
            if bias_column not in frame:
                frame[bias_column] = frame[RAW_COLUMNS[source]].to_numpy(dtype=float)
        frame["_market_day"] = pd.to_datetime(frame["market_date"]).dt.normalize()
        frame["_slot"] = _slot_numbers(frame)
        frame["_period_group"] = _period_group(frame["_slot"].to_numpy(dtype=int))
        cutoff = (target - pd.Timedelta(days=2)).tz_localize(None)
        frame = frame.loc[frame["_market_day"] <= cutoff].copy()
        frame = _complete_days(frame)
        days = sorted(frame["_market_day"].unique())[-self.config.window_days :]
        return frame.loc[frame["_market_day"].isin(days)].copy()

    def _bias_for_group(
        self,
        settled: pd.DataFrame,
        *,
        name: str,
        start: int,
        end: int,
        target: pd.Timestamp,
    ) -> dict[str, object]:
        daily = _day_group_residuals(settled, name, start, end)
        recent_day = (target - pd.Timedelta(days=2)).tz_localize(None)
        if len(daily) < self.config.min_days or recent_day not in set(daily["market_day"]):
            return {
                "bias_status": "insufficient_history_fallback",
                "bias_correction_cny_mwh": 0.0,
                "bias_history_days": int(len(daily)),
            }
        dates = pd.DatetimeIndex(pd.to_datetime(daily["market_day"]))
        weights = time_decay_weights(dates, target - pd.Timedelta(days=2), half_life_days=self.config.half_life_days)
        residuals = daily["daily_residual"].to_numpy(dtype=float)
        center = _weighted_quantile(residuals, weights, 0.5)
        mad = _weighted_quantile(np.abs(residuals - center), weights, 0.5)
        scale = max(float(mad), self.config.robust_scale_floor)
        clipped = np.clip(residuals, center - 3.0 * scale, center + 3.0 * scale)
        long_bias = float(np.average(clipped, weights=weights))
        recent_bias = float(daily.loc[daily["market_day"] == recent_day, "daily_residual"].iloc[0])
        recent_bias = float(np.clip(recent_bias, long_bias - 2.5 * scale, long_bias + 2.5 * scale))
        correction = float(np.clip(
            self.config.long_weight * long_bias + self.config.recent_weight * recent_bias,
            -self.config.bias_clip_cny_mwh,
            self.config.bias_clip_cny_mwh,
        ))
        return {
            "bias_status": "active",
            "bias_correction_cny_mwh": correction,
            "bias_history_days": int(len(daily)),
        }

    def _interval_for_group(self, settled: pd.DataFrame, *, name: str, target: pd.Timestamp) -> dict[str, object]:
        group = settled.loc[settled["_period_group"] == name].copy()
        if group["_market_day"].nunique() >= self.config.min_days:
            return self._interval_quantiles(group, target, "active")
        if settled["_market_day"].nunique() >= self.config.min_days:
            return self._interval_quantiles(settled, target, "full_day_fallback")
        return {
            "interval_status": "insufficient_history_fallback",
            "interval_history_days": int(group["_market_day"].nunique()),
            "interval_lower_expansion_cny_mwh": 0.0,
            "interval_upper_expansion_cny_mwh": 0.0,
        }

    def _interval_quantiles(self, frame: pd.DataFrame, target: pd.Timestamp, status: str) -> dict[str, object]:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["_market_day"]))
        day_weights = time_decay_weights(dates, target - pd.Timedelta(days=2), half_life_days=self.config.half_life_days)
        slot_counts = frame.groupby("_market_day")["_slot"].transform("count").to_numpy(dtype=float)
        weights = day_weights / slot_counts
        lower = np.maximum(
            frame["bias_p10_cny_mwh"].to_numpy(dtype=float) - frame["actual_cny_mwh"].to_numpy(dtype=float), 0.0,
        )
        upper = np.maximum(
            frame["actual_cny_mwh"].to_numpy(dtype=float) - frame["bias_p90_cny_mwh"].to_numpy(dtype=float), 0.0,
        )
        return {
            "interval_status": status,
            "interval_history_days": int(frame["_market_day"].nunique()),
            "interval_lower_expansion_cny_mwh": _weighted_quantile(lower, weights, 0.90),
            "interval_upper_expansion_cny_mwh": _weighted_quantile(upper, weights, 0.90),
        }

    def calibrate(
        self, prediction: pd.DataFrame, history: pd.DataFrame, *, target_date: str | pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Calibrate one 96-slot future prediction with labels settled no later than T-2."""
        _require_columns(
            prediction,
            {"market_date", "period_start", "predicted_cny_mwh", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh", "negative_probability"},
            subject="Prediction",
        )
        target = _local_day(target_date)
        result = _with_raw_columns(prediction)
        result["_slot"] = _slot_numbers(result)
        if len(result) != 96 or len(np.unique(result["_slot"])) != 96 or set(result["_slot"]) != set(range(96)):
            raise ValueError("Prediction must contain exactly one record for every 15-minute slot")
        if not result["negative_probability"].between(0.0, 1.0).all():
            raise ValueError("negative_probability must remain within [0, 1]")
        result["bias_group"] = _period_group(result["_slot"].to_numpy(dtype=int))
        settled = self._settled_history(history, target)
        history_last = None if settled.empty else pd.Timestamp(settled["_market_day"].max()).strftime("%Y-%m-%d")
        label_cutoff = target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
        metadata: dict[str, object] = {
            "bias_interval_calibration_version": CALIBRATION_VERSION,
            "calibration_window_days": self.config.window_days,
            "calibration_half_life_days": self.config.half_life_days,
            "calibration_history_days": int(settled["_market_day"].nunique()),
            "calibration_history_last_date": history_last,
            "calibration_realtime_label_cutoff": label_cutoff.strftime("%Y-%m-%d %H:%M %Z"),
        }

        group_bias: dict[str, dict[str, object]] = {}
        for name, start, end in PERIOD_GROUPS:
            group_bias[name] = self._bias_for_group(settled, name=name, start=start, end=end, target=target)
        result["bias_status"] = result["bias_group"].map(lambda name: group_bias[name]["bias_status"])
        result["bias_correction_cny_mwh"] = result["bias_group"].map(lambda name: group_bias[name]["bias_correction_cny_mwh"]).astype(float)
        result["bias_history_days"] = result["bias_group"].map(lambda name: group_bias[name]["bias_history_days"]).astype(int)

        for final_column, raw_column in RAW_COLUMNS.items():
            bias_column = BIAS_COLUMNS[final_column]
            result[bias_column] = result[raw_column].to_numpy(dtype=float) + result["bias_correction_cny_mwh"].to_numpy(dtype=float)
        result["predicted_cny_mwh"] = result["bias_predicted_cny_mwh"].to_numpy(dtype=float)
        result["p10_cny_mwh"] = result["bias_p10_cny_mwh"].to_numpy(dtype=float)
        result["p50_cny_mwh"] = result["bias_p50_cny_mwh"].to_numpy(dtype=float)
        result["p90_cny_mwh"] = result["bias_p90_cny_mwh"].to_numpy(dtype=float)

        if not self.enable_interval:
            result["interval_status"] = "disabled"
            result["interval_history_days"] = 0
            result["interval_lower_expansion_cny_mwh"] = 0.0
            result["interval_upper_expansion_cny_mwh"] = 0.0
            metadata["calibration_status"] = (
                "bias_active_interval_disabled"
                if (result["bias_status"] == "active").all()
                else "insufficient_history_fallback"
            )
        else:
            group_interval = {name: self._interval_for_group(settled, name=name, target=target) for name, _, _ in PERIOD_GROUPS}
            result["interval_status"] = result["bias_group"].map(lambda name: group_interval[name]["interval_status"])
            result["interval_history_days"] = result["bias_group"].map(lambda name: group_interval[name]["interval_history_days"]).astype(int)
            result["interval_lower_expansion_cny_mwh"] = result["bias_group"].map(
                lambda name: group_interval[name]["interval_lower_expansion_cny_mwh"]
            ).astype(float)
            result["interval_upper_expansion_cny_mwh"] = result["bias_group"].map(
                lambda name: group_interval[name]["interval_upper_expansion_cny_mwh"]
            ).astype(float)
            result["p10_cny_mwh"] = (
                result["bias_p10_cny_mwh"].to_numpy(dtype=float) - result["interval_lower_expansion_cny_mwh"].to_numpy(dtype=float)
            )
            result["p90_cny_mwh"] = (
                result["bias_p90_cny_mwh"].to_numpy(dtype=float) + result["interval_upper_expansion_cny_mwh"].to_numpy(dtype=float)
            )
            if (result["bias_status"] == "insufficient_history_fallback").all() and (
                result["interval_status"] == "insufficient_history_fallback"
            ).all():
                metadata["calibration_status"] = "insufficient_history_fallback"
            elif (result["bias_status"] == "active").all() and (result["interval_status"] == "active").all():
                metadata["calibration_status"] = "active"
            else:
                metadata["calibration_status"] = "partial_fallback"

        quantiles = result[["p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"]].to_numpy(dtype=float)
        lower = np.min(quantiles, axis=1)
        upper = np.max(quantiles, axis=1)
        result["p10_cny_mwh"] = lower
        result["p50_cny_mwh"] = np.clip(quantiles[:, 1], lower, upper)
        result["p90_cny_mwh"] = upper
        if not np.isfinite(result[["predicted_cny_mwh", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"]].to_numpy(dtype=float)).all():
            raise ValueError("Calibration produced non-finite price values")
        return result.drop(columns=["_slot"]), metadata
