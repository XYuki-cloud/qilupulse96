"""Pure feature masks and metrics for QiluPulse-96 ablation audits.

This module operates on already validated, standardized model inputs.  It does
not load data, call a weather API, write prediction runs, or apply production
calibration.  The CLI owns those boundaries and uses these functions for
deterministic, paired comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .feature_schema_v1 import CALENDAR_COLUMNS, STATION_COLUMNS, STATE_COLUMNS
from .input_builder_v1 import CausalInputBundle


METEOROLOGY_COUNT = 18
SOLAR_GEOMETRY_START = METEOROLOGY_COUNT
CALENDAR_DIM = len(CALENDAR_COLUMNS)
STATE_DIM = len(STATE_COLUMNS)

GROUP_VARIANTS = (
    "full",
    "weather_meteorology_off",
    "weather_solar_geometry_off",
    "weather_all_off",
    "calendar_date_off",
    "calendar_all_off",
    "price_state_off",
)


@dataclass(frozen=True)
class VariableSpec:
    """One leave-one-variable-out intervention."""

    name: str
    group: str
    index: int


def variable_specs() -> tuple[VariableSpec, ...]:
    """Return all weather, calendar, and state variable interventions."""
    return (
        *(VariableSpec(f"weather:{name}", "weather", index) for index, name in enumerate(STATION_COLUMNS)),
        *(VariableSpec(f"calendar:{name}", "calendar", index) for index, name in enumerate(CALENDAR_COLUMNS)),
        *(VariableSpec(f"state:{name}", "state", index) for index, name in enumerate(STATE_COLUMNS)),
    )


def all_variant_names() -> tuple[str, ...]:
    """Return group interventions followed by all leave-one-variable tests."""
    return GROUP_VARIANTS + tuple(spec.name for spec in variable_specs())


def _zero_columns(array: np.ndarray, start: int, stop: int) -> np.ndarray:
    result = np.asarray(array).copy()
    if result.ndim < 2 or result.shape[-1] < stop:
        raise ValueError(f"Cannot mask columns [{start}:{stop}] in array with shape {result.shape}")
    result[..., start:stop] = 0.0
    return result


def _replace(
    inputs: CausalInputBundle,
    *,
    history_extra: np.ndarray | None = None,
    history_station_weather: np.ndarray | None = None,
    target_extra: np.ndarray | None = None,
    target_station_weather: np.ndarray | None = None,
    state_features: np.ndarray | None = None,
) -> CausalInputBundle:
    return replace(
        inputs,
        history_extra=history_extra if history_extra is not None else inputs.history_extra,
        history_station_weather=(
            history_station_weather
            if history_station_weather is not None
            else inputs.history_station_weather
        ),
        target_extra=target_extra if target_extra is not None else inputs.target_extra,
        target_station_weather=(
            target_station_weather
            if target_station_weather is not None
            else inputs.target_station_weather
        ),
        state_features=state_features if state_features is not None else inputs.state_features,
    )


def _mask_weather(inputs: CausalInputBundle, start: int, stop: int) -> CausalInputBundle:
    return _replace(
        inputs,
        history_station_weather=_zero_columns(inputs.history_station_weather, start, stop),
        target_station_weather=_zero_columns(inputs.target_station_weather, start, stop),
    )


def _mask_calendar(inputs: CausalInputBundle, start: int, stop: int) -> CausalInputBundle:
    history_extra = _zero_columns(inputs.history_extra, start, stop)
    target_extra = _zero_columns(inputs.target_extra, start, stop)
    return _replace(inputs, history_extra=history_extra, target_extra=target_extra)


def _mask_state(inputs: CausalInputBundle, start: int, stop: int) -> CausalInputBundle:
    if inputs.target_extra.shape[-1] < CALENDAR_DIM + STATE_DIM:
        raise ValueError("target_extra does not contain the repeated state feature block")
    state_features = np.asarray(inputs.state_features).copy()
    if state_features.ndim != 1 or state_features.shape[0] < stop:
        raise ValueError(f"state_features must contain at least {stop} columns")
    state_features[start:stop] = 0.0
    target_extra = np.asarray(inputs.target_extra).copy()
    target_extra[..., CALENDAR_DIM + start:CALENDAR_DIM + stop] = 0.0
    return _replace(inputs, target_extra=target_extra, state_features=state_features)


def ablate_inputs(inputs: CausalInputBundle, variant: str) -> CausalInputBundle:
    """Apply one named group or variable intervention to model inputs.

    Inputs are already standardized, so zero is the bundle training mean.  The
    original bundle is never mutated; ``full`` returns the original object as a
    zero-cost identity intervention.
    """
    name = str(variant)
    if name == "full":
        return inputs
    if name == "weather_meteorology_off":
        return _mask_weather(inputs, 0, METEOROLOGY_COUNT)
    if name == "weather_solar_geometry_off":
        return _mask_weather(inputs, SOLAR_GEOMETRY_START, len(STATION_COLUMNS))
    if name == "weather_all_off":
        return _mask_weather(inputs, 0, len(STATION_COLUMNS))
    if name == "calendar_date_off":
        return _mask_calendar(inputs, 2, CALENDAR_DIM)
    if name == "calendar_all_off":
        return _mask_calendar(inputs, 0, CALENDAR_DIM)
    if name == "price_state_off":
        return _mask_state(inputs, 0, STATE_DIM)

    group, separator, feature = name.partition(":")
    if not separator or not feature:
        raise ValueError(f"Unknown feature ablation variant: {variant}")
    if group == "weather":
        names = tuple(STATION_COLUMNS)
        if feature not in names:
            raise ValueError(f"Unknown weather variable: {feature}")
        index = names.index(feature)
        return _mask_weather(inputs, index, index + 1)
    if group == "calendar":
        names = tuple(CALENDAR_COLUMNS)
        if feature not in names:
            raise ValueError(f"Unknown calendar variable: {feature}")
        index = names.index(feature)
        return _mask_calendar(inputs, index, index + 1)
    if group == "state":
        names = tuple(STATE_COLUMNS)
        if feature not in names:
            raise ValueError(f"Unknown state variable: {feature}")
        index = names.index(feature)
        return _mask_state(inputs, index, index + 1)
    raise ValueError(f"Unknown feature ablation group: {group}")


def _safe_correlation(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return None
    return float(np.corrcoef(actual, predicted)[0, 1])


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate point, direction, negative-probability and interval metrics."""
    required = {
        "actual_cny_mwh",
        "predicted_cny_mwh",
        "negative_probability",
        "p10_cny_mwh",
        "p90_cny_mwh",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metrics frame is missing columns: {missing}")
    actual = frame["actual_cny_mwh"].to_numpy(dtype=float)
    predicted = frame["predicted_cny_mwh"].to_numpy(dtype=float)
    probability = frame["negative_probability"].to_numpy(dtype=float)
    p10 = frame["p10_cny_mwh"].to_numpy(dtype=float)
    p90 = frame["p90_cny_mwh"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([actual, predicted, probability, p10, p90])).all():
        raise ValueError("Metrics frame contains non-finite values")
    differences = predicted - actual
    if len(actual) >= 2:
        direction_accuracy = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))
    else:
        direction_accuracy = None
    return {
        "slot_count": int(len(frame)),
        "mae_cny_mwh": float(np.mean(np.abs(differences))),
        "rmse_cny_mwh": float(np.sqrt(np.mean(differences**2))),
        "mean_bias_cny_mwh": float(np.mean(differences)),
        "within_day_correlation": _safe_correlation(actual, predicted),
        "direction_accuracy": direction_accuracy,
        "brier_score": float(np.mean((probability - (actual < 0).astype(float)) ** 2)),
        "interval_coverage": float(np.mean((actual >= p10) & (actual <= p90))),
        "mean_interval_width_cny_mwh": float(np.mean(p90 - p10)),
    }


def bootstrap_mean_ci(
    values: Iterable[float] | np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Return a deterministic percentile bootstrap CI for a mean."""
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(int(draws), array.size), replace=True).mean(axis=1)
    return {
        "sample_count": int(array.size),
        "draws": int(draws),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def output_delta_metrics(full: pd.DataFrame, variant: pd.DataFrame) -> dict[str, float]:
    """Summarize how one variant changes the raw model output."""
    required = {"predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p90_cny_mwh"}
    missing_full = required - set(full.columns)
    missing_variant = required - set(variant.columns)
    if missing_full or missing_variant:
        details = []
        if missing_full:
            details.append(f"full={sorted(missing_full)}")
        if missing_variant:
            details.append(f"variant={sorted(missing_variant)}")
        raise ValueError(f"Output frames are missing columns: {'; '.join(details)}")
    if len(full) != len(variant):
        raise ValueError("Output frames must contain the same number of slots")
    point_delta = variant["predicted_cny_mwh"].to_numpy(float) - full["predicted_cny_mwh"].to_numpy(float)
    probability_delta = variant["negative_probability"].to_numpy(float) - full["negative_probability"].to_numpy(float)
    lower_delta = variant["p10_cny_mwh"].to_numpy(float) - full["p10_cny_mwh"].to_numpy(float)
    upper_delta = variant["p90_cny_mwh"].to_numpy(float) - full["p90_cny_mwh"].to_numpy(float)
    return {
        "mean_abs_output_delta_cny_mwh": float(np.mean(np.abs(point_delta))),
        "p95_abs_output_delta_cny_mwh": float(np.quantile(np.abs(point_delta), 0.95)),
        "max_abs_output_delta_cny_mwh": float(np.max(np.abs(point_delta))),
        "mean_signed_output_delta_cny_mwh": float(np.mean(point_delta)),
        "mean_abs_negative_probability_delta": float(np.mean(np.abs(probability_delta))),
        "mean_abs_p10_delta_cny_mwh": float(np.mean(np.abs(lower_delta))),
        "mean_abs_p90_delta_cny_mwh": float(np.mean(np.abs(upper_delta))),
    }


def summarize_variant_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Return overall and per-market-day metrics for one variant."""
    if "market_date" not in frame.columns:
        raise ValueError("Variant frame must contain market_date")
    daily_rows: list[dict[str, object]] = []
    for market_date, day in frame.groupby("market_date", sort=True):
        daily_rows.append({"market_date": str(market_date), **calculate_metrics(day)})
    return {
        "overall": calculate_metrics(frame),
        "daily": pd.DataFrame(daily_rows),
    }


def paired_metric_summary(
    full_daily: pd.DataFrame,
    variant_daily: pd.DataFrame,
    *,
    metric: str,
    draws: int = 10_000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Compare one variant with full predictions using paired daily metrics."""
    required = {"market_date", metric}
    for name, frame in (("full", full_daily), ("variant", variant_daily)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} daily metrics are missing columns: {missing}")
    merged = full_daily[["market_date", metric]].merge(
        variant_daily[["market_date", metric]],
        on="market_date",
        how="inner",
        validate="one_to_one",
        suffixes=("_full", "_variant"),
    )
    if len(merged) != len(full_daily) or len(merged) != len(variant_daily):
        raise ValueError("Full and variant daily metrics must cover the same market dates")
    differences = (
        merged[f"{metric}_variant"].to_numpy(dtype=float)
        - merged[f"{metric}_full"].to_numpy(dtype=float)
    )
    ci = bootstrap_mean_ci(differences, draws=draws, seed=seed)
    return {
        "metric": metric,
        "mean_variant_minus_full": float(np.mean(differences)),
        "worse_days": int(np.sum(differences > 0.0)),
        "better_days": int(np.sum(differences < 0.0)),
        "tie_days": int(np.sum(differences == 0.0)),
        "sample_count": ci["sample_count"],
        "draws": ci["draws"],
        "ci95_low": ci["ci95_low"],
        "ci95_high": ci["ci95_high"],
    }
