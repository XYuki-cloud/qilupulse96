"""Causal, formula-driven explanations for the Shandong Province Forecast System."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from da_forecast.features.calendar_v01 import build_calendar_v01


SYSTEM_NAME = "山东省预测系统"
PERIOD_GROUPS: dict[str, tuple[int, int]] = {
    "night": (0, 23),
    "morning": (24, 35),
    "solar_midday": (36, 59),
    "evening_peak": (60, 83),
    "late_night": (84, 95),
}
WEATHER_VARIABLES = ("temperature_2m", "shortwave_radiation", "cloud_cover", "wind_speed_100m")
EPS = 1e-6


@dataclass(frozen=True)
class ExplanationReport:
    payload: dict[str, Any]
    markdown: str


class WhiteBoxExplainer:
    """Generate a read-only explanation from causal data and issued predictions."""

    def __init__(self, *, bootstrap_draws: int = 2_000, random_seed: int = 7) -> None:
        if bootstrap_draws < 1:
            raise ValueError("bootstrap_draws must be positive")
        self.bootstrap_draws = int(bootstrap_draws)
        self.random_seed = int(random_seed)

    def explain(
        self,
        *,
        target_date: str | pd.Timestamp,
        as_of: str | pd.Timestamp,
        price_history: pd.DataFrame,
        observed_weather: pd.DataFrame,
        forecast_weather: pd.DataFrame,
        prediction: pd.DataFrame,
        data_snapshot_hash: str,
        causal_history_label_cutoff: str | pd.Timestamp | None = None,
    ) -> ExplanationReport:
        target = _local_timestamp(target_date).normalize()
        decision = _local_timestamp(as_of)
        default_cutoff = target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
        cutoff = _local_timestamp(causal_history_label_cutoff) if causal_history_label_cutoff is not None else default_cutoff
        decision_day = target - pd.Timedelta(days=1)
        if decision.normalize() != decision_day or decision < decision_day + pd.Timedelta(hours=12):
            raise ValueError("Explanation requires a T-1 12:00-or-later decision timestamp")
        if cutoff > decision:
            raise ValueError("Explanation history cutoff must not be after the decision timestamp")
        prices = _prepare_price_history(price_history, cutoff=cutoff)
        observed = _prepare_weather(observed_weather, cutoff=cutoff, label="observed_weather")
        forecast = _prepare_weather(forecast_weather, cutoff=None, label="forecast_weather")
        prediction_frame = _prepare_prediction(prediction, target=target)
        if forecast.empty or not forecast["timestamp"].dt.normalize().eq(target).all():
            raise ValueError("forecast_weather must contain exactly the target-date forecast")
        market_state = _market_state(prices)
        calendar = _calendar_summary(target)
        historical = _merge_causal_history(prices, observed)
        reference_dates, reference_status = _reference_dates(historical, calendar=calendar)
        groups = self._period_groups(
            historical=historical,
            forecast=forecast,
            prediction=prediction_frame,
            reference_dates=reference_dates,
            reference_status=reference_status,
        )
        prediction_checksum = _prediction_checksum(prediction_frame)
        checksum = (
            str(prediction_frame["parameter_checksum"].iloc[0])
            if "parameter_checksum" in prediction_frame
            else prediction_checksum
        )
        payload: dict[str, Any] = {
            "system_name": SYSTEM_NAME,
            "explanation_version": "whitebox_v1.1",
            "target_date": target.strftime("%Y-%m-%d"),
            "as_of": decision.isoformat(),
            "causal_history_label_cutoff": cutoff.isoformat(),
            "data_snapshot_hash": str(data_snapshot_hash),
            "prediction_checksum": checksum,
            "market_state": market_state,
            "calendar": calendar,
            "reference_dates": {
                "count": int(len(reference_dates)),
                "status": reference_status,
                "last_date": reference_dates.max().strftime("%Y-%m-%d") if len(reference_dates) else None,
            },
            "period_groups": groups,
        }
        payload["claims"] = _build_claims(
            market_state=market_state,
            calendar=calendar,
            groups=groups,
            data_snapshot_hash=str(data_snapshot_hash),
            prediction_checksum=checksum,
        )
        return ExplanationReport(payload=_jsonable(payload), markdown=_render_markdown(payload))

    def _period_groups(
        self,
        *,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        prediction: pd.DataFrame,
        reference_dates: pd.DatetimeIndex,
        reference_status: str,
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for group, (start, end) in PERIOD_GROUPS.items():
            reference = historical[
                historical["market_day"].isin(reference_dates)
                & historical["slot"].between(start, end)
            ].copy()
            target_weather = forecast[forecast["slot"].between(start, end)].copy()
            target_prediction = prediction[prediction["slot"].between(start, end)].copy()
            weather_summary = {
                variable: float(target_weather[variable].mean())
                for variable in WEATHER_VARIABLES
                if variable in target_weather
            }
            z_scores = {
                variable: _weather_z(reference, variable, weather_summary[variable])
                for variable in weather_summary
            }
            associations = {
                variable: self._association(reference, variable, group)
                for variable in weather_summary
            }
            result[group] = {
                "slot_start": start,
                "slot_end": end,
                "reference_status": reference_status,
                "reference_days": int(reference["market_day"].nunique()),
                "forecast_summary": weather_summary,
                "forecast_z_vs_reference": z_scores,
                "weather_price_associations": associations,
                "reference_price_median_cny_mwh": float(reference["value"].median()) if not reference.empty else None,
                "reference_price_robust_scale_cny_mwh": _robust_scale(reference["value"]) if not reference.empty else None,
                "prediction_summary": {
                    "mean_predicted_cny_mwh": float(target_prediction["predicted_cny_mwh"].mean()),
                    "max_negative_probability": float(target_prediction["negative_probability"].max()),
                    "mean_interval_width_cny_mwh": float(
                        (target_prediction["p90_cny_mwh"] - target_prediction["p10_cny_mwh"]).mean()
                    ),
                },
            }
        return result

    def _association(self, reference: pd.DataFrame, variable: str, group: str) -> dict[str, Any]:
        usable = reference[["market_day", "value", variable]].dropna()
        daily = usable.groupby("market_day", sort=True).agg(price=("value", "mean"), weather=(variable, "mean"))
        if len(daily) < 20 or daily["weather"].nunique() < 4:
            return {
                "formula_name": "high_low_quartile_median_difference_v1",
                "status": "insufficient_stable_history",
                "sample_days": int(len(daily)),
                "effect_cny_mwh": None,
                "ci95_low": None,
                "ci95_high": None,
            }
        effect = _high_low_effect(daily)
        seed = int.from_bytes(sha256(f"{self.random_seed}:{group}:{variable}".encode()).digest()[:8], "little")
        ci_low, ci_high = _bootstrap_effect_ci(daily, draws=self.bootstrap_draws, seed=seed)
        status = "stable" if ci_low > 0 or ci_high < 0 else "uncertain"
        return {
            "formula_name": "high_low_quartile_median_difference_v1",
            "status": status,
            "sample_days": int(len(daily)),
            "effect_cny_mwh": float(effect),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
        }


def _prepare_price_history(frame: pd.DataFrame, *, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = {"timestamp", "value"}
    if not required.issubset(frame):
        raise ValueError(f"price_history must contain {sorted(required)}")
    result = frame[["timestamp", "value"]].copy()
    result["timestamp"] = _timestamp_series(result["timestamp"])
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if result["value"].isna().any():
        raise ValueError("price_history values must be finite")
    result = result[result["timestamp"] <= cutoff].copy()
    result["market_day"] = result["timestamp"].dt.normalize()
    result["slot"] = result["timestamp"].dt.hour * 4 + result["timestamp"].dt.minute // 15
    valid_days = result.groupby("market_day")["slot"].agg(lambda values: len(values) == 96 and set(values) == set(range(96)))
    result = result[result["market_day"].isin(valid_days[valid_days].index)]
    if result.empty:
        raise ValueError("No complete causal price days available for explanation")
    return result.sort_values("timestamp", ignore_index=True)


def _prepare_weather(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None, label: str) -> pd.DataFrame:
    required = {"timestamp", "station_code"}
    if not required.issubset(frame):
        raise ValueError(f"{label} must contain {sorted(required)}")
    result = frame.copy()
    result["timestamp"] = _timestamp_series(result["timestamp"])
    for variable in WEATHER_VARIABLES:
        if variable not in result:
            result[variable] = np.nan
        result[variable] = pd.to_numeric(result[variable], errors="coerce")
    if cutoff is not None:
        result = result[result["timestamp"] <= cutoff].copy()
    result["market_day"] = result["timestamp"].dt.normalize()
    result["slot"] = result["timestamp"].dt.hour * 4 + result["timestamp"].dt.minute // 15
    return result.sort_values(["timestamp", "station_code"], ignore_index=True)


def _prepare_prediction(frame: pd.DataFrame, *, target: pd.Timestamp) -> pd.DataFrame:
    required = {"market_date", "period_start", "predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"}
    if not required.issubset(frame):
        raise ValueError(f"prediction must contain {sorted(required)}")
    result = frame.copy()
    date = pd.to_datetime(result["market_date"], errors="coerce")
    slot_time = pd.to_datetime(result["period_start"].astype(str), format="%H:%M", errors="coerce")
    if date.isna().any() or slot_time.isna().any() or not date.dt.normalize().eq(target.tz_localize(None)).all():
        raise ValueError("prediction must be a single 96-slot target date")
    result["slot"] = slot_time.dt.hour * 4 + slot_time.dt.minute // 15
    if len(result) != 96 or set(result["slot"]) != set(range(96)):
        raise ValueError("prediction must contain exactly 96 unique 15-minute slots")
    numeric = ["predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("prediction values must be finite")
    if not result["negative_probability"].between(0, 1).all():
        raise ValueError("prediction negative_probability must be within [0, 1]")
    if not ((result["p10_cny_mwh"] <= result["p50_cny_mwh"]) & (result["p50_cny_mwh"] <= result["p90_cny_mwh"])).all():
        raise ValueError("prediction quantiles must be monotonic")
    return result.sort_values("slot", ignore_index=True)


def _merge_causal_history(prices: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    weather = observed.groupby("timestamp", sort=True)[list(WEATHER_VARIABLES)].mean().reset_index()
    return prices.merge(weather, on="timestamp", how="left")


def _reference_dates(history: pd.DataFrame, *, calendar: dict[str, Any]) -> tuple[pd.DatetimeIndex, str]:
    all_dates = pd.DatetimeIndex(history["market_day"].drop_duplicates().sort_values())
    recent = all_dates[-56:]
    reference_calendar = _calendar_frame(recent)
    same_type = recent[reference_calendar["day_type"].to_numpy() == calendar["day_type"]]
    if len(same_type) >= 14:
        return same_type, "calendar_matched"
    return recent, "recent_all_days_fallback"


def _market_state(prices: pd.DataFrame) -> dict[str, Any]:
    daily = prices.groupby("market_day", sort=True).agg(
        price_median=("value", "median"),
        negative_share=("value", lambda values: float((values < 0).mean())),
        peak_to_valley=("value", lambda values: float(values.max() - values.min())),
    )
    if len(daily) < 90:
        raise ValueError("Explanation requires 90 complete causal price days")
    day_median = daily["price_median"]
    m7 = float(day_median.iloc[-7:].median())
    m28 = float(day_median.iloc[-28:].median())
    m90 = float(day_median.iloc[-90:].median())
    scale7 = _robust_scale(day_median.iloc[-7:])
    scale28 = _robust_scale(day_median.iloc[-28:])
    scale90 = _robust_scale(day_median.iloc[-90:])
    z_price = (m7 - m90) / max(scale90, EPS)
    z_volatility = float(np.log(max(scale7, EPS) / max(scale90, EPS)))
    negative7 = float(daily["negative_share"].iloc[-7:].mean())
    negative28 = float(daily["negative_share"].iloc[-28:].mean())
    negative90 = float(daily["negative_share"].iloc[-90:].mean())
    ptv7 = float(daily["peak_to_valley"].iloc[-7:].mean())
    ptv90 = float(daily["peak_to_valley"].iloc[-90:].mean())
    return {
        "formula_version": "robust_market_state_v1",
        "sample_days": {"window_7d": 7, "window_28d": 28, "window_90d": 90},
        "price_median_cny_mwh": {"m7": m7, "m28": m28, "m90": m90},
        "robust_scale_cny_mwh": {"s7": scale7, "s28": scale28, "s90": scale90},
        "negative_share": {"d7": negative7, "d28": negative28, "d90": negative90},
        "peak_to_valley_cny_mwh": {"d7": ptv7, "d90": ptv90},
        "z_price": float(z_price),
        "z_volatility": z_volatility,
        "delta_negative": float(negative7 - negative90),
        "delta_peak_to_valley": float(ptv7 - ptv90),
        "price_state": _direction(z_price, threshold=0.5, low="偏低", high="偏高", neutral="接近常态"),
        "volatility_state": _direction(z_volatility, threshold=0.15, low="收敛", high="扩大", neutral="接近常态"),
    }


def _calendar_summary(target: pd.Timestamp) -> dict[str, Any]:
    frame = _calendar_frame(pd.DatetimeIndex([target]))
    row = frame.iloc[0]
    return {
        "target_date": target.strftime("%Y-%m-%d"),
        "day_type": str(row["day_type"]),
        "weekday": int(target.weekday()),
        "month_position": "月初" if target.day <= 10 else "月末" if target.day >= 21 else "月中",
        "quarter": int(target.quarter),
        "is_public_holiday": bool(row["is_public_holiday"]),
        "is_adjusted_workday": bool(row["is_adjusted_workday"]),
        "is_weekend_effective": bool(row["is_weekend_effective"]),
    }


def _calendar_frame(days: pd.DatetimeIndex) -> pd.DataFrame:
    frame = build_calendar_v01(days)
    day_type = np.where(
        frame["is_public_holiday"],
        "public_holiday",
        np.where(frame["is_adjusted_workday"], "adjusted_workday", np.where(frame["is_weekend_effective"], "effective_weekend", "regular_workday")),
    )
    result = frame.copy()
    result["day_type"] = day_type
    return result


def _weather_z(reference: pd.DataFrame, variable: str, forecast_value: float) -> float | None:
    values = reference.groupby("market_day", sort=True)[variable].mean().dropna()
    if len(values) < 14:
        return None
    return float((forecast_value - values.median()) / max(_robust_scale(values), EPS))


def _high_low_effect(daily: pd.DataFrame) -> float:
    q25, q75 = daily["weather"].quantile([0.25, 0.75])
    return float(daily.loc[daily["weather"] >= q75, "price"].median() - daily.loc[daily["weather"] <= q25, "price"].median())


def _bootstrap_effect_ci(daily: pd.DataFrame, *, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = daily[["price", "weather"]].to_numpy(dtype=float)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = pd.DataFrame(values[rng.integers(0, len(values), len(values))], columns=["price", "weather"])
        estimates[draw] = _high_low_effect(sample)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _robust_scale(values: pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    if scale < EPS:
        scale = float(np.std(array))
    return max(scale, EPS)


def _direction(value: float, *, threshold: float, low: str, high: str, neutral: str) -> str:
    return low if value <= -threshold else high if value >= threshold else neutral


def _prediction_checksum(prediction: pd.DataFrame) -> str:
    return sha256(prediction.to_csv(index=False).encode("utf-8")).hexdigest()


def _timestamp_series(values: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(values, errors="coerce")
    if timestamp.isna().any():
        raise ValueError("Timestamp column contains unparseable values")
    if getattr(timestamp.dt, "tz", None) is None:
        return timestamp.dt.tz_localize("Asia/Shanghai")
    return timestamp.dt.tz_convert("Asia/Shanghai")


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, str) and value.rstrip().endswith("Asia/Shanghai"):
        return pd.Timestamp(value.rsplit("Asia/Shanghai", maxsplit=1)[0].strip()).tz_localize("Asia/Shanghai")
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("Asia/Shanghai") if timestamp.tz is None else timestamp.tz_convert("Asia/Shanghai")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _build_claims(
    *,
    market_state: dict[str, Any],
    calendar: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    data_snapshot_hash: str,
    prediction_checksum: str,
) -> list[dict[str, Any]]:
    """Build only rule-backed, re-computable explanation claims."""
    common = {
        "data_snapshot_hash": data_snapshot_hash,
        "prediction_checksum": prediction_checksum,
    }
    claims: list[dict[str, Any]] = [
        {
            **common,
            "claim_id": "market.recent_price_state",
            "claim_type": "market_state",
            "period_group": None,
            "direction": _claim_direction(market_state["z_price"]),
            "formula_name": "robust_market_state_v1",
            "input_values": {
                "m7": market_state["price_median_cny_mwh"]["m7"],
                "m90": market_state["price_median_cny_mwh"]["m90"],
                "s90": market_state["robust_scale_cny_mwh"]["s90"],
                "z_price": market_state["z_price"],
            },
            "reference_window": "recent_7d_vs_recent_90d",
            "sample_days": 90,
            "effect_estimate": market_state["z_price"],
            "ci95_low": None,
            "ci95_high": None,
            "confidence_level": "descriptive",
            "language_template_id": "market_price_state_v1",
            "statement": (
                f"最近 7 日价格中心相对 90 日常态为{market_state['price_state']}（z={market_state['z_price']:.2f}）；"
                "这是可观测市场状态描述，不是对未来价格的确定承诺。"
            ),
        },
        {
            **common,
            "claim_id": "calendar.target_day_type",
            "claim_type": "calendar_state",
            "period_group": None,
            "direction": "context",
            "formula_name": "calendar_v01",
            "input_values": {
                "day_type": calendar["day_type"],
                "month_position": calendar["month_position"],
                "quarter": calendar["quarter"],
            },
            "reference_window": "target_date_known_calendar",
            "sample_days": 1,
            "effect_estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "confidence_level": "known_calendar",
            "language_template_id": "calendar_context_v1",
            "statement": (
                f"目标日属于 {calendar['day_type']}，处于{calendar['month_position']}、第 {calendar['quarter']} 季度；"
                "日历信息用于选择历史参考集，不单独断言价格方向。"
            ),
        },
    ]
    for group, details in groups.items():
        reference_median = details["reference_price_median_cny_mwh"]
        reference_scale = details["reference_price_robust_scale_cny_mwh"]
        prediction_mean = details["prediction_summary"]["mean_predicted_cny_mwh"]
        prediction_z = (
            (prediction_mean - reference_median) / max(float(reference_scale), EPS)
            if reference_median is not None and reference_scale is not None
            else None
        )
        details["prediction_z_vs_reference"] = prediction_z
        stable_directions: list[str] = []
        for variable, association in details["weather_price_associations"].items():
            z_value = details["forecast_z_vs_reference"].get(variable)
            stable = (
                z_value is not None
                and abs(float(z_value)) >= 0.5
                and association["status"] == "stable"
                and association["ci95_low"] is not None
                and association["ci95_high"] is not None
            )
            if not stable:
                claims.append(
                    _weather_claim(
                        common=common,
                        group=group,
                        variable=variable,
                        z_value=z_value,
                        association=association,
                        direction="insufficient_evidence",
                        confidence="insufficient" if association["status"] == "insufficient_stable_history" else "uncertain",
                        statement=(
                            f"{_period_label(group)}时段的{_weather_label(variable)}未同时满足明显 Forecast 偏离和稳定历史价格关联，"
                            "不将其列为本次预测的主要方向性解释。"
                        ),
                    )
                )
                continue
            effect = float(association["effect_cny_mwh"])
            signal = float(z_value) * effect
            direction = "downside_support" if signal < 0 else "upside_support"
            stable_directions.append(direction)
            change = "高于" if float(z_value) > 0 else "低于"
            price_direction = "较低" if effect < 0 else "较高"
            risk = "价格下行风险增加" if direction == "downside_support" else "价格上行风险增加"
            claims.append(
                _weather_claim(
                    common=common,
                    group=group,
                    variable=variable,
                    z_value=z_value,
                    association=association,
                    direction=direction,
                    confidence="stable",
                    statement=(
                        f"{_period_label(group)}时段的{_weather_label(variable)}{change}参考集常态（z={float(z_value):.2f}）；"
                        f"历史样本中该变量较强/较高时对应{price_direction}价格的统计关联 CI95=[{association['ci95_low']:.1f}, {association['ci95_high']:.1f}]。"
                        f"因此天气侧支持该时段{risk}；这不是对电力系统机制或市场价格的确定因果判断。"
                    ),
                )
            )
        support = _aggregate_weather_support(stable_directions)
        claims.append(
            {
                **common,
                "claim_id": f"prediction.{group}.reference_comparison",
                "claim_type": "prediction_comparison",
                "period_group": group,
                "direction": _prediction_consistency(support, prediction_z),
                "formula_name": "prediction_group_mean_vs_calendar_reference_v1",
                "input_values": {
                    "mean_predicted_cny_mwh": prediction_mean,
                    "reference_price_median_cny_mwh": reference_median,
                    "reference_price_robust_scale_cny_mwh": reference_scale,
                    "prediction_z_vs_reference": prediction_z,
                    "weather_support": support,
                },
                "reference_window": details["reference_status"],
                "sample_days": details["reference_days"],
                "effect_estimate": prediction_z,
                "ci95_low": None,
                "ci95_high": None,
                "confidence_level": "descriptive" if prediction_z is not None else "insufficient",
                "language_template_id": "prediction_weather_consistency_v1",
                "statement": _prediction_statement(group=group, support=support, prediction_z=prediction_z),
            }
        )
    return claims


def _weather_claim(
    *,
    common: dict[str, str],
    group: str,
    variable: str,
    z_value: float | None,
    association: dict[str, Any],
    direction: str,
    confidence: str,
    statement: str,
) -> dict[str, Any]:
    return {
        **common,
        "claim_id": f"weather.{group}.{variable}",
        "claim_type": "weather_market_evidence",
        "period_group": group,
        "direction": direction,
        "formula_name": "forecast_z_plus_high_low_quartile_effect_v1",
        "input_values": {"variable": variable, "forecast_z_vs_reference": z_value},
        "reference_window": "calendar_matched_recent_56d",
        "sample_days": association["sample_days"],
        "effect_estimate": association["effect_cny_mwh"],
        "ci95_low": association["ci95_low"],
        "ci95_high": association["ci95_high"],
        "confidence_level": confidence,
        "language_template_id": "weather_historical_association_v1",
        "statement": statement,
    }


def _aggregate_weather_support(directions: list[str]) -> str:
    unique = set(directions)
    if len(unique) == 1:
        return next(iter(unique))
    return "insufficient_evidence"


def _prediction_consistency(weather_support: str, prediction_z: float | None) -> str:
    if prediction_z is None or abs(prediction_z) < 0.5 or weather_support == "insufficient_evidence":
        return "insufficient_evidence"
    prediction_direction = "downside_support" if prediction_z < 0 else "upside_support"
    return "consistent" if prediction_direction == weather_support else "inconsistent"


def _prediction_statement(*, group: str, support: str, prediction_z: float | None) -> str:
    label = _period_label(group)
    if prediction_z is None:
        return f"{label}时段缺少完整日历参考价格统计，无法比较预测曲线与历史常态。"
    prediction_direction = "低于" if prediction_z <= -0.5 else "高于" if prediction_z >= 0.5 else "接近"
    if support == "insufficient_evidence":
        return f"{label}时段预测均价{prediction_direction}同类日参考水平（z={prediction_z:.2f}）；天气侧方向性证据不足，不强行归因。"
    support_text = "下行" if support == "downside_support" else "上行"
    consistency = "一致" if _prediction_consistency(support, prediction_z) == "consistent" else "不一致"
    return (
        f"{label}时段预测均价{prediction_direction}同类日参考水平（z={prediction_z:.2f}），"
        f"与天气侧{support_text}风险证据{consistency}。"
    )


def _claim_direction(value: float) -> str:
    return "downside_support" if value <= -0.5 else "upside_support" if value >= 0.5 else "neutral"


def _period_label(group: str) -> str:
    return {
        "night": "夜间",
        "morning": "早间",
        "solar_midday": "午间",
        "evening_peak": "晚高峰",
        "late_night": "深夜",
    }[group]


def _weather_label(variable: str) -> str:
    return {
        "temperature_2m": "2 米气温",
        "shortwave_radiation": "短波辐照",
        "cloud_cover": "云量",
        "wind_speed_100m": "100 米风速",
    }.get(variable, variable)


def _render_markdown(payload: dict[str, Any]) -> str:
    market = payload["market_state"]
    calendar = payload["calendar"]
    lines = [
        f"# 山东省预测系统：{payload['target_date']} 白箱解释",
        "",
        "## 数据完整性与可见性",
        "",
        f"- 决策时点：`{payload['as_of']}`",
        f"- 市场标签历史截止：`{payload['causal_history_label_cutoff']}`",
        f"- 数据快照：`{payload['data_snapshot_hash']}`",
        f"- 预测 checksum：`{payload['prediction_checksum']}`",
        "",
        "## 市场状态",
        "",
        f"- 最近 7 日价格中位数为 {market['price_median_cny_mwh']['m7']:.1f} CNY/MWh，90 日中位数为 {market['price_median_cny_mwh']['m90']:.1f}，稳健偏移 z={market['z_price']:.2f}，当前价格水平：**{market['price_state']}**。",
        f"- 近期波动状态：**{market['volatility_state']}**；7 日负价比例为 {market['negative_share']['d7']:.1%}，90 日为 {market['negative_share']['d90']:.1%}。",
        "",
        "## 日历日型",
        "",
        f"- 目标日为 **{calendar['day_type']}**，{calendar['month_position']}，第 {calendar['quarter']} 季度。",
        "",
        "## 天气与历史价格关联",
        "",
    ]
    for group, details in payload["period_groups"].items():
        forecast = details["forecast_summary"]
        prediction = details["prediction_summary"]
        radiation_z = details["forecast_z_vs_reference"].get("shortwave_radiation")
        radiation_text = "数据不足" if radiation_z is None else f"z={radiation_z:.2f}"
        lines.extend(
            [
                f"### {group}",
                "",
                f"- 目标日平均短波辐照：{forecast.get('shortwave_radiation', float('nan')):.1f}（相对参考集 {radiation_text}）。",
                f"- 预测均价：{prediction['mean_predicted_cny_mwh']:.1f} CNY/MWh；最大负价概率：{prediction['max_negative_probability']:.1%}；平均 P10-P90 宽度：{prediction['mean_interval_width_cny_mwh']:.1f}。",
            ]
        )
        group_claims = [claim for claim in payload["claims"] if claim["period_group"] == group]
        stable_weather_claims = [
            claim
            for claim in group_claims
            if claim["claim_type"] == "weather_market_evidence" and claim["confidence_level"] == "stable"
        ]
        if stable_weather_claims:
            lines.extend(f"- {claim['statement']}" for claim in stable_weather_claims)
        else:
            lines.append("- 天气变量未同时满足明显 Forecast 偏离与稳定历史价格关联；不将天气列为该时段的主要方向性解释。")
        comparison = next(claim for claim in group_claims if claim["claim_type"] == "prediction_comparison")
        lines.append(f"- {comparison['statement']}")
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 本报告使用显式稳健统计、日历匹配和按日 bootstrap；它说明历史证据方向，不把统计关联表述为已证明的市场因果机制。",
            "- 解释层只读取数据快照和已生成预测，不修改模型权重、点预测、负价概率或区间。",
        ]
    )
    return "\n".join(lines) + "\n"
