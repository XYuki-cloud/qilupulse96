"""Run an offline QiluPulse-96 feature ablation audit on settled validation days.

The command deliberately uses cached observed-proxy weather and raw model
inference only.  It never calls a weather API, applies production calibration,
publishes a prediction, or changes an existing prediction run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch


TIMEZONE = "Asia/Shanghai"
CONTEXT_SLOTS = 90 * 96
PARITY_TOLERANCE_CNY_MWH = 1e-4
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _resolve_device(requested: str, *, cuda_available: bool | None = None) -> str:
    """Resolve the CLI's user-facing device choice to a PyTorch device name."""
    available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)


def _resolve_from_root(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_dir(runtime_root: Path, output_dir: Path | None, *, start: str, end: str, checksum: str) -> Path:
    runtime = runtime_root.resolve()
    candidate = (
        (runtime / "runs" / "feature_ablation" / f"{start}_{end}_{checksum}")
        if output_dir is None
        else (output_dir if output_dir.is_absolute() else runtime / output_dir)
    ).resolve()
    try:
        candidate.relative_to(runtime)
    except ValueError as exc:
        raise ValueError(f"Ablation output must stay under runtime-root: {candidate}") from exc
    if candidate == runtime:
        raise ValueError("Ablation output cannot be the runtime root itself")
    return candidate


def _load_observed_panel(runtime_root: Path) -> dict[str, pd.DataFrame]:
    panel = _load_existing_quarter_panel(runtime_root)
    if panel is None:
        from da_forecast.sources.spatial_weather_v01 import load_or_build_observed_spatial_quarters

        panel = load_or_build_observed_spatial_quarters(cache_dir=runtime_root / "data" / "raw")
    history_root = runtime_root / "data" / "raw" / "weather_history_v1"
    history_frames: dict[str, pd.DataFrame] = {}
    if history_root.is_dir():
        for code in panel:
            history_path = history_root / code / "weather.parquet"
            if history_path.is_file():
                history_frames[code] = pd.read_parquet(history_path)
    return _merge_weather_history(panel, history_frames)


def _load_existing_quarter_panel(runtime_root: Path) -> dict[str, pd.DataFrame] | None:
    """Load a complete local quarter cache without rebuilding or fetching data.

    The normal spatial-weather helper may rebuild its derived cache when the
    latest hourly cache extends beyond an older, but otherwise complete,
    quarter cache.  That behavior is appropriate for a data-acquisition
    workflow, but an offline audit must not overwrite a valid historical panel
    with a partial cache.  A present quarter directory is therefore treated as
    an operator-supplied immutable input: it must be complete or the audit
    stops with a diagnostic.
    """
    from da_forecast.config import SHANDONG_SPATIAL_STATIONS, TIMEZONE
    from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
    from da_forecast.sources.spatial_weather_v01 import validate_station_weather

    source = runtime_root / "data" / "raw" / "openmeteo_spatial_v01_quarter"
    if not source.is_dir():
        return None
    expected_codes = [station.code for station in SHANDONG_SPATIAL_STATIONS]
    paths = {code: source / code / "weather.parquet" for code in expected_codes}
    missing_paths = [code for code, path in paths.items() if not path.is_file()]
    if missing_paths:
        raise ValueError(
            "Existing quarter weather cache is incomplete; missing stations="
            + ",".join(missing_paths)
        )

    panel: dict[str, pd.DataFrame] = {}
    for code, path in paths.items():
        frame = pd.read_parquet(path).copy()
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
        if len(index) == 0:
            raise ValueError(f"Existing quarter weather cache is empty for {code}: {path}")
        if index.has_duplicates:
            raise ValueError(f"Existing quarter weather cache has duplicate timestamps for {code}: {path}")
        expected_index = pd.date_range(index.min(), index.max(), freq="15min", tz=TIMEZONE)
        if not index.equals(expected_index):
            missing = expected_index.difference(index)
            raise ValueError(
                f"Existing quarter weather cache has {len(missing)} internal gaps for {code}; "
                f"first_missing={missing[0].isoformat()}"
            )
        missing_columns = sorted(set(STATION_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Existing quarter weather cache misses columns for {code}: {missing_columns}")
        frame.index = index
        frame.index.name = "timestamp"
        panel[code] = frame.sort_index()
    validate_station_weather(panel)
    return panel


def _market_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    result.index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    return result.sort_index()


def _merge_weather_history(
    panel: dict[str, pd.DataFrame],
    history_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Merge weather sources after normalizing every index to market timezone."""
    result: dict[str, pd.DataFrame] = {}
    for code, frame in panel.items():
        parts = [_market_weather_frame(frame)]
        if code in history_frames:
            parts.append(_market_weather_frame(history_frames[code]))
        merged = pd.concat(parts).sort_index()
        result[code] = merged[~merged.index.duplicated(keep="last")]
    return result


def _load_realtime(runtime_root: Path, manual_workbook: Path) -> tuple[pd.Series, list[str]]:
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        series = DataResolverV1(runtime_root, manual_workbook=manual_workbook).load_price("realtime")
    captured_warnings.extend(str(record.message) for record in records)
    return series, captured_warnings


def _complete_days(series: pd.Series, *, before: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    index = pd.DatetimeIndex(series.index)
    index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    normalized = pd.Series(series.to_numpy(dtype=float), index=index).sort_index()
    result: list[pd.Timestamp] = []
    for day, values in normalized.groupby(normalized.index.normalize()):
        if before is not None and day >= before:
            continue
        expected = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
        values = values[~values.index.duplicated(keep="last")].sort_index()
        if len(values) == 96 and values.index.equals(expected) and values.notna().all():
            result.append(day)
    return sorted(result)


def _actual_frame(series: pd.Series, day: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
    values = series.reindex(index)
    if values.isna().any():
        missing = int(values.isna().sum())
        raise ValueError(f"Missing {missing} settled realtime slots for {day:%Y-%m-%d}")
    return pd.DataFrame(
        {
            "market_date": day.strftime("%Y-%m-%d"),
            "period_start": index.strftime("%H:%M"),
            "actual_cny_mwh": values.to_numpy(dtype=float),
        }
    )


def _build_inputs(bundle, panel: dict[str, pd.DataFrame], realtime: pd.Series, runtime_root: Path, day: pd.Timestamp):
    from da_forecast.production.input_builder_v1 import CausalInputBuilderV1

    cutoff = day - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
    history_start = cutoff - pd.Timedelta(minutes=15 * (CONTEXT_SLOTS - 1))
    target_end = day + pd.Timedelta(hours=23, minutes=45)
    history_weather = {code: frame.loc[history_start:cutoff] for code, frame in panel.items()}
    target_weather = {code: frame.loc[day:target_end] for code, frame in panel.items()}
    return CausalInputBuilderV1(
        bundle,
        calendar_reference_dir=str(runtime_root / "data" / "reference" / "calendar"),
    ).build(
        target_date=day,
        realtime=realtime,
        day_ahead=None,
        history_weather=history_weather,
        target_weather=target_weather,
    )


def _with_actual(prediction: pd.DataFrame, actual: pd.DataFrame, variant: str) -> pd.DataFrame:
    merged = prediction.merge(actual, on=["market_date", "period_start"], how="inner", validate="one_to_one")
    if len(merged) != 96:
        raise ValueError(f"Variant {variant} did not produce exactly 96 slots")
    merged.insert(0, "variant", variant)
    return merged


def _load_ledger(runtime_root: Path, checksum: str) -> pd.DataFrame:
    path = _ledger_path(runtime_root, checksum)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration ledger is required for full parity: {path}")
    return pd.read_csv(path)


def _ledger_path(runtime_root: Path, checksum: str) -> Path:
    return runtime_root / "data" / "calibration" / "realtime_only" / checksum / "ledger.csv"


def _price_source_records(runtime_root: Path, manual_workbook: Path) -> list[dict[str, str]]:
    candidates = (
        runtime_root / "data" / "curated" / "realtime_prices_15min.parquet",
        runtime_root / "data" / "bootstrap" / "curated" / "shandong_all_network" / "SD" / "realtime_prices_15min.parquet",
        runtime_root / "data" / "raw" / "shandong_all_network" / "SD" / "realtime_prices_15min.parquet",
    )
    paths = [path for path in candidates if path.is_file()]
    paths.append(manual_workbook)
    records: list[dict[str, str]] = []
    for path in paths:
        try:
            display_path = path.relative_to(runtime_root).as_posix()
        except ValueError:
            display_path = str(path)
        records.append({"path": display_path, "sha256": _sha256(path)})
    return records


def _check_full_parity(
    full: pd.DataFrame,
    ledger: pd.DataFrame,
    day: str,
    *,
    allow_numeric_drift: bool = False,
) -> float:
    reference = ledger.loc[ledger["market_date"].astype(str) == day, ["period_start", "predicted_cny_mwh"]]
    if len(reference) != 96:
        raise ValueError(f"Calibration ledger has no complete 96-slot reference for {day}")
    merged = full.merge(reference, on="period_start", how="inner", validate="one_to_one", suffixes=("_audit", "_ledger"))
    if len(merged) != 96:
        raise ValueError(f"Full parity reference is incomplete for {day}")
    max_abs_delta = float(np.max(np.abs(merged["predicted_cny_mwh_audit"] - merged["predicted_cny_mwh_ledger"])))
    if max_abs_delta > PARITY_TOLERANCE_CNY_MWH and not allow_numeric_drift:
        raise RuntimeError(
            f"Full replay parity failed for {day}: max_abs_delta={max_abs_delta:.9f} "
            f"> tolerance={PARITY_TOLERANCE_CNY_MWH}"
        )
    return max_abs_delta


def _simple_baseline_frames(
    realtime: pd.Series,
    days: list[pd.Timestamp],
) -> dict[str, list[pd.DataFrame]]:
    from da_forecast.production.feature_ablation_v1 import calculate_metrics

    complete = _complete_days(realtime, before=days[-1] + pd.Timedelta(days=1))
    frames: dict[str, list[pd.DataFrame]] = {
        "baseline:flat_mean_28d": [],
        "baseline:flat_median_28d": [],
        "baseline:same_slot_mean_28d": [],
        "baseline:same_slot_median_28d": [],
        "baseline:last_day_same_slot": [],
    }
    for day in days:
        prior = [candidate for candidate in complete if candidate < day]
        if len(prior) < 28:
            raise ValueError(f"Need 28 complete causal days before {day:%Y-%m-%d}")
        history_days = prior[-28:]
        parts = []
        for history_day in history_days:
            frame = _actual_frame(realtime, history_day)
            parts.append(frame[["period_start", "actual_cny_mwh"]])
        history = pd.concat(parts, ignore_index=True)
        actual = _actual_frame(realtime, day)
        flat_mean = float(history["actual_cny_mwh"].mean())
        flat_median = float(history["actual_cny_mwh"].median())
        slot_mean = history.groupby("period_start")["actual_cny_mwh"].mean()
        slot_median = history.groupby("period_start")["actual_cny_mwh"].median()
        last_day = _actual_frame(realtime, prior[-1]).set_index("period_start")["actual_cny_mwh"]
        base_values = {
            "baseline:flat_mean_28d": np.full(96, flat_mean),
            "baseline:flat_median_28d": np.full(96, flat_median),
            "baseline:same_slot_mean_28d": actual["period_start"].map(slot_mean).to_numpy(float),
            "baseline:same_slot_median_28d": actual["period_start"].map(slot_median).to_numpy(float),
            "baseline:last_day_same_slot": actual["period_start"].map(last_day).to_numpy(float),
        }
        for name, values in base_values.items():
            predicted = actual.copy()
            predicted["predicted_cny_mwh"] = values
            predicted["negative_probability"] = 0.0
            predicted["p10_cny_mwh"] = values
            predicted["p90_cny_mwh"] = values
            predicted.insert(0, "variant", name)
            frames[name].append(predicted)
    return frames


def _variant_group(name: str) -> str:
    if name.startswith("weather:"):
        return "weather_variable"
    if name.startswith("calendar:"):
        return "calendar_variable"
    if name.startswith("state:"):
        return "state_variable"
    if name.startswith("weather_"):
        return "weather_group"
    if name.startswith("calendar_"):
        return "calendar_group"
    if name.startswith("price_"):
        return "state_group"
    return "reference"


def _rank_variable_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["output_sensitivity_rank"] = (
        frame["output_mean_abs_output_delta_cny_mwh"].rank(method="min", ascending=False).astype(int)
    )
    frame["predictive_contribution_rank"] = (
        frame["paired_mae_variant_minus_full"].rank(method="min", ascending=False).astype(int)
    )
    return frame.to_dict(orient="records")


def _description(name: str) -> str:
    descriptions = {
        "full": "完整输入基准",
        "weather_meteorology_off": "关闭18个气象变量，保留太阳几何变量",
        "weather_solar_geometry_off": "关闭7个太阳几何变量，保留气象变量",
        "weather_all_off": "关闭全部25个天气相关变量",
        "calendar_date_off": "保留时段编码，关闭日期属性",
        "calendar_all_off": "关闭全部14个日历变量，压力测试",
        "price_state_off": "关闭5个近期价格状态变量",
    }
    if name in descriptions:
        return descriptions[name]
    return f"将 {name} 在历史和目标输入中置为训练均值"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _build_report(
    *,
    output_dir: Path,
    manifest: dict[str, object],
    variant_frames: dict[str, list[pd.DataFrame]],
    baseline_frames: dict[str, list[pd.DataFrame]],
) -> dict[str, object]:
    from da_forecast.production.feature_ablation_v1 import (
        calculate_metrics,
        output_delta_metrics,
        paired_metric_summary,
        summarize_variant_frame,
    )

    all_daily: list[pd.DataFrame] = []
    group_rows: list[dict[str, object]] = []
    variable_rows: list[dict[str, object]] = []
    delta_rows: list[pd.DataFrame] = []
    full_frames = variant_frames["full"]
    full_by_day = {str(frame["market_date"].iloc[0]): frame for frame in full_frames}
    full_summary = summarize_variant_frame(pd.concat(full_frames, ignore_index=True))
    all_daily.append(full_summary["daily"].assign(variant="full"))

    for name, frames in variant_frames.items():
        if name == "full":
            continue
        combined = pd.concat(frames, ignore_index=True)
        summary = summarize_variant_frame(combined)
        daily = summary["daily"].assign(variant=name)
        all_daily.append(daily)
        full_daily_for_compare = full_summary["daily"]
        paired_mae = paired_metric_summary(full_daily_for_compare, summary["daily"], metric="mae_cny_mwh")
        paired_rmse = paired_metric_summary(full_daily_for_compare, summary["daily"], metric="rmse_cny_mwh")
        output_deltas: list[dict[str, float]] = []
        for frame in frames:
            day = str(frame["market_date"].iloc[0])
            output_deltas.append(output_delta_metrics(full_by_day[day], frame))
            delta = frame[["market_date", "period_start", "actual_cny_mwh", "predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p90_cny_mwh"]].copy()
            reference = full_by_day[day][["period_start", "predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p90_cny_mwh"]].rename(
                columns={
                    "predicted_cny_mwh": "full_predicted_cny_mwh",
                    "negative_probability": "full_negative_probability",
                    "p10_cny_mwh": "full_p10_cny_mwh",
                    "p90_cny_mwh": "full_p90_cny_mwh",
                }
            )
            delta = delta.merge(reference, on="period_start", how="inner", validate="one_to_one")
            delta.insert(0, "variant", name)
            delta["predicted_delta_cny_mwh"] = delta["predicted_cny_mwh"] - delta["full_predicted_cny_mwh"]
            delta["negative_probability_delta"] = delta["negative_probability"] - delta["full_negative_probability"]
            delta["p10_delta_cny_mwh"] = delta["p10_cny_mwh"] - delta["full_p10_cny_mwh"]
            delta["p90_delta_cny_mwh"] = delta["p90_cny_mwh"] - delta["full_p90_cny_mwh"]
            delta_rows.append(delta)
        output_delta = pd.DataFrame(output_deltas).mean(numeric_only=True).to_dict()
        row = {
            "variant": name,
            "group": _variant_group(name),
            "description": _description(name),
            **{f"overall_{key}": value for key, value in summary["overall"].items()},
            **{f"output_{key}": value for key, value in output_delta.items()},
            "paired_mae_variant_minus_full": paired_mae["mean_variant_minus_full"],
            "paired_mae_ci95_low": paired_mae["ci95_low"],
            "paired_mae_ci95_high": paired_mae["ci95_high"],
            "paired_mae_worse_days": paired_mae["worse_days"],
            "paired_mae_better_days": paired_mae["better_days"],
            "paired_rmse_variant_minus_full": paired_rmse["mean_variant_minus_full"],
            "paired_rmse_ci95_low": paired_rmse["ci95_low"],
            "paired_rmse_ci95_high": paired_rmse["ci95_high"],
        }
        (group_rows if _variant_group(name).endswith("group") else variable_rows).append(row)

    daily_metrics = pd.concat(all_daily, ignore_index=True)
    daily_metrics.to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    if delta_rows:
        pd.concat(delta_rows, ignore_index=True).to_csv(output_dir / "prediction_deltas.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(output_dir / "prediction_deltas.csv", index=False, encoding="utf-8-sig")
    group_summary = pd.DataFrame(group_rows)
    variable_rows = _rank_variable_rows(variable_rows)
    variable_summary = pd.DataFrame(variable_rows)
    full_group_row = {
        "variant": "full",
        "group": "reference",
        "description": _description("full"),
        **{f"overall_{key}": value for key, value in full_summary["overall"].items()},
    }
    group_summary = pd.concat([pd.DataFrame([full_group_row]), group_summary], ignore_index=True)
    group_summary.to_csv(output_dir / "group_summary.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(output_dir / "variable_sensitivity.csv", index=False, encoding="utf-8-sig")

    baseline_rows: list[dict[str, object]] = []
    for name, frames in baseline_frames.items():
        combined = pd.concat(frames, ignore_index=True)
        point = calculate_metrics(combined)
        baseline_rows.append({"variant": name, **point})

    report = {
        "audit_status": (
            "strict"
            if manifest.get("strict_full_parity_passed", True)
            else "exploratory_backend_numeric_drift"
        ),
        "manifest": manifest,
        "full": full_summary["overall"],
        "groups": group_rows,
        "variables": variable_rows,
        "baselines": baseline_rows,
        "interpretation_rules": {
            "model_dependency": "Output change after ablation indicates model reliance, not causal effect.",
            "predictive_contribution": "A variable is supported as useful only when paired validation error worsens consistently.",
            "weather_caveat": "Historical weather is observed_proxy and is not identical to production forecast weather.",
            "small_sample": "The validation window contains 14 days; bootstrap intervals are descriptive, not a definitive significance test.",
        },
    }
    _write_json(output_dir / "feature_ablation_report.json", report)
    markdown = _render_markdown(report)
    (output_dir / "feature_ablation_report.md").write_text(markdown, encoding="utf-8")
    return report


def _render_markdown(report: dict[str, object]) -> str:
    manifest = report["manifest"]
    full = report["full"]
    lines = [
        "# QiluPulse-96 特征消融与敏感度审计",
        "",
        f"- 审计状态：`{report['audit_status']}`",
        f"- 验证区间：`{manifest['start_date']} 至 {manifest['end_date']}`",
        f"- Bundle 参数 checksum：`{manifest['bundle_parameter_checksum']}`",
        f"- Bundle SHA-256：`{manifest['bundle_sha256']}`",
        f"- 天气类型：`{manifest['weather_kind']}`",
        f"- 设备：`{manifest['device']}`",
        "- 本实验只做 raw inference，不执行校准、发布或模型更新。",
        "",
        "## 完整输入基准",
        "",
        f"- MAE：`{full['mae_cny_mwh']:.3f} CNY/MWh`",
        f"- RMSE：`{full['rmse_cny_mwh']:.3f} CNY/MWh`",
        f"- 日内相关系数：`{full['within_day_correlation']}`",
        "",
        "## 分组消融结果",
        "",
        "`paired_mae_variant_minus_full > 0` 表示关闭该组后误差变大；输出变化本身不等于预测收益。",
        "",
        "| 变体 | 平均绝对输出变化 | MAE 变化 | 95% CI | 变差天数/改善天数 |",
        "|---|---:|---:|---:|---:|",
    ]
    if manifest.get("full_parity_override"):
        marker = lines.index("## 完整输入基准")
        lines[marker:marker] = [
            "> WARNING: strict full-replay parity failed because the current inference backend differs numerically from the backend that produced the historical ledger. This report is exploratory and is not parity-certified.",
            f"> Drift days: `{', '.join(manifest['full_parity_drift_days'])}`; tolerance: `{manifest['full_parity_tolerance_cny_mwh']}` CNY/MWh.",
            "",
        ]
    for row in report["groups"]:
        lines.append(
            f"| `{row['variant']}` | {row['output_mean_abs_output_delta_cny_mwh']:.3f} | "
            f"{row['paired_mae_variant_minus_full']:.3f} | "
            f"[{row['paired_mae_ci95_low']:.3f}, {row['paired_mae_ci95_high']:.3f}] | "
            f"{row['paired_mae_worse_days']}/{row['paired_mae_better_days']} |"
        )
    lines.extend([
        "",
        "## 单变量敏感度",
        "",
        "逐变量结果见 `variable_sensitivity.csv`。排名同时提供输出敏感度和 MAE 变化，不能只按输出变化排序宣称变量有用。",
        "",
        "## 简单基线",
        "",
        "| 基线 | MAE | RMSE |",
        "|---|---:|---:|",
    ])
    for row in report["baselines"]:
        lines.append(f"| `{row['variant']}` | {row['mae_cny_mwh']:.3f} | {row['rmse_cny_mwh']:.3f} |")
    lines.extend([
        "",
        "## 限制",
        "",
        "- 历史天气为 `observed_proxy`，不能直接代表正式 forecast 天气条件。",
        "- 14 天验证样本较小，bootstrap 区间用于描述不确定性，不作为单独的显著性证明。",
        "- 消融结果说明模型输入依赖和预测误差变化，不证明天气或日历变量与价格之间存在因果关系。",
        "- 本实验不读取、修改或重新生成任何生产预测。",
        "",
    ])
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    runtime_root = args.runtime_root.resolve()
    bundle_path = _resolve_from_root(root, args.bundle_path)
    manual_workbook = _resolve_from_root(root, args.manual_workbook)
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"runtime-root does not exist: {runtime_root}")
    if not manual_workbook.is_file():
        raise FileNotFoundError(f"manual workbook does not exist: {manual_workbook}")

    from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
    from da_forecast.production.feature_ablation_v1 import (
        GROUP_VARIANTS,
        all_variant_names,
        ablate_inputs,
    )
    from da_forecast.production.inference_v1 import infer_qilupulse96

    bundle = QiluPulse96ProductionBundle.load(bundle_path)
    start = _local_day(args.start_date).normalize()
    end = _local_day(args.end_date).normalize()
    if end < start:
        raise ValueError("end-date must not precede start-date")
    days = list(pd.date_range(start, end, freq="D", tz=TIMEZONE))
    if len(days) != 14 or start.strftime("%Y-%m-%d") != "2026-08-07" or end.strftime("%Y-%m-%d") != "2026-08-20":
        raise ValueError("This audit is fixed to the 14-day validation window 2026-08-07 through 2026-08-20")

    device = _resolve_device(args.device)
    torch.manual_seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)

    realtime, warning_messages = _load_realtime(runtime_root, manual_workbook)
    panel = _load_observed_panel(runtime_root)
    ledger_path = _ledger_path(runtime_root, bundle.parameter_checksum)
    ledger = _load_ledger(runtime_root, bundle.parameter_checksum)
    output_dir = _safe_output_dir(
        runtime_root,
        args.output_dir,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        checksum=bundle.parameter_checksum,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in all_variant_names()}
    parity: dict[str, float] = {}
    parity_drift_days: list[str] = []
    for day in days:
        inputs = _build_inputs(bundle, panel, realtime, runtime_root, day)
        actual = _actual_frame(realtime, day)
        full = _with_actual(infer_qilupulse96(bundle, inputs, device=device), actual, "full")
        day_text = day.strftime("%Y-%m-%d")
        delta = _check_full_parity(
            full,
            ledger,
            day_text,
            allow_numeric_drift=args.allow_backend_numeric_drift,
        )
        parity[day_text] = delta
        if delta > PARITY_TOLERANCE_CNY_MWH:
            parity_drift_days.append(day_text)
        variant_frames["full"].append(full)
        for name in all_variant_names()[1:]:
            variant_inputs = ablate_inputs(inputs, name)
            prediction = _with_actual(infer_qilupulse96(bundle, variant_inputs, device=device), actual, name)
            variant_frames[name].append(prediction)

    baseline_frames = _simple_baseline_frames(realtime, days)
    price_hash = _sha256(manual_workbook)
    training = bundle.manifest_data.get("training_metadata", {})
    manifest = {
        "experiment": "qilupulse96_feature_ablation_v1",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "bundle_path": str(bundle_path),
        "bundle_parameter_checksum": bundle.parameter_checksum,
        "bundle_sha256": bundle.bundle_sha256,
        "model_version": bundle.manifest_data.get("model_version"),
        "training_metadata": training,
        "weather_kind": "observed_proxy",
        "weather_api_called": False,
        "calibration_applied": False,
        "publish_status": "not_applicable",
        "device": device,
        "manual_workbook": str(manual_workbook),
        "manual_workbook_sha256": price_hash,
        "input_price_sources": _price_source_records(runtime_root, manual_workbook),
        "calibration_ledger": str(ledger_path),
        "calibration_ledger_sha256": _sha256(ledger_path),
        "manual_workbook_warnings": warning_messages,
        "full_parity_max_abs_delta_by_day": parity,
        "full_parity_tolerance_cny_mwh": PARITY_TOLERANCE_CNY_MWH,
        "strict_full_parity_passed": not parity_drift_days,
        "full_parity_override": bool(args.allow_backend_numeric_drift and parity_drift_days),
        "full_parity_drift_days": parity_drift_days,
        "full_parity_status": (
            "strict_pass"
            if not parity_drift_days
            else "exploratory_backend_numeric_drift"
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runtime_output_dir": str(output_dir),
        "group_variants": list(GROUP_VARIANTS),
        "single_variable_variants": [name for name in all_variant_names() if ":" in name],
        "current_prediction_modified": False,
    }
    _write_json(output_dir / "experiment_manifest.json", manifest)
    _build_report(
        output_dir=output_dir,
        manifest=manifest,
        variant_frames=variant_frames,
        baseline_frames=baseline_frames,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=_path, required=True, help="Public source root.")
    parser.add_argument("--runtime-root", type=_path, required=True, help="Ignored private runtime root.")
    parser.add_argument("--bundle-path", type=_path, required=True, help="Explicit authorized bundle directory.")
    parser.add_argument("--manual-workbook", type=_path, required=True, help="Explicit manual realtime workbook.")
    parser.add_argument("--start-date", default="2026-08-07", help="Fixed validation start date.")
    parser.add_argument("--end-date", default="2026-08-20", help="Fixed validation end date.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-backend-numeric-drift",
        action="store_true",
        help=(
            "Allow exploratory output when strict full-replay parity fails due to "
            "backend numeric drift; the result is not parity-certified."
        ),
    )
    parser.add_argument("--output-dir", type=_path, help="Optional output directory under runtime-root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.resolve()
    args.runtime_root = _resolve_from_root(args.root, args.runtime_root)
    try:
        output_dir = run_audit(args)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    manifest = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "ready",
                "audit_status": manifest["full_parity_status"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
