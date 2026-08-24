"""Write auditable final report artifacts for a calibrated QiluPulse-96 run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from openpyxl.styles import Font, PatternFill
import pandas as pd


FINAL_REPORT_VERSION = "final_report_v2"
_FINAL_COLUMNS = (
    "market_date",
    "period_start",
    "predicted_cny_mwh",
    "negative_probability",
    "p10_cny_mwh",
    "p50_cny_mwh",
    "p90_cny_mwh",
    "raw_predicted_cny_mwh",
    "raw_p10_cny_mwh",
    "raw_p50_cny_mwh",
    "raw_p90_cny_mwh",
    "bias_group",
    "bias_status",
    "bias_correction_cny_mwh",
    "interval_status",
    "interval_lower_expansion_cny_mwh",
    "interval_upper_expansion_cny_mwh",
)


@dataclass(frozen=True)
class FinalReportArtifacts:
    report_dir: Path
    plot_path: Path
    excel_path: Path
    markdown_path: Path
    report_json_path: Path
    explanation_json_path: Path
    explanation_status: str


def report_directory(root: str | Path, *, target_date: str, run_id: str) -> Path:
    """Canonical date-partitioned report path for new production runs."""
    return Path(root) / "runs" / "reports" / str(target_date) / str(run_id)


def write_final_report_artifacts(
    root: str | Path,
    *,
    result,
    target_date: str,
    as_of: str,
    explanation_payload: dict[str, Any] | None = None,
    explanation_error: str | None = None,
    report_dir: str | Path | None = None,
) -> FinalReportArtifacts:
    """Persist a complete final-report package without changing prediction data."""
    root = Path(root)
    detail = _load_final_detail(result.detail_path, target_date=target_date, metadata=result.metadata)
    report_dir = Path(report_dir) if report_dir is not None else root / "runs" / "reports" / str(result.run_id)
    report_dir.mkdir(parents=True, exist_ok=True)

    explanation_status = "active" if explanation_payload is not None else "unavailable"
    if explanation_payload is None and not explanation_error:
        explanation_error = "White-box explanation did not produce a payload."

    plot_path = report_dir / "final_prediction.png"
    excel_path = report_dir / "final_prediction.xlsx"
    markdown_path = report_dir / "final_report.md"
    report_json_path = report_dir / "final_report.json"
    explanation_json_path = report_dir / "whitebox_explanation.json"

    summary = _summary(
        detail,
        result=result,
        target_date=target_date,
        as_of=as_of,
        explanation_status=explanation_status,
        explanation_error=explanation_error,
    )
    _plot_final_prediction(detail, plot_path, summary)
    _write_excel(
        excel_path,
        detail=detail,
        explanation_payload=explanation_payload,
        summary=summary,
        result=result,
    )
    explanation_document = {
        "status": explanation_status,
        "error": explanation_error,
        "payload": explanation_payload,
    }
    explanation_json_path.write_text(
        json.dumps(explanation_document, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_document = {
        **summary,
        "ai_interpretation_status": "pending",
        "report_revision": 0,
        "artifact_paths": {
            "final_prediction_png": plot_path.name,
            "final_prediction_xlsx": excel_path.name,
            "final_report_markdown": markdown_path.name,
            "whitebox_explanation_json": explanation_json_path.name,
            "ai_interpretation_json": "ai_interpretation.json",
            "ai_interpretation_markdown": "ai_interpretation.md",
            "prediction_detail": str(Path(result.detail_path)),
        },
    }
    report_json_path.write_text(
        json.dumps(report_document, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    markdown_path.write_text(
        _render_markdown(report_document, explanation_payload=explanation_payload), encoding="utf-8"
    )
    return FinalReportArtifacts(
        report_dir=report_dir,
        plot_path=plot_path,
        excel_path=excel_path,
        markdown_path=markdown_path,
        report_json_path=report_json_path,
        explanation_json_path=explanation_json_path,
        explanation_status=explanation_status,
    )


def _load_final_detail(path: str | Path, *, target_date: str, metadata: dict[str, Any]) -> pd.DataFrame:
    calibration = metadata.get("calibration", {}) if isinstance(metadata, dict) else {}
    if calibration.get("calibration_status") != "active":
        raise ValueError("Final report requires calibration_status=active")
    detail = pd.read_csv(path)
    required = {
        "market_date",
        "period_start",
        "predicted_cny_mwh",
        "negative_probability",
        "p10_cny_mwh",
        "p50_cny_mwh",
        "p90_cny_mwh",
        "raw_predicted_cny_mwh",
        "bias_status",
        "interval_status",
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"Final report prediction detail is missing {missing}")
    if len(detail) != 96 or not detail["market_date"].astype(str).eq(str(target_date)).all():
        raise ValueError("Final report requires exactly 96 rows for the target date")
    if not detail["bias_status"].astype(str).eq("active").all() or not detail["interval_status"].astype(str).eq("active").all():
        raise ValueError("Final report requires active bias and interval post-processing for all 96 slots")
    slots = pd.to_datetime(detail["period_start"].astype(str), format="%H:%M", errors="coerce")
    expected = set(range(96))
    values = set((slots.dt.hour * 4 + slots.dt.minute // 15).dropna().astype(int))
    if slots.isna().any() or values != expected:
        raise ValueError("Final report requires unique 15-minute slots from 00:00 to 23:45")
    numeric = ["predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"]
    detail[numeric] = detail[numeric].apply(pd.to_numeric, errors="coerce")
    if detail[numeric].isna().any().any() or not detail["negative_probability"].between(0, 1).all():
        raise ValueError("Final report values must be finite and have valid negative probabilities")
    if not ((detail["p10_cny_mwh"] <= detail["p50_cny_mwh"]) & (detail["p50_cny_mwh"] <= detail["p90_cny_mwh"])).all():
        raise ValueError("Final report quantiles must be monotonic")
    result = detail.copy()
    result["_slot"] = slots.dt.hour * 4 + slots.dt.minute // 15
    return result.sort_values("_slot", ignore_index=True)


def _summary(
    detail: pd.DataFrame,
    *,
    result,
    target_date: str,
    as_of: str,
    explanation_status: str,
    explanation_error: str | None,
) -> dict[str, Any]:
    peak_index = int(detail["predicted_cny_mwh"].idxmax())
    valley_index = int(detail["predicted_cny_mwh"].idxmin())
    risk_index = int(detail["negative_probability"].idxmax())
    metadata = result.metadata
    calibration = metadata.get("calibration", {})
    return {
        "report_version": FINAL_REPORT_VERSION,
        "run_id": str(result.run_id),
        "target_date": str(target_date),
        "as_of": str(as_of),
        "publish_status": str(result.publish_status),
        "row_count": int(len(detail)),
        "calibration_status": calibration.get("calibration_status"),
        "calibration_history_days": calibration.get("calibration_history_days"),
        "mean_predicted_cny_mwh": float(detail["predicted_cny_mwh"].mean()),
        "peak": _point(detail, peak_index),
        "valley": _point(detail, valley_index),
        "max_negative_probability": _point(detail, risk_index),
        "postprocess": {
            "mean_bias_correction_cny_mwh": _optional_mean(detail, "bias_correction_cny_mwh"),
            "mean_interval_lower_expansion_cny_mwh": _optional_mean(detail, "interval_lower_expansion_cny_mwh"),
            "mean_interval_upper_expansion_cny_mwh": _optional_mean(detail, "interval_upper_expansion_cny_mwh"),
            "bias_status": _first(detail, "bias_status"),
            "interval_status": _first(detail, "interval_status"),
        },
        "data_coverage": {
            "realtime_cutoff": metadata.get("realtime_cutoff"),
            "weather_source_hash": metadata.get("weather_source_hash"),
            "weather_source_counts": metadata.get("weather_source_counts"),
            "weather_completion_manifest": metadata.get("weather_completion_manifest"),
            "bundle_parameter_checksum": metadata.get("parameter_checksum"),
            "bundle_sha256": metadata.get("bundle_sha256"),
        },
        "explanation_status": explanation_status,
        "explanation_error": explanation_error,
        "prediction_sha256": sha256(detail.drop(columns=["_slot"]).to_csv(index=False).encode("utf-8")).hexdigest(),
    }


def _point(detail: pd.DataFrame, index: int) -> dict[str, Any]:
    row = detail.loc[index]
    return {
        "period_start": str(row["period_start"]),
        "predicted_cny_mwh": float(row["predicted_cny_mwh"]),
        "p10_cny_mwh": float(row["p10_cny_mwh"]),
        "p90_cny_mwh": float(row["p90_cny_mwh"]),
        "negative_probability": float(row["negative_probability"]),
    }


def _first(frame: pd.DataFrame, column: str) -> Any:
    return None if column not in frame else frame[column].iloc[0]


def _optional_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _plot_final_prediction(detail: pd.DataFrame, path: Path, summary: dict[str, Any]) -> None:
    x = detail["_slot"].to_numpy()
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    figure, (price_axis, risk_axis) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.14}
    )
    price_axis.fill_between(
        x,
        detail["p10_cny_mwh"],
        detail["p90_cny_mwh"],
        color="#4C78A8",
        alpha=0.20,
        label="P10-P90 预测区间",
    )
    price_axis.plot(x, detail["predicted_cny_mwh"], color="#245B8E", linewidth=2.4, label="最终预测（后处理）")
    for key, color, label in (("valley", "#D65F5F", "谷值"), ("peak", "#59A14F", "峰值")):
        point = summary[key]
        slot = int(detail.index[detail["period_start"].eq(point["period_start"])][0])
        price_axis.scatter(slot, point["predicted_cny_mwh"], color=color, zorder=4)
        price_axis.annotate(
            f"{label} {point['period_start']}：{point['predicted_cny_mwh']:.1f}",
            (slot, point["predicted_cny_mwh"]),
            xytext=(8, 16),
            textcoords="offset points",
            color=color,
        )
    price_axis.set_title(f"QiluPulse-96 最终预测 | {summary['target_date']}", loc="left", weight="bold")
    price_axis.set_ylabel("元/MWh")
    price_axis.set_ylim(bottom=min(0.0, float(detail["p10_cny_mwh"].min()) - 20.0))
    price_axis.grid(axis="y", alpha=0.25)
    price_axis.legend(loc="upper left", frameon=False)
    risk_axis.fill_between(x, 0, detail["negative_probability"] * 100, color="#F28E2B", alpha=0.22)
    risk_axis.plot(x, detail["negative_probability"] * 100, color="#D06A13", linewidth=2, label="负价概率")
    risk_axis.set_ylabel("概率")
    risk_axis.set_ylim(0, max(15.0, float(detail["negative_probability"].max()) * 100 + 1.0))
    risk_axis.yaxis.set_major_formatter(PercentFormatter(100))
    risk_axis.grid(axis="y", alpha=0.25)
    risk_axis.legend(loc="upper left", frameon=False)
    ticks = list(range(0, 97, 12))
    labels = [detail["period_start"].iloc[index] if index < len(detail) else "24:00" for index in ticks]
    risk_axis.set_xticks(ticks, labels)
    risk_axis.set_xlabel("周期起点（北京时间）")
    figure.text(
        0.125,
        0.01,
        f"后处理：{summary['calibration_status']}；校准历史：{summary['calibration_history_days']} 天；共 96 个 15 分钟点",
        fontsize=9,
        color="#555555",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_excel(
    path: Path,
    *,
    detail: pd.DataFrame,
    explanation_payload: dict[str, Any] | None,
    summary: dict[str, Any],
    result,
) -> None:
    final_columns = [column for column in _FINAL_COLUMNS if column in detail]
    final = detail[final_columns].copy()
    hourly = _hourly_summary(detail)
    explanation = _explanation_rows(explanation_payload)
    audit = _audit_rows(summary, result.metadata)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="Final_96", index=False)
        hourly.to_excel(writer, sheet_name="Hourly_24", index=False)
        explanation.to_excel(writer, sheet_name="Explanation", index=False)
        audit.to_excel(writer, sheet_name="Audit", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
            for column_cells in worksheet.columns:
                width = min(48, max(12, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def _hourly_summary(detail: pd.DataFrame) -> pd.DataFrame:
    frame = detail.copy()
    frame["hour"] = frame["period_start"].str.slice(0, 2) + ":00"
    return (
        frame.groupby("hour", sort=True)
        .agg(
            mean_predicted_cny_mwh=("predicted_cny_mwh", "mean"),
            mean_p10_cny_mwh=("p10_cny_mwh", "mean"),
            mean_p50_cny_mwh=("p50_cny_mwh", "mean"),
            mean_p90_cny_mwh=("p90_cny_mwh", "mean"),
            max_negative_probability=("negative_probability", "max"),
        )
        .reset_index()
    )


def _explanation_rows(payload: dict[str, Any] | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame([{"kind": "status", "value": "unavailable"}])
    rows: list[dict[str, Any]] = []
    for claim in payload.get("claims", []):
        rows.append(
            {
                "kind": "claim",
                "identifier": claim.get("claim_id"),
                "period_group": claim.get("period_group"),
                "statement": claim.get("statement"),
                "confidence_level": claim.get("confidence_level"),
                "reference_window": claim.get("reference_window"),
                "effect_estimate": claim.get("effect_estimate"),
            }
        )
    for group, values in payload.get("period_groups", {}).items():
        prediction = values.get("prediction_summary", {})
        rows.append(
            {
                "kind": "period_group",
                "identifier": group,
                "period_group": group,
                "statement": None,
                "confidence_level": values.get("reference_status"),
                "reference_window": f"slots {values.get('slot_start')}..{values.get('slot_end')}",
                "effect_estimate": prediction.get("mean_predicted_cny_mwh"),
            }
        )
    return pd.DataFrame(rows or [{"kind": "status", "value": "no claims"}])


def _audit_rows(summary: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    values = {
        "run_id": summary["run_id"],
        "target_date": summary["target_date"],
        "as_of": summary["as_of"],
        "publish_status": summary["publish_status"],
        "calibration_status": summary["calibration_status"],
        "calibration_history_days": summary["calibration_history_days"],
        "realtime_cutoff": summary["data_coverage"]["realtime_cutoff"],
        "weather_source_hash": summary["data_coverage"]["weather_source_hash"],
        "weather_completion_manifest": summary["data_coverage"]["weather_completion_manifest"],
        "bundle_parameter_checksum": summary["data_coverage"]["bundle_parameter_checksum"],
        "bundle_sha256": summary["data_coverage"]["bundle_sha256"],
        "prediction_sha256": summary["prediction_sha256"],
        "explanation_status": summary["explanation_status"],
        "ai_interpretation_status": summary.get("ai_interpretation_status", "pending"),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
    }
    return pd.DataFrame({"field": list(values), "value": [values[key] for key in values]})


def _render_markdown(report: dict[str, Any], *, explanation_payload: dict[str, Any] | None) -> str:
    peak = report["peak"]
    valley = report["valley"]
    risk = report["max_negative_probability"]
    lines = [
        f"# QiluPulse-96 最终预测报告（{report['target_date']}）",
        "",
        "## 最终预测",
        "",
        f"- 运行编号：`{report['run_id']}`",
        f"- 决策时点：`{report['as_of']}`",
        f"- 发布状态：`{report['publish_status']}`",
        f"- 最终预测均值：`{report['mean_predicted_cny_mwh']:.2f} 元/MWh`",
        f"- 峰值：`{peak['period_start']} / {peak['predicted_cny_mwh']:.2f} 元/MWh`",
        f"- 谷值：`{valley['period_start']} / {valley['predicted_cny_mwh']:.2f} 元/MWh`",
        f"- 最高负价概率：`{risk['period_start']} / {risk['negative_probability']:.2%}`",
        "",
        "## 后处理与数据边界",
        "",
        f"- 后处理状态：`{report['calibration_status']}`；校准历史：`{report['calibration_history_days']} 天`。",
        f"- 实时价格因果截止：`{report['data_coverage']['realtime_cutoff']}`。",
        f"- 天气来源 hash：`{report['data_coverage']['weather_source_hash']}`。",
        "- 未经后处理的 raw 结果不作为最终预测；它们仅保留在 Excel 中用于审计与对比。",
        "",
        "## AI 主要方向性解读",
        "",
    ]
    ai_status = report.get("ai_interpretation_status", "pending")
    if ai_status != "active":
        lines.extend(
            [
                f"- 状态：`{ai_status}`。",
                "- 白箱证据已经单独保存，等待 QiluPulse操作筛选与本次预测方向直接相关的解释。",
                "- 在 AI 解读完成前，本报告不机械罗列白箱 claim，也不补造方向性结论。",
            ]
        )
    if explanation_payload is None:
        lines.extend(
            [
                "",
                f"- 解释层状态：`{report['explanation_status']}`。",
                f"- 白箱解释状态：`{report['explanation_status']}`。",
                f"- 原因：{report['explanation_error']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- 图：`final_prediction.png`",
            "- 96 点最终结果与附属数据：`final_prediction.xlsx`",
            "- 白箱解释结构化数据：`whitebox_explanation.json`",
            "- AI 方向性解读：`ai_interpretation.md` / `ai_interpretation.json`（生成后可用）",
            "- 原始预测审计明细：`prediction_detail.csv`",
        ]
    )
    return "\n".join(lines) + "\n"
