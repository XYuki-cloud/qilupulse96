"""Production prediction workflow shared by the CLI and archived GUI."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
import json
import pandas as pd

from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.calendar_runtime_v1 import CalendarRuntimeV1
from da_forecast.production.data_resolver_v1 import DataResolverV1
from da_forecast.production.final_report_v1 import FinalReportArtifacts, report_directory, write_final_report_artifacts
from da_forecast.production.manual_revision_v1 import ManualRevisionStore
from da_forecast.production.input_builder_v1 import CausalInputBuilderV1
from da_forecast.production.run_service_v1 import RunServiceV1
from da_forecast.production.weather_runtime_v1 import WeatherRuntimeV1
from da_forecast.production.calibration_bootstrap_v1 import ensure_realtime_only_calibration_history
from da_forecast.production.inference_v1 import infer_qilupulse96
from da_forecast.sources.spatial_weather_v01 import load_or_build_observed_spatial_quarters
from da_forecast.system.explanation import WhiteBoxExplainer
from da_forecast.system.prediction_layer import PredictionLayerRegistry
import torch


class ProductionWorkflow:
    def __init__(
        self,
        root: str | Path,
        *,
        runtime_root: str | Path | None = None,
        manual_workbook: str | Path | None = None,
        weather_source: str = "fetch",
    ) -> None:
        self.root = Path(root)
        self.runtime_root = Path(runtime_root) if runtime_root is not None else self.root
        self.resolver = DataResolverV1(self.runtime_root, manual_workbook=manual_workbook)
        self.calendar = CalendarRuntimeV1(self.runtime_root)
        self.revisions = ManualRevisionStore(self.runtime_root)
        self.registry = PredictionLayerRegistry(self.runtime_root)
        self.manual_workbook = manual_workbook
        self.weather_source = weather_source
        self.last_weather_completion = None
        self.last_calibration_report = None
        self.last_readiness = None
        self.last_final_artifacts: FinalReportArtifacts | None = None

    def resolve_bundle(self, bundle_path: str | Path | None = None) -> QiluPulse96ProductionBundle:
        """Load an explicitly selected bundle without using a local default pointer."""
        if bundle_path is None:
            raise FileNotFoundError(
                "No model bundle was supplied. The public workflow requires "
                "an explicit --bundle-path or bundle_path argument."
            )
        return QiluPulse96ProductionBundle.load(bundle_path)

    def readiness(self, target_date: str, *, weather_complete: bool = False) -> dict[str, object]:
        report = self.resolver.readiness(target_date=target_date, weather_complete=weather_complete, calendar_confirmed=self.calendar.is_confirmed(pd.Timestamp(target_date).year))
        return report.to_dict()

    def list_prediction_runs(self) -> list[dict[str, object]]:
        """Return readable prediction ledgers, newest first, for the result page."""
        predictions_root = (self.runtime_root / "runs" / "predictions").resolve()
        if not predictions_root.is_dir():
            return []
        runs: list[dict[str, object]] = []
        for run_dir in predictions_root.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_path = run_dir / "run_metadata.json"
            detail_path = run_dir / "prediction_detail.csv"
            if not metadata_path.is_file() or not detail_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_id = str(metadata.get("run_id") or run_dir.name)
            runs.append(
                {
                    **metadata,
                    "run_id": run_id,
                    "target_date": str(metadata.get("target_date") or run_id.split("_", 1)[0]),
                    "detail_path": str(detail_path),
                    "metadata_path": str(metadata_path),
                    "modified_at": detail_path.stat().st_mtime,
                }
            )
        return sorted(runs, key=lambda item: float(item.get("modified_at", 0)), reverse=True)

    def load_prediction_run(self, run_id: str) -> tuple[dict[str, object], pd.DataFrame]:
        """Load one prediction ledger after constraining the run id to its root."""
        predictions_root = (self.runtime_root / "runs" / "predictions").resolve()
        run_dir = (predictions_root / str(run_id)).resolve()
        if run_dir.parent != predictions_root or not run_dir.is_dir():
            raise FileNotFoundError(f"Unknown prediction run: {run_id}")
        metadata_path = run_dir / "run_metadata.json"
        detail_path = run_dir / "prediction_detail.csv"
        if not metadata_path.is_file() or not detail_path.is_file():
            raise FileNotFoundError(f"Prediction run is incomplete: {run_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_id"] = str(metadata.get("run_id") or run_dir.name)
        metadata["detail_path"] = str(detail_path)
        detail = pd.read_csv(detail_path)
        return metadata, detail

    def save_revision(self, frame: pd.DataFrame, *, target_date: str, source_kind: str, operator_note: str, accepted_imputation: bool = False):
        return self.revisions.save(frame, market_date=target_date, source_kind=source_kind, operator_note=operator_note, accepted_imputation=accepted_imputation)

    def write_final_artifacts(
        self,
        result,
        *,
        target_date: str,
        as_of: str,
        realtime: pd.Series,
        history_weather: dict[str, pd.DataFrame],
        target_weather: dict[str, pd.DataFrame],
    ) -> FinalReportArtifacts:
        """Write final deliverables without reopening or changing the forecast."""
        explanation_payload: dict[str, object] | None = None
        explanation_error: str | None = None
        destination = report_directory(self.runtime_root, target_date=target_date, run_id=result.run_id)
        cutoff = _result_realtime_cutoff(result, target_date=target_date)
        result.metadata["realtime_cutoff"] = cutoff.isoformat()
        try:
            whitebox = WhiteBoxExplainer().explain(
                target_date=target_date,
                as_of=as_of,
                price_history=_price_history_frame(realtime),
                observed_weather=_weather_panel_frame(history_weather),
                forecast_weather=_weather_panel_frame(target_weather),
                prediction=pd.read_csv(result.detail_path),
                data_snapshot_hash=_explanation_input_hash(
                    realtime=realtime,
                    history_weather=history_weather,
                    target_weather=target_weather,
                    metadata=result.metadata,
                ),
                causal_history_label_cutoff=cutoff,
            )
            explanation_payload = whitebox.payload
            RunServiceV1(self.runtime_root).write_explanation(
                run_id=result.run_id,
                target_date=target_date,
                report=whitebox,
                output_dir=destination,
                basename="whitebox_explanation",
            )
        except Exception as exc:
            explanation_error = f"{type(exc).__name__}: {exc}"
        artifacts = write_final_report_artifacts(
            self.runtime_root,
            result=result,
            target_date=target_date,
            as_of=as_of,
            explanation_payload=explanation_payload,
            explanation_error=explanation_error,
            report_dir=destination,
        )
        _record_final_artifacts(self.runtime_root, result=result, artifacts=artifacts)
        self.last_final_artifacts = artifacts
        return artifacts

    def run_prediction_draft(
        self,
        *,
        target_date: str,
        as_of: str,
        fetch_weather: bool = True,
        progress: Callable[[str], None] | None = None,
        bundle_path: str | Path | None = None,
    ):
        def report(message: str) -> None:
            if progress is not None:
                progress(message)

        report("加载模型 bundle")
        bundle = self.resolve_bundle(bundle_path)
        weather = WeatherRuntimeV1(self.runtime_root, weather_source=self.weather_source)
        self.last_weather_completion = None
        report("补齐历史天气和目标日 Forecast")
        weather_run = weather.ensure_weather_to_target(
            target_date=target_date,
            as_of=as_of,
            progress=report,
        )
        self.last_weather_completion = weather_run
        report(
            "天气完成："
            f"history={weather_run.history_start}..{weather_run.history_end}；"
            f"target={weather_run.target_start}..{weather_run.target_end}；"
            f"source_counts={weather_run.source_counts}；"
            f"forecast_backfill={weather_run.used_forecast_backfill}；"
            f"source_hash={weather_run.source_hash}；"
            f"manifest={weather_run.manifest_path}"
        )
        report("加载人工实时价格（日前价格不参与当前生产模型）")
        realtime = self.resolver.load_price("realtime")
        manual_path = self.resolver.manual_workbook_path
        if manual_path is not None:
            report(f"已读取人工实时价表：{manual_path}")
        realtime_only = getattr(getattr(bundle, "spec", None), "history_extra_dim", 18) == 14
        day_ahead = None if realtime_only else self.resolver.load_price("day_ahead")
        if realtime_only:
            report("当前生产 bundle 为实时价-only，不读取日前价格")
        calibration_report = None
        self.last_calibration_report = None
        if realtime_only:
            report("确保生产后处理校准历史")
            panel = None

            def replay_day(day: pd.Timestamp) -> pd.DataFrame:
                nonlocal panel
                if panel is None:
                    panel = load_or_build_observed_spatial_quarters(
                        cache_dir=self.runtime_root / "data" / "raw"
                    )
                    history_root = self.runtime_root / "data" / "raw" / "weather_history_v1"
                    if history_root.is_dir():
                        for code, frame in list(panel.items()):
                            history_path = history_root / code / "weather.parquet"
                            if history_path.is_file():
                                history_frame = pd.read_parquet(history_path)
                                merged = pd.concat([frame, history_frame]).sort_index()
                                panel[code] = merged[~merged.index.duplicated(keep="last")]
                day = pd.Timestamp(day).tz_localize("Asia/Shanghai") if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day).tz_convert("Asia/Shanghai")
                cutoff = day - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
                history_start = cutoff - pd.Timedelta(minutes=15 * (CausalInputBuilderV1.context_slots - 1))
                target_end = day + pd.Timedelta(hours=23, minutes=45)
                history_weather = {code: frame.loc[history_start:cutoff] for code, frame in panel.items()}
                target_weather = {code: frame.loc[day:target_end] for code, frame in panel.items()}
                inputs = CausalInputBuilderV1(
                    bundle, calendar_reference_dir=str(self.runtime_root / "data" / "reference" / "calendar")
                ).build(
                    target_date=day,
                    realtime=realtime,
                    day_ahead=None,
                    history_weather=history_weather,
                    target_weather=target_weather,
                )
                return infer_qilupulse96(
                    bundle,
                    inputs,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            calibration_report = ensure_realtime_only_calibration_history(
                self.runtime_root,
                bundle=bundle,
                actual_prices=realtime,
                target_date=target_date,
                replay_day=replay_day,
                progress=report,
            )
            self.last_calibration_report = calibration_report
            report(
                f"后处理校准历史：status={calibration_report.status}；"
                f"days={calibration_report.history_days}；"
                f"last={calibration_report.history_last_date}；"
                f"ledger={calibration_report.ledger_path}"
            )
        history_weather = weather_run.history_panel
        target_weather = weather_run.target_panel if fetch_weather else history_weather
        weather_kind = "forecast" if fetch_weather else "observed_proxy"
        readiness = self.resolver.readiness(
            target_date=target_date,
            weather_complete=fetch_weather,
            calendar_confirmed=self.calendar.is_confirmed(pd.Timestamp(target_date).year),
            realtime_only=realtime_only,
        )
        self.last_readiness = readiness
        report("构建模型输入")
        inputs = CausalInputBuilderV1(bundle, calendar_reference_dir=str(self.runtime_root / "data" / "reference" / "calendar")).build(
            target_date=target_date, realtime=realtime, day_ahead=day_ahead,
            history_weather=history_weather, target_weather=target_weather,
        )
        report("运行预测并写入草稿")
        result = RunServiceV1(self.runtime_root).run_draft(
            bundle=bundle,
            inputs=inputs,
            target_date=target_date,
            weather_kind=weather_kind,
            readiness=readiness,
            target_weather_snapshot_hash=getattr(getattr(weather_run, "target_forecast", None), "snapshot_hash", None),
            actual_prices=realtime,
            calibration_history_metadata=(
                {
                    "calibration_history_status": calibration_report.status,
                    "calibration_history_days": calibration_report.history_days,
                    "calibration_history_last_date": calibration_report.history_last_date,
                    "calibration_history_ledger_path": str(calibration_report.ledger_path),
                    "calibration_history_manifest_path": str(calibration_report.manifest_path),
                    "calibration_history_source": calibration_report.source,
                    "calibration_history_bundle_parameter_checksum": getattr(calibration_report, "bundle_parameter_checksum", None),
                }
                if calibration_report is not None else None
            ),
            weather_metadata={
                "weather_completion_manifest": str(weather_run.manifest_path),
                "weather_source_hash": weather_run.source_hash,
                "weather_source_counts": weather_run.source_counts,
                "history_weather_contains_forecast_backfill": weather_run.used_forecast_backfill,
            },
        )
        report("生成最终预测图、Excel 和 Markdown 报告")
        artifacts = self.write_final_artifacts(
            result,
            target_date=target_date,
            as_of=as_of,
            realtime=realtime,
            history_weather=history_weather,
            target_weather=target_weather,
        )
        report(
            f"最终结果包已写入：{artifacts.report_dir}；"
            f"explanation_status={artifacts.explanation_status}"
        )
        return result

    def write_report(self, result, *, target_date: str, as_of: str) -> tuple[Path, Path]:
        """Write a compact machine-readable and human-readable result summary."""
        if result.detail_path is None:
            raise ValueError("Prediction result has no detail ledger")
        detail = pd.read_csv(result.detail_path)
        report_dir = report_directory(self.runtime_root, target_date=target_date, run_id=result.run_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        numeric = detail["predicted_cny_mwh"].astype(float)
        peak = int(numeric.idxmax())
        valley = int(numeric.idxmin())
        summary = {
            "run_id": result.run_id,
            "target_date": str(target_date),
            "as_of": str(as_of),
            "publish_status": result.publish_status,
            "row_count": int(len(detail)),
            "mean_predicted_cny_mwh": float(numeric.mean()),
            "peak": {
                "period_start": str(detail.loc[peak, "period_start"]),
                "predicted_cny_mwh": float(numeric.loc[peak]),
            },
            "valley": {
                "period_start": str(detail.loc[valley, "period_start"]),
                "predicted_cny_mwh": float(numeric.loc[valley]),
            },
            "negative_probability_max": float(detail["negative_probability"].astype(float).max()),
            "p10_min_cny_mwh": float(detail["p10_cny_mwh"].astype(float).min()),
            "p90_max_cny_mwh": float(detail["p90_cny_mwh"].astype(float).max()),
            "calibration": result.metadata.get("calibration", {}),
            "weather": {
                key: result.metadata[key]
                for key in ("weather_completion_manifest", "weather_source_hash", "weather_source_counts")
                if key in result.metadata
            },
            "paths": {
                "prediction_detail": str(result.detail_path),
                "run_metadata": str(self.runtime_root / "runs" / "predictions" / result.run_id / "run_metadata.json"),
            },
        }
        if self.last_final_artifacts is not None and self.last_final_artifacts.report_dir.name == str(result.run_id):
            summary["paths"].update(
                {
                    "final_prediction_png": str(self.last_final_artifacts.plot_path),
                    "final_prediction_xlsx": str(self.last_final_artifacts.excel_path),
                    "final_report_markdown": str(self.last_final_artifacts.markdown_path),
                    "final_report_json": str(self.last_final_artifacts.report_json_path),
                    "whitebox_explanation_json": str(self.last_final_artifacts.explanation_json_path),
                    "ai_interpretation_json": str(self.last_final_artifacts.report_dir / "ai_interpretation.json"),
                    "ai_interpretation_markdown": str(self.last_final_artifacts.report_dir / "ai_interpretation.md"),
                }
            )
            summary["explanation_status"] = self.last_final_artifacts.explanation_status
            summary["ai_interpretation_status"] = "pending"
        json_path = report_dir / "summary.json"
        markdown_path = report_dir / "summary.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        markdown_path.write_text(
            "\n".join(
                [
                    f"# QiluPulse-96 预测报告（{target_date}）",
                    "",
                    f"- 运行编号：`{result.run_id}`",
                    f"- as-of：`{as_of}`",
                    f"- 状态：`{result.publish_status}`",
                    f"- 完整点数：`{len(detail)}`",
                    f"- 平均预测：`{summary['mean_predicted_cny_mwh']:.2f} 元/MWh`",
                    f"- 峰值：`{summary['peak']['period_start']} / {summary['peak']['predicted_cny_mwh']:.2f}`",
                    f"- 谷值：`{summary['valley']['period_start']} / {summary['valley']['predicted_cny_mwh']:.2f}`",
                    f"- 最高负价概率：`{summary['negative_probability_max']:.4f}`",
                    f"- 后处理状态：`{summary['calibration'].get('calibration_status', 'unknown')}`",
                    *(
                        [
                            f"- 最终报告：`{summary['paths']['final_report_markdown']}`",
                            f"- 最终 Excel：`{summary['paths']['final_prediction_xlsx']}`",
                            f"- 解释状态：`{summary['explanation_status']}`",
                        ]
                        if "final_report_markdown" in summary["paths"] else []
                    ),
                    "",
                    "详细 96 点结果见 `prediction_detail.csv`，数据来源和审计信息见 `run_metadata.json`。",
                ]
            ),
            encoding="utf-8",
        )
        return json_path, markdown_path

    def publish_prediction_draft(self, run_id: str, *, operator_note: str):
        """Promote a reviewed, readiness-approved draft without re-running inference."""
        return RunServiceV1(self.runtime_root).publish_existing_draft(
            run_id,
            operator_note=operator_note,
        )


def _price_history_frame(realtime: pd.Series) -> pd.DataFrame:
    index = pd.DatetimeIndex(realtime.index)
    index = index.tz_localize("Asia/Shanghai") if index.tz is None else index.tz_convert("Asia/Shanghai")
    return pd.DataFrame({"timestamp": index, "value": pd.to_numeric(realtime.to_numpy(), errors="coerce")})


def _weather_panel_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for station_code, frame in sorted(panel.items()):
        item = frame.copy()
        index = pd.DatetimeIndex(item.index)
        item.insert(0, "timestamp", index.tz_localize("Asia/Shanghai") if index.tz is None else index.tz_convert("Asia/Shanghai"))
        item.insert(1, "station_code", station_code)
        frames.append(item.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "station_code"])


def _result_realtime_cutoff(result, *, target_date: str) -> pd.Timestamp:
    metadata_cutoff = result.metadata.get("realtime_cutoff") if isinstance(result.metadata, dict) else None
    if metadata_cutoff:
        return _local_timestamp(metadata_cutoff)
    detail = pd.read_csv(result.detail_path, usecols=lambda column: column == "realtime_cutoff")
    if "realtime_cutoff" in detail and detail["realtime_cutoff"].notna().any():
        return _local_timestamp(str(detail["realtime_cutoff"].dropna().iloc[0]))
    return _local_timestamp(target_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)


def _explanation_input_hash(
    *,
    realtime: pd.Series,
    history_weather: dict[str, pd.DataFrame],
    target_weather: dict[str, pd.DataFrame],
    metadata: dict[str, object],
) -> str:
    digest = sha256()
    digest.update(pd.util.hash_pandas_object(realtime.sort_index(), index=True).to_numpy().tobytes())
    for label, panel in (("history", history_weather), ("target", target_weather)):
        for station_code, frame in sorted(panel.items()):
            digest.update(f"{label}:{station_code}".encode("utf-8"))
            digest.update(pd.util.hash_pandas_object(frame.sort_index(), index=True).to_numpy().tobytes())
    digest.update(
        json.dumps(
            {
                "parameter_checksum": metadata.get("parameter_checksum"),
                "weather_source_hash": metadata.get("weather_source_hash"),
                "target_weather_snapshot_hash": metadata.get("target_weather_snapshot_hash"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("Asia/Shanghai") if timestamp.tz is None else timestamp.tz_convert("Asia/Shanghai")


def _record_final_artifacts(root: Path, *, result, artifacts: FinalReportArtifacts) -> None:
    fields = {
        "realtime_cutoff": result.metadata.get("realtime_cutoff"),
        "final_prediction_png_path": str(artifacts.plot_path.relative_to(root)),
        "final_prediction_xlsx_path": str(artifacts.excel_path.relative_to(root)),
        "final_report_markdown_path": str(artifacts.markdown_path.relative_to(root)),
        "final_report_json_path": str(artifacts.report_json_path.relative_to(root)),
        "whitebox_explanation_json_path": str(artifacts.explanation_json_path.relative_to(root)),
        "explanation_status": artifacts.explanation_status,
        "ai_interpretation_status": "pending",
        "report_revision": 0,
    }
    result.metadata.update(fields)
    metadata_path = root / "runs" / "predictions" / str(result.run_id) / "run_metadata.json"
    if metadata_path.is_file():
        persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
        persisted.update(fields)
        metadata_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
