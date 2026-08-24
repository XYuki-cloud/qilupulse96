"""Orchestrate readiness, inference, calibration and append-only publication."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd

from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.calibration_bootstrap_v1 import load_bootstrap_calibration_history
from da_forecast.production.calibration_runtime_v1 import calibrate_final, load_settled_ledger_history
from da_forecast.production.data_resolver_v1 import DataResolverV1, ReadinessReport
from da_forecast.production.inference_v1 import infer_qilupulse96
from da_forecast.system.prediction_layer import PredictionLayerRegistry


@dataclass(frozen=True)
class RunResult:
    run_id: str
    publish_status: str
    detail_path: Path | None
    metadata: dict[str, object]


class RunServiceV1:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.resolver = DataResolverV1(self.root)
        self.registry = PredictionLayerRegistry(self.root)

    def readiness(self, *, target_date: str, weather_complete: bool, calendar_confirmed: bool) -> ReadinessReport:
        return self.resolver.readiness(target_date=target_date, weather_complete=weather_complete, calendar_confirmed=calendar_confirmed)

    def run_draft(
        self,
        *,
        bundle: QiluPulse96ProductionBundle,
        inputs,
        target_date: str,
        actual_prices: pd.Series | None = None,
        publish: bool = False,
        weather_kind: str = "forecast",
        readiness: ReadinessReport | None = None,
        input_snapshot_hash: str | None = None,
        target_weather_snapshot_hash: str | None = None,
        calendar_snapshot_hash: str | None = None,
        weather_metadata: dict[str, object] | None = None,
        calibration_history_metadata: dict[str, object] | None = None,
    ) -> RunResult:
        if publish:
            if weather_kind != "forecast":
                raise ValueError("Official publication requires target weather_kind=forecast")
            if readiness is None or not readiness.official_publish_allowed:
                raise ValueError("Official publication is blocked by readiness checks")
        prediction = infer_qilupulse96(bundle, inputs)
        history_parts: list[pd.DataFrame] = []
        if actual_prices is not None:
            history_parts.append(
                load_settled_ledger_history(
                    self.root,
                    actual_prices=actual_prices,
                    bundle_parameter_checksum=bundle.parameter_checksum,
                )
            )
        ledger_path = (calibration_history_metadata or {}).get("calibration_history_ledger_path")
        if ledger_path:
            history_parts.append(load_bootstrap_calibration_history(ledger_path, checksum=bundle.parameter_checksum))
        history = pd.concat([part for part in history_parts if not part.empty], ignore_index=True) if any(not part.empty for part in history_parts) else pd.DataFrame()
        if not history.empty and {"market_date", "period_start"}.issubset(history.columns):
            history = history.drop_duplicates(subset=["market_date", "period_start"], keep="last")
        final, calibration_meta = calibrate_final(prediction, history=history, target_date=target_date)
        calibration_required = getattr(getattr(bundle, "spec", None), "history_extra_dim", None) == 14
        if calibration_required and calibration_meta.get("calibration_status") != "active":
            raise ValueError(
                "生产后处理未激活：realtime-only 模型要求 bias 和 interval calibration 均为 active；"
                f"当前状态={calibration_meta.get('calibration_status')}"
            )
        final["raw_predicted_cny_mwh"] = prediction["predicted_cny_mwh"]
        final["raw_p10_cny_mwh"] = prediction["p10_cny_mwh"]
        final["raw_p50_cny_mwh"] = prediction["p50_cny_mwh"]
        final["raw_p90_cny_mwh"] = prediction["p90_cny_mwh"]
        final["bundle_sha256"] = bundle.bundle_sha256
        final["parameter_checksum"] = bundle.parameter_checksum
        final["input_snapshot_hash"] = input_snapshot_hash
        final["target_weather_snapshot_hash"] = target_weather_snapshot_hash
        final["calendar_snapshot_hash"] = calendar_snapshot_hash
        final["weather_kind"] = weather_kind
        final["bias_interval_calibration_version"] = str(
            calibration_meta.get("bias_interval_calibration_version", "frozen_adaln_bias_interval_v02")
        )
        final["calibration_history_last_date"] = calibration_meta.get("calibration_history_last_date")
        final["calibration_realtime_label_cutoff"] = calibration_meta.get("calibration_realtime_label_cutoff")
        publish_status = "official_published" if publish else "draft"
        final["publish_status"] = publish_status
        run_id = f"{target_date}_{bundle.parameter_checksum[:12]}"
        run_dir = self.root / "runs" / "predictions" / run_id
        suffix = 1
        while run_dir.exists():
            run_id = f"{target_date}_{bundle.parameter_checksum[:12]}_{suffix}"
            run_dir = self.root / "runs" / "predictions" / run_id
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        detail_path = run_dir / "prediction_detail.csv"
        final.to_csv(detail_path, index=False, encoding="utf-8-sig")
        dated_detail_path = run_dir / f"prediction_{target_date}.csv"
        final.to_csv(dated_detail_path, index=False, encoding="utf-8-sig")
        metadata = {
            "run_id": run_id,
            "publish_status": publish_status,
            "calibration": calibration_meta,
            "row_count": int(len(final)),
            "weather_kind": weather_kind,
            "official_publish_allowed": bool(readiness and readiness.official_publish_allowed),
            "input_snapshot_hash": input_snapshot_hash,
            "target_weather_snapshot_hash": target_weather_snapshot_hash,
            "calendar_snapshot_hash": calendar_snapshot_hash,
            "bundle_sha256": bundle.bundle_sha256,
            "parameter_checksum": bundle.parameter_checksum,
        }
        if weather_metadata:
            metadata.update(weather_metadata)
        if calibration_history_metadata:
            metadata.update(calibration_history_metadata)
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return RunResult(run_id, publish_status, detail_path, metadata)

    def write_explanation(
        self,
        *,
        run_id: str,
        target_date: str,
        report,
        output_dir: str | Path | None = None,
        basename: str | None = None,
    ) -> tuple[Path, Path]:
        """Persist a read-only explanation without reopening or changing the prediction."""
        run_dir = self.root / "runs" / "predictions" / run_id
        metadata_path = run_dir / "run_metadata.json"
        if not run_dir.is_dir() or not metadata_path.is_file():
            raise FileNotFoundError(f"Unknown prediction run: {run_id}")
        explanation_dir = Path(output_dir) if output_dir is not None else self.root / "runs" / "explanations" / run_id
        explanation_dir.mkdir(parents=True, exist_ok=True)
        stem = basename or f"explanation_{target_date}"
        markdown_path = explanation_dir / f"{stem}.md"
        json_path = explanation_dir / f"{stem}.json"
        markdown_path.write_text(str(report.markdown), encoding="utf-8")
        json_path.write_text(
            json.dumps(report.payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["explanation_markdown_path"] = str(markdown_path.relative_to(self.root))
        metadata["explanation_json_path"] = str(json_path.relative_to(self.root))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return markdown_path, json_path

    def publish_existing_draft(self, run_id: str, *, operator_note: str) -> RunResult:
        """Atomically promote a reviewed forecast draft without re-running inference.

        A production forecast is first written as a forecast-weather draft.  The
        native confirmation dialog calls this method only after the operator has
        reviewed the ready report.  It deliberately refuses proxy-weather or
        incomplete audits so a post-hoc GUI action cannot manufacture an
        ``official_published`` ledger record.
        """
        if not operator_note.strip():
            raise ValueError("operator_note is required to publish a draft")
        run_dir = self.root / "runs" / "predictions" / run_id
        detail_path = run_dir / "prediction_detail.csv"
        metadata_path = run_dir / "run_metadata.json"
        if not detail_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Unknown prediction run: {run_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("publish_status") != "draft":
            raise ValueError("Only a draft prediction can be published")
        if metadata.get("weather_kind") != "forecast" or not metadata.get("official_publish_allowed", False):
            raise ValueError("This draft cannot be published because its forecast/readiness audit is incomplete")
        detail = pd.read_csv(detail_path)
        if len(detail) != 96 or not detail.get("publish_status", pd.Series(dtype=str)).eq("draft").all():
            raise ValueError("This draft cannot be published because its prediction ledger is invalid")
        detail["publish_status"] = "official_published"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        target_date = str(detail["market_date"].iloc[0])
        dated_detail_path = run_dir / f"prediction_{target_date}.csv"
        detail.to_csv(dated_detail_path, index=False, encoding="utf-8-sig")
        metadata["publish_status"] = "official_published"
        metadata["publish_operator_note"] = operator_note
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return RunResult(run_id=run_id, publish_status="official_published", detail_path=detail_path, metadata=metadata)
