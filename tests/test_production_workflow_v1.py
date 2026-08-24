from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Spec
from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1
from da_forecast.production.workflow_v1 import ProductionWorkflow
from run_qilupulse96_production import main as production_cli_main


def test_workflow_writes_structured_llm_report(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "predictions" / "run-1"
    run_dir.mkdir(parents=True)
    detail = pd.DataFrame(
        {
            "period_start": ["00:00", "00:15", "00:30"],
            "predicted_cny_mwh": [10.0, 25.0, 5.0],
            "negative_probability": [0.1, 0.2, 0.8],
            "p10_cny_mwh": [0.0, 5.0, -10.0],
            "p90_cny_mwh": [20.0, 40.0, 15.0],
        }
    )
    detail_path = run_dir / "prediction_detail.csv"
    detail.to_csv(detail_path, index=False)
    result = SimpleNamespace(
        run_id="run-1",
        publish_status="draft",
        detail_path=detail_path,
        metadata={
            "calibration": {"calibration_status": "active"},
            "weather_source_hash": "weather-hash",
        },
    )

    json_path, markdown_path = ProductionWorkflow(tmp_path).write_report(
        result, target_date="2026-08-23", as_of="2026-08-22T12:00:00+08:00"
    )

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["row_count"] == 3
    assert summary["peak"]["period_start"] == "00:15"
    assert summary["valley"]["period_start"] == "00:30"
    assert summary["calibration"]["calibration_status"] == "active"
    assert "最高负价概率" in markdown_path.read_text(encoding="utf-8")


def test_workflow_keeps_final_report_when_whitebox_explanation_is_unavailable(tmp_path) -> None:
    slots = pd.date_range("2026-08-23", periods=96, freq="15min")
    detail = pd.DataFrame(
        {
            "market_date": slots.strftime("%Y-%m-%d"),
            "period_start": slots.strftime("%H:%M"),
            "predicted_cny_mwh": 300.0,
            "negative_probability": 0.1,
            "p10_cny_mwh": 200.0,
            "p50_cny_mwh": 300.0,
            "p90_cny_mwh": 400.0,
            "raw_predicted_cny_mwh": 250.0,
            "raw_p10_cny_mwh": 150.0,
            "raw_p50_cny_mwh": 250.0,
            "raw_p90_cny_mwh": 350.0,
            "bias_status": "active",
            "interval_status": "active",
        }
    )
    run_dir = tmp_path / "runs" / "predictions" / "run-2"
    run_dir.mkdir(parents=True)
    detail_path = run_dir / "prediction_detail.csv"
    detail.to_csv(detail_path, index=False)
    result = SimpleNamespace(
        run_id="run-2",
        publish_status="draft",
        detail_path=detail_path,
        metadata={"calibration": {"calibration_status": "active", "calibration_history_days": 56}},
    )

    artifacts = ProductionWorkflow(tmp_path).write_final_artifacts(
        result,
        target_date="2026-08-23",
        as_of="2026-08-22T12:00:00+08:00",
        realtime=pd.Series(dtype=float),
        history_weather={},
        target_weather={},
    )

    assert artifacts.explanation_status == "unavailable"
    assert artifacts.excel_path.is_file()
    assert "解释层状态：`unavailable`" in artifacts.markdown_path.read_text(encoding="utf-8")


def test_workflow_uses_an_explicit_bundle_without_changing_the_default(tmp_path) -> None:
    spec = QiluPulse96V1Spec(
        station_variable_dim=25,
        history_extra_dim=14,
        target_extra_dim=19,
        n_stations=16,
    )
    bundle = QiluPulse96ProductionBundle(
        spec=spec,
        model=spec.build_model(),
        preprocessing=PreprocessingStateV1.identity(history_extra_dim=14, target_extra_dim=14),
        manifest_data={"feature_schema": {"price_features": "realtime_only"}},
    )
    path = bundle.save(tmp_path / "candidate")
    (tmp_path / "artifacts" / "prediction-layer").mkdir(parents=True)
    (tmp_path / "artifacts" / "prediction-layer" / "default.json").write_text(
        json.dumps({"bundle_path": "missing-default"}), encoding="utf-8"
    )

    resolved = ProductionWorkflow(tmp_path).resolve_bundle(path)

    assert resolved.root == path
    assert resolved.parameter_checksum == bundle.parameter_checksum


def test_public_workflow_requires_an_explicit_bundle(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--bundle-path"):
        ProductionWorkflow(tmp_path).resolve_bundle()


def test_workflow_separates_code_root_from_runtime_root(tmp_path) -> None:
    from da_forecast.production.workflow_v1 import ProductionWorkflow

    code_root = tmp_path / "qilupulse96"
    runtime_root = tmp_path / ".private-runtime"

    workflow = ProductionWorkflow(code_root, runtime_root=runtime_root)

    assert workflow.root == code_root
    assert workflow.runtime_root == runtime_root
    assert workflow.resolver.root == runtime_root
    assert workflow.registry.root == runtime_root


def test_public_cli_reports_a_missing_bundle_as_blocked(tmp_path, capsys) -> None:
    exit_code = production_cli_main(
        [
            "--root",
            str(tmp_path),
            "--target-date",
            "2026-08-23",
            "--as-of",
            "2026-08-22T12:00:00+08:00",
            "--bundle-path",
            "missing-bundle",
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert error["status"] == "blocked"
    assert error["error_type"] == "FileNotFoundError"
    assert "bundle directory not found" in error["message"]
