from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from da_forecast.production.data_resolver_v1 import ReadinessReport


@dataclass(frozen=True)
class _Bundle:
    bundle_sha256: str = "bundle-sha"
    parameter_checksum: str = "parameter-checksum"


@dataclass(frozen=True)
class _RealtimeOnlyBundle(_Bundle):
    spec: object = field(default_factory=lambda: SimpleNamespace(history_extra_dim=14))


def _prediction(target_date: str) -> pd.DataFrame:
    index = pd.date_range(target_date, periods=96, freq="15min", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "market_date": index.strftime("%Y-%m-%d"),
            "period_start": index.strftime("%H:%M"),
            "predicted_cny_mwh": 300.0,
            "negative_probability": 0.2,
            "p10_cny_mwh": 200.0,
            "p50_cny_mwh": 300.0,
            "p90_cny_mwh": 400.0,
            "normalization_center": 280.0,
            "normalization_scale": 30.0,
        }
    )


def _readiness(*, allowed: bool) -> ReadinessReport:
    return ReadinessReport(
        target_date="2026-08-19",
        official_publish_allowed=allowed,
        status="ready" if allowed else "blocked",
        missing_realtime=(),
        missing_day_ahead=(),
        missing_weather=(),
        calendar_confirmed=True,
    )


def _stub_inference_and_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    import da_forecast.production.run_service_v1 as module

    monkeypatch.setattr(module, "infer_qilupulse96", lambda _bundle, _inputs: _prediction("2026-08-19"))

    def calibrate(frame: pd.DataFrame, **_kwargs):
        result = frame.copy()
        for name in ("predicted_cny_mwh", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"):
            result[f"raw_{name}"] = result[name]
            result[f"bias_{name}"] = result[name]
        result["bias_status"] = "insufficient_history_fallback"
        result["bias_group"] = "night"
        result["bias_correction_cny_mwh"] = 0.0
        result["bias_history_days"] = 0
        result["interval_status"] = "insufficient_history_fallback"
        result["interval_history_days"] = 0
        result["interval_lower_expansion_cny_mwh"] = 0.0
        result["interval_upper_expansion_cny_mwh"] = 0.0
        result["calibration_history_last_date"] = None
        result["calibration_realtime_label_cutoff"] = "2026-08-17 23:45 CST"
        return result, {"bias_interval_calibration_version": "frozen_adaln_bias_interval_v02"}

    monkeypatch.setattr(module, "calibrate_final", calibrate)


def test_run_service_marks_confirmed_forecast_run_official_and_keeps_audit_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    result = RunServiceV1(tmp_path).run_draft(
        bundle=_Bundle(),
        inputs=object(),
        target_date="2026-08-19",
        publish=True,
        weather_kind="forecast",
        readiness=_readiness(allowed=True),
        input_snapshot_hash="input-hash",
        target_weather_snapshot_hash="weather-hash",
        calendar_snapshot_hash="calendar-hash",
    )

    assert result.publish_status == "official_published"
    assert result.metadata["publish_status"] == "official_published"
    assert result.detail_path is not None and result.detail_path.is_file()
    detail = pd.read_csv(result.detail_path)
    assert len(detail) == 96
    for column, expected in {
        "bundle_sha256": "bundle-sha",
        "parameter_checksum": "parameter-checksum",
        "input_snapshot_hash": "input-hash",
        "target_weather_snapshot_hash": "weather-hash",
        "calendar_snapshot_hash": "calendar-hash",
        "publish_status": "official_published",
        "bias_interval_calibration_version": "frozen_adaln_bias_interval_v02",
        "calibration_history_last_date": "",
    }.items():
        values = detail[column].fillna("").astype(str)
        assert values.eq(expected).all()
    run_dir = result.detail_path.parent
    assert (run_dir / "prediction_2026-08-19.csv").is_file()
    assert json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["bundle_sha256"] == "bundle-sha"


def test_run_service_rejects_observed_proxy_or_blocked_readiness_before_writing_official_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    service = RunServiceV1(tmp_path)
    with pytest.raises(ValueError, match="weather_kind=forecast"):
        service.run_draft(
            bundle=_Bundle(), inputs=object(), target_date="2026-08-19", publish=True,
            weather_kind="observed_proxy", readiness=_readiness(allowed=True),
        )
    with pytest.raises(ValueError, match="readiness"):
        service.run_draft(
            bundle=_Bundle(), inputs=object(), target_date="2026-08-19", publish=True,
            weather_kind="forecast", readiness=_readiness(allowed=False),
        )
    assert not list((tmp_path / "runs" / "predictions").glob("*/run_metadata.json"))


def test_run_service_uses_a_unique_run_directory_without_overwriting_previous_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    service = RunServiceV1(tmp_path)
    first = service.run_draft(bundle=_Bundle(), inputs=object(), target_date="2026-08-19")
    second = service.run_draft(bundle=_Bundle(), inputs=object(), target_date="2026-08-19")

    assert first.run_id != second.run_id
    assert first.detail_path is not None and first.detail_path.is_file()
    assert second.detail_path is not None and second.detail_path.is_file()


def test_run_service_writes_read_only_explanation_beside_the_same_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1
    from da_forecast.system.explanation import ExplanationReport

    _stub_inference_and_calibration(monkeypatch)
    service = RunServiceV1(tmp_path)
    run = service.run_draft(bundle=_Bundle(), inputs=object(), target_date="2026-08-19")
    report = ExplanationReport(payload={"claims": [{"claim_id": "market-1"}]}, markdown="# 白箱解释\n")

    markdown_path, json_path = service.write_explanation(
        run_id=run.run_id,
        target_date="2026-08-19",
        report=report,
    )

    assert markdown_path.read_text(encoding="utf-8") == "# 白箱解释\n"
    assert json.loads(json_path.read_text(encoding="utf-8"))["claims"][0]["claim_id"] == "market-1"
    metadata = json.loads((run.detail_path.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["explanation_markdown_path"].endswith("explanation_2026-08-19.md")
    assert metadata["explanation_json_path"].endswith("explanation_2026-08-19.json")


def test_run_service_promotes_an_existing_ready_forecast_draft_without_reinference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    service = RunServiceV1(tmp_path)
    draft = service.run_draft(
        bundle=_Bundle(),
        inputs=object(),
        target_date="2026-08-19",
        weather_kind="forecast",
        readiness=_readiness(allowed=True),
        input_snapshot_hash="input-hash",
        target_weather_snapshot_hash="weather-hash",
        calendar_snapshot_hash="calendar-hash",
    )

    published = service.publish_existing_draft(draft.run_id, operator_note="复核后正式发布")

    assert published.publish_status == "official_published"
    assert pd.read_csv(draft.detail_path)["publish_status"].eq("official_published").all()
    metadata = json.loads((draft.detail_path.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["publish_status"] == "official_published"
    assert metadata["publish_operator_note"] == "复核后正式发布"


def test_run_service_never_promotes_a_draft_without_forecast_and_ready_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    service = RunServiceV1(tmp_path)
    draft = service.run_draft(bundle=_Bundle(), inputs=object(), target_date="2026-08-19", weather_kind="observed_proxy")

    with pytest.raises(ValueError, match="cannot be published"):
        service.publish_existing_draft(draft.run_id, operator_note="不能发布")

    assert pd.read_csv(draft.detail_path)["publish_status"].eq("draft").all()


def test_realtime_only_run_requires_active_postprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from da_forecast.production.run_service_v1 import RunServiceV1

    _stub_inference_and_calibration(monkeypatch)
    with pytest.raises(ValueError, match="后处理未激活"):
        RunServiceV1(tmp_path).run_draft(
            bundle=_RealtimeOnlyBundle(), inputs=object(), target_date="2026-08-19",
        )
