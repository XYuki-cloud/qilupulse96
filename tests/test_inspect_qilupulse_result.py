from __future__ import annotations

import json

from inspect_qilupulse_result import inspect_result


def test_inspector_reports_weather_block_without_a_prediction_package(tmp_path) -> None:
    manifest = tmp_path / "data" / "raw" / "weather_completion" / "weather_completion_20260823_20260822T120000+0800.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "status": "error",
                "target_date": "2026-08-23",
                "error": "target forecast snapshot missing for the exact as-of contract",
            }
        ),
        encoding="utf-8",
    )

    result = inspect_result(tmp_path, target_date="2026-08-23")

    assert result["status"] == "blocked"
    assert result["reason"] == "target weather snapshot missing"
    assert result["paths"]["weather_completion_manifest"] == str(manifest)
