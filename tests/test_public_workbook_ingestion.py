from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

import ingest_public_shandong_workbooks as ingestion


MARKET_FILES = tuple(ingestion.PUBLIC_MARKET_FILES)
MANUAL_FILE = ingestion.PUBLIC_MANUAL_FILE


def _quarter_hour_labels() -> list[str]:
    return [*(pd.date_range("2000-01-01 00:15", periods=95, freq="15min").strftime("%H:%M")), "24:00"]


def _market_rows(day: str, *, offset: float = 0.0, include_last: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = _quarter_hour_labels()
    if not include_last:
        labels = labels[:-1]
    target = pd.Timestamp(day)
    current = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    dates = [day] * len(labels)
    common = {
        "目标日期": dates,
        "时刻": labels,
    }
    realtime = pd.DataFrame({**common, "实时出清电价": [300.0 + offset + i for i in range(len(labels))]})
    day_ahead = pd.DataFrame({**common, "日前出清电价": [280.0 + offset + i for i in range(len(labels))]})
    disclosure = pd.DataFrame(
        {
            "当前日期": [current] * len(labels),
            "目标日期": dates,
            "时刻": labels,
            "相隔天数": [1] * len(labels),
            "负荷信息预测": [50000.0 + i for i in range(len(labels))],
            "日前风电总加（MW）": [1000.0 + i for i in range(len(labels))],
            "日前光伏总加（MW）": [2000.0 + i for i in range(len(labels))],
            "联络线信息预测": [3000.0 + i for i in range(len(labels))],
            "竞价空间预测(MW)": [4000.0 + i for i in range(len(labels))],
            "负荷率预测": [0.4] * len(labels),
            "日前地方电厂发电总加（MW）": [5000.0 + i for i in range(len(labels))],
            "日前自备机组发电总加（MW）": [6000.0 + i for i in range(len(labels))],
        }
    ).rename(columns={"日前自备机组发电总加（MW）": "日前自备机组总加（MW）"})
    return realtime, day_ahead, disclosure


def _write_fixture_package(root: Path, *, conflict: bool = False, incomplete: bool = False) -> Path:
    input_dir = root / "public"
    input_dir.mkdir(parents=True)
    for index, filename in enumerate(MARKET_FILES):
        day = "2026-01-01" if conflict and index == 1 else f"2026-01-{index + 1:02d}"
        realtime, day_ahead, disclosure = _market_rows(
            day,
            offset=100.0 if conflict and index == 1 else 0.0,
            include_last=not (incomplete and index == 0),
        )
        with pd.ExcelWriter(input_dir / filename, engine="openpyxl") as writer:
            realtime.to_excel(writer, sheet_name="实时出清数据", index=False)
            day_ahead.to_excel(writer, sheet_name="日前出清数据", index=False)
            disclosure.to_excel(writer, sheet_name="日前披露数据", index=False)
    pd.DataFrame(
        {
            "日期": ["2026-01-01"],
            "时间": ["00:15"],
            "实时电价": [350.0],
        }
    ).to_excel(input_dir / MANUAL_FILE, sheet_name="Sheet1", index=False)
    for workbook_path in input_dir.glob("*.xlsx"):
        workbook = openpyxl.load_workbook(workbook_path)
        workbook.properties.creator = None
        workbook.properties.lastModifiedBy = None
        workbook.properties.title = None
        workbook.properties.subject = None
        workbook.properties.description = None
        workbook.save(workbook_path)
    (input_dir / "MANIFEST.json").write_text(
        json.dumps(ingestion.build_public_manifest(input_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return input_dir


def test_repository_public_data_package_has_exact_files_and_manifest() -> None:
    input_dir = Path(__file__).resolve().parents[1] / "data" / "public"
    assert {path.name for path in input_dir.glob("*.xlsx")} == {*MARKET_FILES, MANUAL_FILE}
    manifest = json.loads((input_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {entry["file"] for entry in manifest["files"]} == {*MARKET_FILES, MANUAL_FILE}
    assert manifest["source_files_are_private"] is True
    assert manifest["original_workbooks_are_not_modified"] is True


def test_repository_public_workbooks_pass_metadata_and_privacy_boundary() -> None:
    input_dir = Path(__file__).resolve().parents[1] / "data" / "public"
    for path in sorted(input_dir.glob("*.xlsx")):
        ingestion._validate_zip_boundary(path)


def test_ingestion_writes_canonical_outputs_only_under_runtime(tmp_path: Path) -> None:
    input_dir = _write_fixture_package(tmp_path)
    runtime_root = tmp_path / ".private-runtime"

    result = ingestion.ingest_public_workbooks(input_dir, runtime_root)

    assert result["check_only"] is False
    output_root = runtime_root / "data" / "raw" / "shandong_all_network" / "SD"
    assert set(path.name for path in output_root.glob("*.parquet")) == {
        "realtime_prices_15min.parquet",
        "day_ahead_prices_15min.parquet",
        "day_ahead_disclosure.parquet",
        "manual_realtime_prices_15min.parquet",
    }
    assert all(Path(path).is_relative_to(runtime_root) for path in result["outputs"].values())
    assert not list(input_dir.glob("*.parquet"))
    realtime = pd.read_parquet(output_root / "realtime_prices_15min.parquet")
    assert len(realtime) == 3 * 96
    assert realtime.index.is_monotonic_increasing
    assert "price_cny_mwh" in realtime.columns


def test_check_only_does_not_write_runtime(tmp_path: Path) -> None:
    input_dir = _write_fixture_package(tmp_path)
    runtime_root = tmp_path / ".private-runtime"

    result = ingestion.ingest_public_workbooks(input_dir, runtime_root, check_only=True)

    assert result["check_only"] is True
    assert not runtime_root.exists()


def test_conflicting_overlap_is_rejected(tmp_path: Path) -> None:
    input_dir = _write_fixture_package(tmp_path, conflict=True)
    with pytest.raises(ValueError, match="conflict"):
        ingestion.ingest_public_workbooks(input_dir, tmp_path / ".private-runtime")


def test_incomplete_market_day_is_rejected(tmp_path: Path) -> None:
    input_dir = _write_fixture_package(tmp_path, incomplete=True)
    with pytest.raises(ValueError, match="96"):
        ingestion.ingest_public_workbooks(input_dir, tmp_path / ".private-runtime")
