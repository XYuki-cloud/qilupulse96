from __future__ import annotations

from datetime import datetime, time, timedelta
import warnings

import pandas as pd
import pytest

from da_forecast.sources.manual_realtime_xlsx import parse_manual_realtime_prices


def test_manual_workbook_normalizes_excel_2400_sentinel_to_same_market_day_last_slot():
    raw = pd.DataFrame(
        {
            "日期": ["2026-08-20", "2026-08-21"],
            "时间": [time(23, 45), datetime(1900, 1, 1)],
            "实时电价": [480.0, 460.0],
        }
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-20 23:30", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-21 23:45", tz="Asia/Shanghai"),
    ]


def test_manual_workbook_rejects_unbracketed_next_day_midnight_without_layout_evidence():
    raw = pd.DataFrame(
        {
            "目标日期": ["2026-08-21"],
            "时刻": ["00:00"],
            "实时出清电价": [460.0],
        }
    )

    with pytest.raises(ValueError, match="cannot determine cross-midnight slot"):
        parse_manual_realtime_prices(raw)


def test_manual_workbook_detects_full_period_start_day_after_2400_correction():
    labels = ["00:00"] + [stamp.strftime("%H:%M") for stamp in pd.date_range("2026-08-21 00:15", periods=95, freq="15min")]
    raw = pd.DataFrame(
        {"日期": ["2026-08-21"] * 96, "时间": labels, "实时电价": list(range(96))}
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.min() == pd.Timestamp("2026-08-21 00:00", tz="Asia/Shanghai")
    assert result.index.max() == pd.Timestamp("2026-08-21 23:45", tz="Asia/Shanghai")


def test_manual_workbook_keeps_first_duplicate_normalized_slot_and_warns():
    raw = pd.DataFrame(
        {
            "目标日期": ["2026-08-20", "2026-08-20"],
            "时刻": ["24:00", datetime(1900, 1, 1)],
            "实时电价": [460.0, 460.0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = parse_manual_realtime_prices(raw)
    assert result.loc[pd.Timestamp("2026-08-20 23:45", tz="Asia/Shanghai"), "price_cny_mwh"] == 460.0
    assert any("duplicate slot" in str(item.message) for item in caught)


def test_manual_workbook_accepts_openpyxl_timedelta_2400_and_ignores_blank_rows():
    raw = pd.DataFrame(
        {
            "日期": ["2026-08-21", "2026-08-21", "2026-08-21"],
            "时间": [time(23, 45), timedelta(days=1), timedelta(days=1)],
            "实时电价": [460.0, 461.0, None],
        }
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-21 23:30", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-21 23:45", tz="Asia/Shanghai"),
    ]
    assert result["price_cny_mwh"].tolist() == [460.0, 461.0]


@pytest.mark.parametrize(
    ("middle_date", "middle_time"),
    [
        ("2026-08-16", "24:00"),
        ("2026-08-17", "00:00"),
        ("2026-08-16", "00:00"),
        ("2026-08-17", "23:45"),
    ],
)
def test_manual_workbook_reconstructs_the_cross_midnight_slot_from_neighbors(middle_date, middle_time):
    raw = pd.DataFrame(
        {
            "日期": ["2026-08-16", middle_date, "2026-08-17"],
            "时间": ["23:45", middle_time, "00:15"],
            "实时电价": [460.0, 461.0, 462.0],
        }
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-16 23:30", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-16 23:45", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-17 00:00", tz="Asia/Shanghai"),
    ]


def test_manual_workbook_ignores_blank_prefilled_rows_after_available_prices():
    raw = pd.DataFrame(
        {
            "日期": ["2026-08-21", "2026-08-21"],
            "时间": ["15:00", timedelta(days=1)],
            "实时电价": [431.9, None],
        }
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.tolist() == [pd.Timestamp("2026-08-21 14:45", tz="Asia/Shanghai")]
    assert result["price_cny_mwh"].tolist() == [431.9]


def test_manual_workbook_uses_blank_prefilled_layout_to_preserve_partial_period_start_day():
    times = [timedelta(days=1)] + list(pd.date_range("2026-08-21 00:15", periods=95, freq="15min").time)
    raw = pd.DataFrame(
        {
            "日期": ["2026-08-21"] * 96,
            "时间": times,
            "实时电价": [460.06, 461.0] + [None] * 94,
        }
    )

    result = parse_manual_realtime_prices(raw)

    assert result.index.tolist() == [
        pd.Timestamp("2026-08-21 00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-21 00:15", tz="Asia/Shanghai"),
    ]


def test_resolver_overlays_manual_workbook_without_requiring_day_ahead(tmp_path):
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    manual = tmp_path / "data" / "manual_realtime_prices.xlsx"
    manual.parent.mkdir(parents=True)
    pd.DataFrame(
        {"目标日期": ["2026-08-20"], "时刻": ["00:15"], "实时电价": [456.0]}
    ).to_excel(manual, index=False)
    resolver = DataResolverV1(tmp_path)

    series = resolver.load_price("realtime")

    assert series.loc[pd.Timestamp("2026-08-20 00:00", tz="Asia/Shanghai")] == 456.0


def test_resolver_overlays_cross_midnight_manual_record_at_reconstructed_slot(tmp_path):
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    curated = tmp_path / "data" / "curated"
    curated.mkdir(parents=True)
    baseline_index = pd.date_range("2026-08-16 23:30", periods=3, freq="15min", tz="Asia/Shanghai")
    pd.DataFrame({"price_cny_mwh": [400.0, 401.0, 402.0]}, index=baseline_index).to_parquet(
        curated / "realtime_prices_15min.parquet"
    )
    pd.DataFrame(
        {
            "日期": ["2026-08-16", "2026-08-17", "2026-08-17"],
            "时间": ["23:45", "23:45", "00:15"],
            "实时电价": [460.0, 461.0, 462.0],
        }
    ).to_excel(tmp_path / "data" / "manual_realtime_prices.xlsx", index=False)

    series = DataResolverV1(tmp_path).load_price("realtime")

    assert series.loc[pd.Timestamp("2026-08-16 23:45", tz="Asia/Shanghai")] == 461.0


def test_resolver_can_load_manual_workbook_as_the_only_realtime_source(tmp_path):
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    manual = tmp_path / "data" / "manual_realtime_prices.xlsx"
    manual.parent.mkdir(parents=True)
    pd.DataFrame(
        {"目标日期": ["2026-08-20"], "时刻": ["00:15"], "实时电价": [456.0]}
    ).to_excel(manual, index=False)

    series = DataResolverV1(tmp_path).load_price("realtime")

    assert len(series) == 1
    assert series.index[0] == pd.Timestamp("2026-08-20 00:00", tz="Asia/Shanghai")


def test_resolver_finds_manual_workbook_in_sibling_data_directory(tmp_path):
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    manual = tmp_path / "data" / "manual_realtime_prices.xlsx"
    manual.parent.mkdir(parents=True)
    pd.DataFrame(
        {"日期": ["2026-08-20"], "时间": ["00:15"], "实时电价": [456.0]}
    ).to_excel(manual, index=False)
    project_root = tmp_path / "qilupulse96-demo"
    project_root.mkdir()

    series = DataResolverV1(project_root).load_price("realtime")

    assert series.loc[pd.Timestamp("2026-08-20 00:00", tz="Asia/Shanghai")] == 456.0


def test_resolver_accepts_an_explicit_manual_workbook_outside_runtime_root(tmp_path):
    from da_forecast.production.data_resolver_v1 import DataResolverV1

    runtime_root = tmp_path / "runtime"
    manual = tmp_path / "operator-input.xlsx"
    pd.DataFrame(
        {"目标日期": ["2026-08-22"], "时间": ["00:15"], "实时电价": [489.19]}
    ).to_excel(manual, index=False)

    series = DataResolverV1(runtime_root, manual_workbook=manual).load_price("realtime")

    assert series.loc[pd.Timestamp("2026-08-22 00:00", tz="Asia/Shanghai")] == 489.19
    assert DataResolverV1(runtime_root, manual_workbook=manual).manual_workbook_path == manual


def test_read_manual_workbook_reports_actionable_openpyxl_error(tmp_path, monkeypatch):
    import da_forecast.sources.manual_realtime_xlsx as module

    workbook = tmp_path / "manual_realtime_prices.xlsx"
    workbook.write_bytes(b"placeholder")

    def missing_engine(*_args, **_kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl.")

    monkeypatch.setattr(module.pd, "read_excel", missing_engine)

    with pytest.raises(ImportError, match="读取人工实时价格 Excel 需要 openpyxl") as caught:
        module.read_manual_realtime_prices(workbook)
    assert "uv pip install --python" in str(caught.value)
