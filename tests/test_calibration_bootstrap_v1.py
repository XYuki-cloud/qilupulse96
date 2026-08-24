from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


def _prices(start: str, days: int) -> pd.Series:
    index = pd.date_range(start, periods=days * 96, freq="15min", tz="Asia/Shanghai")
    return pd.Series(range(len(index)), index=index, name="value", dtype=float)


def _replay(day: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(day, periods=96, freq="15min", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "market_date": index.strftime("%Y-%m-%d"),
            "period_start": index.strftime("%H:%M"),
            "predicted_cny_mwh": 100.0,
            "negative_probability": 0.2,
            "p10_cny_mwh": 50.0,
            "p50_cny_mwh": 100.0,
            "p90_cny_mwh": 150.0,
            "actual_cny_mwh": 90.0,
            "weather_kind": "observed_proxy",
            "realtime_cutoff": (day - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)).isoformat(),
        }
    )


def test_bootstrap_replay_creates_checksum_scoped_ledger(tmp_path):
    from da_forecast.production.calibration_bootstrap_v1 import ensure_realtime_only_calibration_history

    report = ensure_realtime_only_calibration_history(
        tmp_path,
        bundle=SimpleNamespace(parameter_checksum="abc", bundle_sha256="bundle"),
        actual_prices=_prices("2026-01-01", 80),
        target_date="2026-03-21",
        replay_day=_replay,
        required_days=56,
        min_days=14,
    )

    assert report.status == "bootstrap_replay"
    assert report.history_days == 56
    assert report.ledger_path == tmp_path / "data" / "calibration" / "realtime_only" / "abc" / "ledger.csv"
    assert report.ledger_path.is_file()
    assert report.manifest_path.is_file()
    saved = pd.read_csv(report.ledger_path)
    assert saved["bundle_parameter_checksum"].eq("abc").all()
    assert saved["calibration_source"].eq("bootstrap_replay").all()
    assert saved["market_date"].nunique() == 56


def test_bootstrap_rejects_checksum_mismatch_in_existing_ledger(tmp_path):
    from da_forecast.production.calibration_bootstrap_v1 import ensure_realtime_only_calibration_history

    ledger = tmp_path / "data" / "calibration" / "realtime_only" / "abc" / "ledger.csv"
    ledger.parent.mkdir(parents=True)
    _replay(pd.Timestamp("2026-02-01", tz="Asia/Shanghai")).assign(
        bundle_parameter_checksum="wrong", bundle_sha256="bundle", calibration_source="bootstrap_replay"
    ).to_csv(ledger, index=False)

    with pytest.raises(ValueError, match="checksum"):
        ensure_realtime_only_calibration_history(
            tmp_path,
            bundle=SimpleNamespace(parameter_checksum="abc", bundle_sha256="bundle"),
            actual_prices=_prices("2026-01-01", 80),
            target_date="2026-03-21",
            replay_day=_replay,
            required_days=14,
            min_days=14,
        )


def test_bootstrap_blocks_when_fewer_than_minimum_complete_days(tmp_path):
    from da_forecast.production.calibration_bootstrap_v1 import ensure_realtime_only_calibration_history

    with pytest.raises(ValueError, match="at least 14"):
        ensure_realtime_only_calibration_history(
            tmp_path,
            bundle=SimpleNamespace(parameter_checksum="abc", bundle_sha256="bundle"),
            actual_prices=_prices("2026-01-01", 12),
            target_date="2026-01-13",
            replay_day=_replay,
            required_days=56,
            min_days=14,
        )
