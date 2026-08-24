"""Causal bootstrap history for the production post-processing layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from collections.abc import Callable

import pandas as pd

from da_forecast.config import TIMEZONE


REQUIRED_COLUMNS = {
    "market_date", "period_start", "actual_cny_mwh", "predicted_cny_mwh",
    "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh", "negative_probability",
    "bundle_parameter_checksum", "bundle_sha256", "weather_kind", "realtime_cutoff",
    "calibration_source",
}


@dataclass(frozen=True)
class CalibrationHistoryReport:
    status: str
    history_days: int
    history_last_date: str | None
    ledger_path: Path
    manifest_path: Path
    bundle_parameter_checksum: str
    source: str


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)
    return stamp.normalize()


def _series(values: pd.Series) -> pd.Series:
    result = values.astype(float).copy()
    index = pd.DatetimeIndex(result.index)
    result.index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    return result.sort_index()


def _complete_actual_days(actual_prices: pd.Series, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    series = _series(actual_prices)
    series = series.loc[series.index <= cutoff + pd.Timedelta(hours=23, minutes=45)]
    result: list[pd.Timestamp] = []
    for day, frame in series.groupby(series.index.normalize()):
        expected = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
        if len(frame) == 96 and frame.index.equals(expected) and frame.notna().all():
            result.append(day)
    return sorted(result)


def _read_existing(path: Path, checksum: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Calibration ledger is missing columns: {sorted(missing)}")
    if not frame["bundle_parameter_checksum"].astype(str).eq(checksum).all():
        raise ValueError("Calibration ledger bundle parameter checksum mismatch")
    if not frame["calibration_source"].astype(str).eq("bootstrap_replay").all():
        raise ValueError("Calibration ledger contains an unsupported calibration source")
    return frame


def load_bootstrap_calibration_history(path: str | Path, *, checksum: str) -> pd.DataFrame:
    """Load a checksum-scoped bootstrap ledger for the calibrator."""
    return _read_existing(Path(path), checksum)


def _actual_frame(actual_prices: pd.Series, day: pd.Timestamp) -> pd.DataFrame:
    series = _series(actual_prices)
    index = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
    values = series.reindex(index)
    if values.isna().any():
        raise ValueError(f"Missing settled realtime prices for calibration day {day:%Y-%m-%d}")
    return pd.DataFrame({
        "market_date": day.strftime("%Y-%m-%d"),
        "period_start": index.strftime("%H:%M"),
        "actual_cny_mwh": values.to_numpy(dtype=float),
    })


def ensure_realtime_only_calibration_history(
    root: str | Path,
    *,
    bundle,
    actual_prices: pd.Series,
    target_date: str | pd.Timestamp,
    replay_day: Callable[[pd.Timestamp], pd.DataFrame],
    progress: Callable[[str], None] | None = None,
    required_days: int = 56,
    min_days: int = 14,
) -> CalibrationHistoryReport:
    """Ensure a checksum-scoped, fully settled replay ledger exists."""
    if required_days < min_days or min_days < 1:
        raise ValueError("required_days must be at least min_days >= 1")
    target = _local_day(target_date)
    label_cutoff = target - pd.Timedelta(days=2)
    checksum = str(bundle.parameter_checksum)
    base = Path(root) / "data" / "calibration" / "realtime_only" / checksum
    ledger_path = base / "ledger.csv"
    manifest_path = base / "manifest.json"
    existing = _read_existing(ledger_path, checksum)
    eligible = _complete_actual_days(actual_prices, label_cutoff)
    eligible = eligible[-required_days:]
    existing_days = set(pd.to_datetime(existing.get("market_date", pd.Series(dtype=str))).dt.strftime("%Y-%m-%d"))
    missing_days = [day for day in eligible if day.strftime("%Y-%m-%d") not in existing_days]
    if len(existing_days.intersection({day.strftime("%Y-%m-%d") for day in eligible})) < min_days:
        missing_days = eligible
    if missing_days:
        generated: list[pd.DataFrame] = []
        for number, day in enumerate(missing_days, start=1):
            if progress is not None:
                progress(f"补齐历史预测 {number}/{len(missing_days)}：{day:%Y-%m-%d}")
            prediction = replay_day(day)
            if len(prediction) != 96:
                raise ValueError(f"Calibration replay for {day:%Y-%m-%d} did not produce 96 slots")
            actual = _actual_frame(actual_prices, day)
            if "actual_cny_mwh" in prediction:
                prediction = prediction.drop(columns=["actual_cny_mwh"])
            frame = prediction.merge(actual, on=["market_date", "period_start"], how="inner", validate="one_to_one")
            if len(frame) != 96:
                raise ValueError(f"Calibration replay for {day:%Y-%m-%d} has incomplete slots")
            frame["bundle_parameter_checksum"] = checksum
            frame["bundle_sha256"] = str(bundle.bundle_sha256)
            frame["calibration_source"] = "bootstrap_replay"
            frame["weather_kind"] = frame.get("weather_kind", "observed_proxy")
            frame["realtime_cutoff"] = frame.get(
                "realtime_cutoff",
                (day - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)).isoformat(),
            )
            generated.append(frame)
        combined = pd.concat([existing, *generated], ignore_index=True) if not existing.empty else pd.concat(generated, ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        raise ValueError("Calibration history cannot be bootstrapped: no complete settled days")
    combined = combined.drop_duplicates(subset=["market_date", "period_start"], keep="last")
    combined["_day"] = pd.to_datetime(combined["market_date"]).dt.normalize()
    combined = combined.loc[combined["_day"] <= label_cutoff.tz_localize(None)].drop(columns=["_day"])
    keep_days = sorted(pd.to_datetime(combined["market_date"]).dt.normalize().unique())[-required_days:]
    combined = combined.loc[pd.to_datetime(combined["market_date"]).dt.normalize().isin(keep_days)].copy()
    complete_days = combined.groupby("market_date").filter(lambda frame: len(frame) == 96)
    history_days = int(complete_days["market_date"].nunique())
    if history_days < min_days:
        raise ValueError(f"Calibration history has only {history_days} complete days; requires at least {min_days}")
    base.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["market_date", "period_start"]).to_csv(ledger_path, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    manifest = {
        "bundle_parameter_checksum": checksum,
        "bundle_sha256": str(bundle.bundle_sha256),
        "calibration_source": "bootstrap_replay",
        "weather_kind": "observed_proxy",
        "history_days": history_days,
        "history_last_date": str(complete_days["market_date"].max()),
        "target_date": target.strftime("%Y-%m-%d"),
        "label_cutoff": label_cutoff.strftime("%Y-%m-%d"),
        "ledger_sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return CalibrationHistoryReport(
        status="bootstrap_replay" if missing_days else "existing",
        history_days=history_days,
        history_last_date=manifest["history_last_date"],
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        bundle_parameter_checksum=checksum,
        source="bootstrap_replay",
    )
