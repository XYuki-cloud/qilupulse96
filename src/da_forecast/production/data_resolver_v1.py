"""Effective, append-only data view for a deployed Shandong system root."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from da_forecast.config import TIMEZONE
from da_forecast.sources.manual_realtime_xlsx import DEFAULT_MANUAL_WORKBOOK_NAME, read_manual_realtime_prices


@dataclass(frozen=True)
class ReadinessReport:
    target_date: str
    official_publish_allowed: bool
    status: str
    missing_realtime: tuple[str, ...]
    missing_day_ahead: tuple[str, ...]
    missing_weather: tuple[str, ...]
    calendar_confirmed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DataResolverV1:
    """Load stable canonical price snapshots and overlay manual revisions."""

    def __init__(self, root: str | Path, *, manual_workbook: str | Path | None = None) -> None:
        self.root = Path(root)
        self._manual_workbook = (
            Path(manual_workbook).expanduser().resolve()
            if manual_workbook is not None
            else None
        )

    @property
    def manual_workbook_path(self) -> Path | None:
        """Return the operator workbook currently used for realtime overlay."""
        if self._manual_workbook is not None:
            return self._manual_workbook
        candidates = (
            self.root / "data" / DEFAULT_MANUAL_WORKBOOK_NAME,
            self.root.parent / "data" / DEFAULT_MANUAL_WORKBOOK_NAME,
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def load_price(self, kind: Literal["realtime", "day_ahead"]) -> pd.Series:
        names = {
            "realtime": ("realtime_prices_15min.parquet", "realtime_prices.parquet"),
            "day_ahead": ("day_ahead_prices_15min.parquet", "day_ahead_prices.parquet"),
        }[kind]
        candidates = [
            *(self.root / "data" / "curated" / name for name in names),
            *(self.root / "data" / "bootstrap" / "curated" / name for name in names),
            *(self.root / "data" / "raw" / "shandong_all_network" / "SD" / name for name in names),
        ]
        for name in names:
            candidates.extend((self.root / "data" / "bootstrap" / "curated").rglob(name))
            candidates.extend((self.root / "data" / "curated").rglob(name))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        manual = self.manual_workbook_path
        manual_frame = None
        if kind == "realtime" and manual is not None:
            manual_frame = read_manual_realtime_prices(manual)
        if path is None:
            if manual_frame is None:
                raise FileNotFoundError(f"No {kind} price parquet found under {self.root / 'data'}")
            frame = manual_frame
        else:
            frame = pd.read_parquet(path)
        column = next((name for name in ("price_cny_mwh", "day_ahead_price_cny_mwh", "realtime_price_cny_mwh") if name in frame), None)
        if column is None:
            numeric = frame.select_dtypes(include="number").columns.tolist()
            if len(numeric) != 1:
                raise ValueError(f"Could not identify price column in {path}")
            column = numeric[0]
        index = pd.DatetimeIndex(frame.index)
        index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
        series = pd.Series(frame[column].to_numpy(dtype=float), index=index, name="value").sort_index()
        if manual_frame is not None and path is not None:
            manual_index = pd.DatetimeIndex(manual_frame.index)
            manual_values = pd.Series(manual_frame["price_cny_mwh"].to_numpy(dtype=float), index=manual_index)
            series = pd.concat([series, manual_values]).groupby(level=0).last().sort_index()
        return self._apply_manual_revisions(series, kind)

    def _apply_manual_revisions(self, series: pd.Series, kind: str) -> pd.Series:
        directory = self.root / "data" / "manual_daily"
        if not directory.exists():
            return series
        result = series.copy()
        for revision in sorted(directory.glob(f"*/{kind}/revision-*.csv")):
            frame = pd.read_csv(revision)
            if not {"timestamp", "value"}.issubset(frame):
                continue
            index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
            index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
            result.loc[index] = frame["value"].to_numpy(dtype=float)
        return result.sort_index()

    def snapshot_hash(self) -> str:
        digest = hashlib.sha256()
        for path in sorted((self.root / "data").rglob("*.json")):
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(path.read_bytes())
        for path in sorted((self.root / "data").rglob("revision-*.csv")):
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def readiness(
        self,
        *,
        target_date: str | pd.Timestamp,
        weather_complete: bool,
        calendar_confirmed: bool,
        realtime_only: bool = False,
    ) -> ReadinessReport:
        target = _local_day(target_date)
        rt_end = target - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
        rt_index = pd.date_range(rt_end - pd.Timedelta(minutes=15 * (90 * 96 - 1)), rt_end, freq="15min", tz=TIMEZONE)
        bundle_realtime_only = realtime_only
        da_end = target - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
        da_index = pd.date_range(rt_index[0], da_end, freq="15min", tz=TIMEZONE)
        try:
            rt = self.load_price("realtime")
            missing_rt = tuple(value.isoformat() for value in rt_index.difference(rt.index))
        except (FileNotFoundError, ImportError, ValueError) as exc:
            missing_rt = (f"price source unavailable: {exc}",)
        if bundle_realtime_only:
            missing_da = ()
        else:
            try:
                da = self.load_price("day_ahead")
                missing_da = tuple(value.isoformat() for value in da_index.difference(da.index))
            except (FileNotFoundError, ImportError, ValueError) as exc:
                missing_da = (f"price source unavailable: {exc}",)
        missing_weather = () if weather_complete else ("16-city forecast/history panel incomplete",)
        allowed = not (missing_rt or missing_da or missing_weather or not calendar_confirmed)
        status = "ready" if allowed else "blocked"
        return ReadinessReport(target.strftime("%Y-%m-%d"), allowed, status, missing_rt, missing_da, missing_weather, calendar_confirmed)

def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)
    return stamp.normalize()
