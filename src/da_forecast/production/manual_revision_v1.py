"""Append-only manual data revisions and causal imputation suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from da_forecast.config import TIMEZONE


@dataclass(frozen=True)
class ManualRevision:
    revision_id: str
    data_path: Path
    manifest_path: Path
    sha256: str


class ManualRevisionStore:
    """Persist accepted human edits without mutating any imported data source."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, frame: pd.DataFrame, *, market_date: str | pd.Timestamp, source_kind: str, operator_note: str, accepted_imputation: bool = False) -> ManualRevision:
        if not operator_note.strip():
            raise ValueError("operator_note is required for a manual revision")
        required = {"timestamp", "value"}
        if not required.issubset(frame):
            raise ValueError(f"Manual revision requires {sorted(required)}")
        result = frame.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result["timestamp"] = result["timestamp"].dt.tz_localize(TIMEZONE) if result["timestamp"].dt.tz is None else result["timestamp"].dt.tz_convert(TIMEZONE)
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        if result["timestamp"].isna().any() or result["value"].isna().any() or not np.isfinite(result["value"]).all():
            raise ValueError("Manual timestamps and values must be finite")
        now = datetime.now(timezone.utc)
        seed = f"{source_kind}|{market_date}|{now.isoformat()}|{operator_note}".encode()
        revision_id = f"manual_{hashlib.sha256(seed).hexdigest()[:16]}"
        day = pd.Timestamp(market_date).strftime("%Y-%m-%d")
        directory = self.root / "data" / "manual_daily" / day / source_kind
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"revision-{now.strftime('%Y%m%dT%H%M%SZ')}-{revision_id[-8:]}.csv"
        result.to_csv(path, index=False, encoding="utf-8-sig")
        sha = _sha256(path)
        manifest = {
            "revision_id": revision_id, "source_kind": source_kind, "market_date": day,
            "operator_note": operator_note, "accepted_imputation": bool(accepted_imputation),
            "created_at": now.isoformat(), "source_sha256": sha, "row_count": int(len(result)),
            "data_path": str(path.relative_to(self.root)),
        }
        manifest_path = self.root / "data" / "metadata" / f"manual-{revision_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return ManualRevision(revision_id, path, manifest_path, sha)

    @staticmethod
    def imputation_suggestions(values: pd.Series, *, missing_timestamps: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Causal median suggestions from T-7/T-14/T-21/T-28 only."""
        cutoff = _local(cutoff)
        source = values.copy()
        source.index = pd.DatetimeIndex([_local(value) for value in source.index])
        rows: list[dict[str, object]] = []
        for timestamp in pd.DatetimeIndex(missing_timestamps):
            target = _local(timestamp)
            donors = [target - pd.Timedelta(days=days) for days in (7, 14, 21, 28)]
            allowed = [day for day in donors if day <= cutoff and day in source.index and pd.notna(source.loc[day])]
            donor_values = [float(source.loc[day]) for day in allowed]
            rows.append({
                "timestamp": target.isoformat(), "suggestion": float(np.median(donor_values)) if len(donor_values) >= 2 else np.nan,
                "donor_timestamps": [day.isoformat() for day in allowed], "donor_values": donor_values,
                "status": "available" if len(donor_values) >= 2 else "insufficient_donors",
            })
        return pd.DataFrame(rows)


def _local(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
