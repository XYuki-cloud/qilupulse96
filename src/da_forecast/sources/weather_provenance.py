"""Immutable storage for Open-Meteo production forecast snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ForecastSnapshotArchive:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def store(
        self,
        station_code: str,
        weather: pd.DataFrame,
        payloads: list[dict[str, Any]],
        *,
        issued_at: str | pd.Timestamp,
    ) -> Path:
        issued = pd.Timestamp(issued_at)
        if issued.tz is None:
            issued = issued.tz_localize("Asia/Shanghai")
        safe_issued = issued.strftime("%Y%m%dT%H%M%S%z")
        path = self.root / station_code / f"{safe_issued}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = weather.copy()
        serializable.index.name = "timestamp"
        payload = {
            "weather_kind": "forecast",
            "station_code": station_code,
            "forecast_issued_at": issued.isoformat(),
            "weather": json.loads(serializable.reset_index().to_json(orient="records", date_format="iso")),
            "payloads": payloads,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            suffix = 1
            while True:
                candidate = path.with_stem(f"{path.stem}_{suffix}")
                if not candidate.exists():
                    path = candidate
                    break
                suffix += 1
        path.write_text(encoded, encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path | str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
