"""Audited China adjusted-workday references for production publication."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import requests


class CalendarRuntimeV1:
    """Store downloadable candidates and require an explicit local confirmation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.directory = self.root / "data" / "reference" / "calendar"
        self.directory.mkdir(parents=True, exist_ok=True)

    def confirmed_path(self, year: int) -> Path:
        return self.directory / f"china_workday_overrides_{year}_confirmed.json"

    def is_confirmed(self, year: int) -> bool:
        return self.confirmed_path(year).is_file()

    def install_existing_confirmation(self, source: str | Path, year: int, *, operator_note: str = "bootstrap import") -> Path:
        source_path = Path(source)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        target = self.confirmed_path(year)
        target.write_text(json.dumps({"year": year, "adjusted_workdays": sorted(payload.get("adjusted_workdays", [])), "status": "confirmed", "operator_note": operator_note, "source_sha256": _sha256(source_path)}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def fetch_candidate(self, *, year: int, source_url: str) -> Path:
        response = requests.get(source_url, timeout=30)
        response.raise_for_status()
        text = response.text
        dates = sorted(set(re.findall(fr"{year}年?\s*(\d{{1,2}})月\s*(\d{{1,2}})日", text)))
        # Candidate extraction is intentionally conservative: human confirmation is always required.
        candidate = {
            "year": year, "status": "candidate_unconfirmed", "source_url": source_url,
            "source_host": urlparse(source_url).netloc, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "response_sha256": hashlib.sha256(response.content).hexdigest(), "raw_response": text,
            "date_tokens": [f"{year}-{int(month):02d}-{int(day):02d}" for month, day in dates],
            "adjusted_workdays": [], "parse_notice": "Automatic extraction stores candidates only; operator must enter/confirm actual adjusted workdays.",
        }
        path = self.directory / f"china_workday_overrides_{year}_candidate.json"
        path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def confirm(self, *, year: int, adjusted_workdays: list[str], operator_note: str) -> Path:
        if not operator_note.strip():
            raise ValueError("operator_note is required to confirm calendar overrides")
        path = self.confirmed_path(year)
        payload = {
            "year": year, "adjusted_workdays": sorted(set(adjusted_workdays)), "status": "confirmed",
            "confirmed_at": datetime.now(timezone.utc).isoformat(), "operator_note": operator_note,
        }
        candidate = self.directory / f"china_workday_overrides_{year}_candidate.json"
        if candidate.is_file():
            payload["candidate_sha256"] = _sha256(candidate)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        confirmation = self.directory / f"china_workday_overrides_{year}_confirmation.json"
        confirmation.write_text(json.dumps({"year": year, "confirmed_path": str(path), "confirmed_sha256": _sha256(path), "operator_note": operator_note}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
