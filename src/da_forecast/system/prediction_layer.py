"""Version, candidate, adaptation and prediction-ledger governance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_ID = "QiluPulse-96"
VALID_WEATHER_KINDS = frozenset({"forecast", "actual", "observed_proxy"})


@dataclass(frozen=True)
class RecordedPrediction:
    run_id: str
    detail_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


class PredictionLayerRegistry:
    """Append-only operational registry; no candidate is auto-promoted."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.layer_dir = self.root / "artifacts" / "prediction-layer"
        self.version_dir = self.layer_dir / "versions"
        self.adaptation_dir = self.root / "runs" / "updates"
        self.prediction_dir = self.root / "runs" / "predictions"
        for directory in (self.version_dir, self.adaptation_dir, self.prediction_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def register_version(
        self,
        *,
        version: str,
        artifact_path: str | Path,
        data_snapshot_hash: str,
        operator_note: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a candidate model artifact without changing the default."""
        if not version.strip() or not data_snapshot_hash.strip() or not operator_note.strip() or not kind.strip():
            raise ValueError("version, data_snapshot_hash, operator_note and kind are required")
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        path = self._version_path(version)
        if path.exists():
            raise ValueError(f"Prediction-layer version already exists: {version}")
        manifest = {
            "model_id": MODEL_ID,
            "version": version,
            "kind": kind,
            "status": "candidate",
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": _artifact_sha256(artifact),
            "data_snapshot_hash": data_snapshot_hash,
            "operator_note": operator_note,
            "registered_at": _utc_now(),
            "metadata": dict(metadata or {}),
        }
        _write_json(path, manifest)
        return manifest

    def get_version(self, version: str) -> dict[str, Any]:
        path = self._version_path(version)
        if not path.is_file():
            raise FileNotFoundError(f"Unknown prediction-layer version: {version}")
        return _read_json(path)

    def default_version(self) -> str | None:
        path = self.layer_dir / "default.json"
        return _read_json(path)["version"] if path.is_file() else None

    def promote(self, *, version: str, operator_note: str) -> dict[str, Any]:
        """Explicitly promote one existing candidate; never performed by training."""
        if not operator_note.strip():
            raise ValueError("operator_note is required for manual promotion")
        candidate = self.get_version(version)
        artifact = Path(candidate["artifact_path"])
        if not artifact.exists() or _artifact_sha256(artifact) != candidate["artifact_sha256"]:
            raise ValueError("Candidate artifact is missing or checksum no longer matches its manifest")
        previous = self.default_version()
        if previous and previous != version:
            previous_manifest = self.get_version(previous)
            previous_manifest["status"] = "previous_default"
            previous_manifest["superseded_at"] = _utc_now()
            _write_json(self._version_path(previous), previous_manifest)
        candidate["status"] = "default"
        candidate["promoted_at"] = _utc_now()
        candidate["promotion_note"] = operator_note
        _write_json(self._version_path(version), candidate)
        _write_json(
            self.layer_dir / "default.json",
            {"model_id": MODEL_ID, "version": version, "promoted_at": candidate["promoted_at"], "operator_note": operator_note},
        )
        return candidate

    def create_adaptation_run(
        self,
        *,
        base_model_version: str,
        base_checkpoint_checksum: str,
        training_start: str,
        training_end: str,
        label_cutoff: str,
        validation_start: str,
        validation_end: str,
        weighting_method: str,
        half_life_days: float | None,
        epochs: int,
        patience: int,
        data_snapshot_hash: str,
        operator_note: str,
    ) -> dict[str, Any]:
        """Record a manually authorized adaptation configuration; does not train."""
        base = self.get_version(base_model_version)
        if self.default_version() != base_model_version:
            raise ValueError("Manual adaptation must warm-start from the currently promoted default version")
        if not base_checkpoint_checksum.strip() or not data_snapshot_hash.strip() or not operator_note.strip():
            raise ValueError("base_checkpoint_checksum, data_snapshot_hash and operator_note are required")
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must be positive")
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be positive when supplied")
        run_seed = json.dumps(
            {
                "base": base_model_version,
                "snapshot": data_snapshot_hash,
                "start": training_start,
                "end": training_end,
                "created": _utc_now(),
            },
            sort_keys=True,
        )
        run_id = f"adaptation_{hashlib.sha256(run_seed.encode()).hexdigest()[:16]}"
        manifest = {
            "adaptation_run_id": run_id,
            "status": "configured",
            "base_model_version": base_model_version,
            "base_checkpoint_checksum": base_checkpoint_checksum,
            "training_start": training_start,
            "training_end": training_end,
            "label_cutoff": _local_timestamp(label_cutoff).isoformat(),
            "validation_start": validation_start,
            "validation_end": validation_end,
            "weighting_method": weighting_method,
            "half_life_days": half_life_days,
            "epochs": int(epochs),
            "patience": int(patience),
            "data_snapshot_hash": data_snapshot_hash,
            "operator_note": operator_note,
            "created_at": _utc_now(),
        }
        _write_json(self.adaptation_dir / f"{run_id}.json", manifest)
        return manifest

    def complete_adaptation_run(
        self,
        *,
        adaptation_run_id: str,
        candidate_version: str,
        candidate_artifact_path: str | Path,
        candidate_checkpoint_checksum: str,
        report_path: str | Path,
        operator_note: str,
    ) -> dict[str, Any]:
        """Attach a completed external training result as a reviewable candidate."""
        run_path = self.adaptation_dir / f"{adaptation_run_id}.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"Unknown adaptation run: {adaptation_run_id}")
        run = _read_json(run_path)
        if run["status"] != "configured":
            raise ValueError(f"Adaptation run is not awaiting a candidate: {adaptation_run_id}")
        report = Path(report_path)
        if not report.is_file():
            raise FileNotFoundError(report)
        candidate = self.register_version(
            version=candidate_version,
            artifact_path=candidate_artifact_path,
            data_snapshot_hash=run["data_snapshot_hash"],
            operator_note=operator_note,
            kind="adaptation_candidate",
            metadata={
                "adaptation_run_id": adaptation_run_id,
                "base_model_version": run["base_model_version"],
                "candidate_checkpoint_checksum": candidate_checkpoint_checksum,
                "report_path": str(report.resolve()),
                "report_sha256": _sha256(report),
            },
        )
        run["status"] = "candidate_created"
        run["completed_at"] = _utc_now()
        run["candidate_version"] = candidate_version
        run["candidate_checkpoint_checksum"] = candidate_checkpoint_checksum
        run["report_path"] = str(report.resolve())
        _write_json(run_path, run)
        return candidate

    def record_prediction(
        self,
        prediction: pd.DataFrame,
        *,
        model_version: str,
        parameter_checksum: str,
        data_snapshot_hash: str,
        as_of: str | pd.Timestamp,
        realtime_cutoff: str | pd.Timestamp,
        day_ahead_cutoff: str | pd.Timestamp,
        weather_kind: str,
    ) -> RecordedPrediction:
        """Persist one immutable 96-slot issued forecast and its input audit fields."""
        if weather_kind not in VALID_WEATHER_KINDS:
            raise ValueError(f"weather_kind must be one of {sorted(VALID_WEATHER_KINDS)}")
        frame = _validate_prediction(prediction)
        target = pd.Timestamp(frame["market_date"].iloc[0]).normalize()
        as_of_value = _local_timestamp(as_of)
        realtime = _local_timestamp(realtime_cutoff)
        day_ahead = _local_timestamp(day_ahead_cutoff)
        if as_of_value.normalize() != target.tz_localize("Asia/Shanghai") - pd.Timedelta(days=1) or as_of_value.hour < 12:
            raise ValueError("Prediction as_of must be T-1 12:00-or-later Asia/Shanghai")
        expected_realtime = target.tz_localize("Asia/Shanghai") - pd.Timedelta(days=1) + pd.Timedelta(hours=10, minutes=45)
        expected_day_ahead = target.tz_localize("Asia/Shanghai") - pd.Timedelta(days=2) + pd.Timedelta(hours=23, minutes=45)
        if realtime != expected_realtime or day_ahead != expected_day_ahead:
            raise ValueError("Prediction cutoffs must match the T-1 11:00 contract")
        frame = frame.copy()
        for standard, raw in (
            ("predicted_cny_mwh", "raw_predicted_cny_mwh"),
            ("p10_cny_mwh", "raw_p10_cny_mwh"),
            ("p50_cny_mwh", "raw_p50_cny_mwh"),
            ("p90_cny_mwh", "raw_p90_cny_mwh"),
        ):
            if raw not in frame:
                frame[raw] = frame[standard]
        run_id = f"{target.strftime('%Y-%m-%d')}_{_slug(model_version)}_{parameter_checksum[:12]}"
        run_dir = self.prediction_dir / run_id
        if run_dir.exists():
            raise ValueError(f"Prediction run already exists: {run_id}")
        run_dir.mkdir(parents=True)
        frame["model_id"] = MODEL_ID
        frame["model_version"] = model_version
        frame["parameter_checksum"] = parameter_checksum
        frame["input_snapshot_hash"] = data_snapshot_hash
        frame["as_of"] = as_of_value.isoformat()
        frame["realtime_cutoff"] = realtime.isoformat()
        frame["day_ahead_cutoff"] = day_ahead.isoformat()
        frame["weather_kind"] = weather_kind
        detail_path = run_dir / "prediction_detail.csv"
        frame.to_csv(detail_path, index=False, encoding="utf-8-sig")
        metadata = {
            "run_id": run_id,
            "model_id": MODEL_ID,
            "model_version": model_version,
            "parameter_checksum": parameter_checksum,
            "input_snapshot_hash": data_snapshot_hash,
            "as_of": as_of_value.isoformat(),
            "realtime_cutoff": realtime.isoformat(),
            "day_ahead_cutoff": day_ahead.isoformat(),
            "weather_kind": weather_kind,
            "target_date": target.strftime("%Y-%m-%d"),
            "row_count": 96,
            "detail_sha256": _sha256(detail_path),
            "created_at": _utc_now(),
        }
        metadata_path = run_dir / "run_metadata.json"
        _write_json(metadata_path, metadata)
        return RecordedPrediction(run_id=run_id, detail_path=detail_path, metadata_path=metadata_path, metadata=metadata)

    def _version_path(self, version: str) -> Path:
        return self.version_dir / f"{_slug(version)}.json"


def _validate_prediction(prediction: pd.DataFrame) -> pd.DataFrame:
    required = {"market_date", "period_start", "predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"}
    if not required.issubset(prediction):
        raise ValueError(f"Prediction is missing required fields: {sorted(required - set(prediction))}")
    frame = prediction.copy()
    dates = pd.to_datetime(frame["market_date"], errors="coerce")
    slots = pd.to_datetime(frame["period_start"].astype(str), format="%H:%M", errors="coerce")
    if dates.isna().any() or slots.isna().any() or dates.dt.normalize().nunique() != 1:
        raise ValueError("Prediction must have one valid market_date and HH:MM period_start values")
    frame["_slot"] = slots.dt.hour * 4 + slots.dt.minute // 15
    if len(frame) != 96 or set(frame["_slot"]) != set(range(96)):
        raise ValueError("Prediction must contain exactly 96 unique 15-minute slots")
    numeric = ["predicted_cny_mwh", "negative_probability", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any():
        raise ValueError("Prediction values must be finite")
    if not frame["negative_probability"].between(0, 1).all():
        raise ValueError("negative_probability must be within [0, 1]")
    if not ((frame["p10_cny_mwh"] <= frame["p50_cny_mwh"]) & (frame["p50_cny_mwh"] <= frame["p90_cny_mwh"])).all():
        raise ValueError("Prediction quantiles must satisfy P10 <= P50 <= P90")
    return frame.drop(columns="_slot").sort_values(["market_date", "period_start"], ignore_index=True)


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not result:
        raise ValueError("Version must contain at least one ASCII letter or digit")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Checksum a file or an immutable bundle directory deterministically."""
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, str) and value.rstrip().endswith("Asia/Shanghai"):
        return pd.Timestamp(value.rsplit("Asia/Shanghai", maxsplit=1)[0].strip()).tz_localize("Asia/Shanghai")
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("Asia/Shanghai") if timestamp.tz is None else timestamp.tz_convert("Asia/Shanghai")
