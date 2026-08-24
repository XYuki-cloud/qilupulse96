"""Auditable, self-contained QiluPulse-96 production bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from da_forecast.models.qilupulse96_v1 import CONTRACT_VERSION, MODEL_ID, QiluPulse96V1Spec, _checksum
from .feature_schema_v1 import feature_schema
from .preprocessing_v1 import PreprocessingStateV1


@dataclass(frozen=True)
class ProductionBundleManifest:
    model_id: str
    model_version: str
    contract_version: str
    parameter_checksum: str
    bundle_sha256: str
    feature_schema_version: str
    station_order: tuple[str, ...]
    training_metadata: dict[str, Any]


@dataclass
class QiluPulse96ProductionBundle:
    """Bundle containing weights, topology, preprocess state and schemas."""

    spec: QiluPulse96V1Spec
    model: torch.nn.Module
    preprocessing: PreprocessingStateV1
    manifest_data: dict[str, Any]
    root: Path | None = None

    @property
    def parameter_checksum(self) -> str:
        return _checksum(self.model)

    @property
    def bundle_sha256(self) -> str:
        return str(self.manifest_data.get("bundle_sha256", ""))

    @classmethod
    def from_artifact(cls, artifact_path: str | Path, *, preprocessing: PreprocessingStateV1 | None = None, training_metadata: dict[str, Any] | None = None):
        payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError("Source checkpoint is missing state_dict")
        if payload.get("model_id") not in (None, MODEL_ID):
            raise ValueError("Not a QiluPulse-96 artifact")
        spec = QiluPulse96V1Spec(**payload.get("spec", {
            "station_variable_dim": 25, "history_extra_dim": 18, "target_extra_dim": 19,
            "n_stations": 16,
        }))
        model = spec.build_model()
        model.load_state_dict(payload["state_dict"])
        checksum = _checksum(model)
        if payload.get("parameter_checksum") and payload["parameter_checksum"] != checksum:
            raise ValueError("Source checkpoint checksum mismatch")
        return cls(spec, model, preprocessing or PreprocessingStateV1.identity(history_extra_dim=16, target_extra_dim=14, station_dim=25), {
            "model_id": MODEL_ID, "model_version": "1.0.0", "contract_version": CONTRACT_VERSION,
            "parameter_checksum": checksum, "training_metadata": {**payload.get("training_metadata", {}), **(training_metadata or {})},
            "feature_schema": feature_schema(), "station_order": [], "weather_kind": "observed_proxy", "production_weather_kind": "forecast",
        })

    def save(self, root: str | Path) -> Path:
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination / "model_state.pt")
        self.preprocessing.save_npz(destination / "preprocessing.npz")
        (destination / "feature_schema.json").write_text(json.dumps(self.manifest_data.get("feature_schema", feature_schema()), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (destination / "station_schema.json").write_text(json.dumps({"station_order": self.manifest_data.get("station_order", [])}, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "calibration_config.json").write_text(json.dumps(self.manifest_data.get("calibration_config", {}), ensure_ascii=False, indent=2), encoding="utf-8")
        files = [destination / name for name in ("model_state.pt", "preprocessing.npz", "feature_schema.json", "station_schema.json", "calibration_config.json")]
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        manifest = {**self.manifest_data, "parameter_checksum": self.parameter_checksum, "bundle_sha256": digest.hexdigest(), "spec": asdict(self.spec), "bundle_format": "qilupulse96_production_bundle_v1"}
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (destination / "bundle.sha256").write_text(manifest["bundle_sha256"] + "\n", encoding="ascii")
        self.root = destination
        self.manifest_data = manifest
        return destination

    @classmethod
    def load(cls, root: str | Path) -> "QiluPulse96ProductionBundle":
        destination = Path(root)
        if not destination.is_dir():
            raise FileNotFoundError(
                f"QiluPulse-96 bundle directory not found: {destination}. "
                "The public source release does not include production weights."
            )
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"QiluPulse-96 bundle manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_format") != "qilupulse96_production_bundle_v1":
            raise ValueError("Unsupported production bundle format")
        spec = QiluPulse96V1Spec(**manifest["spec"])
        model = spec.build_model()
        state = torch.load(destination / "model_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        if _checksum(model) != manifest.get("parameter_checksum"):
            raise ValueError("Production bundle parameter checksum mismatch")
        expected = (destination / "bundle.sha256").read_text(encoding="ascii").strip()
        files = [destination / name for name in ("model_state.pt", "preprocessing.npz", "feature_schema.json", "station_schema.json", "calibration_config.json")]
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        if digest.hexdigest() != expected or expected != manifest.get("bundle_sha256"):
            raise ValueError("Production bundle content checksum mismatch")
        preprocessing = PreprocessingStateV1.load_npz(destination / "preprocessing.npz")
        return cls(spec, model, preprocessing, manifest, destination)
