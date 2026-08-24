"""Versioned inference-kernel contract for QiluPulse-96 v1.0.

The kernel owns model topology, checkpoint serialization and the final 96-row
prediction schema.  Data collection, feature construction and weather-Forecast
snapshot retrieval deliberately remain outside this module so they can be
audited independently at the T-1 12:00 decision boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from da_forecast.forecasting.realtime_longseq_v01 import validate_tminus1_1100_contract
from da_forecast.models.longseq_v01 import SpatialTemporalTransformer


MODEL_ID = "QiluPulse-96"
VERSION = "1.0.0"
CONTRACT_VERSION = "realtime_tminus1_1100_endpoint_v1"

# The v1.0 model ID names one evaluated architecture and one feature schema.
    # A capacity or schema change must receive a new version rather than silently
    # loading under the same operational contract.  The 14-column variant is the
    # retrained production contract with no day-ahead-price features.
_MAINLINE_SPEC = {
    "station_variable_dim": 25,
    "history_extra_dim": 18,
    "target_extra_dim": 19,
    "n_stations": 16,
    "d_model": 64,
    "nhead": 4,
    "patch_size": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.2,
    "conditioning": "adaln",
    "state_dim": 5,
}


@dataclass(frozen=True)
class QiluPulse96V1Spec:
    """Frozen neural topology for a QiluPulse-96 v1.0 artifact."""

    station_variable_dim: int
    history_extra_dim: int
    target_extra_dim: int
    n_stations: int
    d_model: int = 64
    nhead: int = 4
    patch_size: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    conditioning: str = "adaln"
    state_dim: int = 5

    def __post_init__(self) -> None:
        actual = asdict(self)
        mismatches = {
            key: (actual[key], expected)
            for key, expected in _MAINLINE_SPEC.items()
            if actual[key] != expected and not (key == "history_extra_dim" and actual[key] == 14)
        }
        if mismatches:
            detail = ", ".join(f"{key}={received!r} (expected {expected!r})" for key, (received, expected) in mismatches.items())
            raise ValueError(f"QiluPulse-96 v1.0 requires the fixed mainline topology and feature schema: {detail}")

    def build_model(self) -> SpatialTemporalTransformer:
        return SpatialTemporalTransformer(**asdict(self))


def _checksum(model: torch.nn.Module) -> str:
    return hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()


@dataclass
class QiluPulse96V1Artifact:
    """Self-describing model-state artifact; preprocessing is supplied by the adapter."""

    spec: QiluPulse96V1Spec
    model: SpatialTemporalTransformer
    training_metadata: dict[str, Any]
    model_id: str = MODEL_ID
    version: str = VERSION

    @property
    def parameter_checksum(self) -> str:
        return _checksum(self.model)

    def manifest(self) -> dict[str, Any]:
        """Return the model and decision-time contract without running inference."""
        return {
            "model_id": self.model_id,
            "model_version": self.version,
            "contract_version": CONTRACT_VERSION,
            "output_slots": 96,
            "spec": asdict(self.spec),
            "training_metadata": dict(self.training_metadata),
            "parameter_checksum": self.parameter_checksum,
        }

    @classmethod
    def create(
        cls, spec: QiluPulse96V1Spec, *, training_metadata: dict[str, Any] | None = None
    ) -> "QiluPulse96V1Artifact":
        return cls(spec=spec, model=spec.build_model(), training_metadata=dict(training_metadata or {}))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "version": self.version,
                "spec": asdict(self.spec),
                "state_dict": self.model.state_dict(),
                "training_metadata": self.training_metadata,
                "parameter_checksum": self.parameter_checksum,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "QiluPulse96V1Artifact":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("model_id") != MODEL_ID or payload.get("version") != VERSION:
            raise ValueError("Artifact is not a QiluPulse-96 v1.0 model bundle")
        spec = QiluPulse96V1Spec(**payload["spec"])
        model = spec.build_model()
        model.load_state_dict(payload["state_dict"])
        artifact = cls(spec=spec, model=model, training_metadata=dict(payload.get("training_metadata", {})))
        if payload.get("parameter_checksum") != artifact.parameter_checksum:
            raise ValueError("Artifact parameter checksum does not match serialized model state")
        return artifact


def normalized_output_frame(
    *,
    point: np.ndarray,
    negative_probability: np.ndarray,
    quantiles: np.ndarray,
    normalization_center: float,
    normalization_scale: float,
    target_date: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
    parameter_checksum: str,
) -> pd.DataFrame:
    """Convert one normalized model output into the audited 96-row v1.0 schema."""
    contract = validate_tminus1_1100_contract(
        target_date=target_date,
        as_of=_coerce_as_of_timestamp(as_of),
    )
    point_values = np.asarray(point, dtype=float).reshape(-1)
    probability = np.asarray(negative_probability, dtype=float).reshape(-1)
    quantile_values = np.asarray(quantiles, dtype=float)
    if point_values.shape != (96,) or probability.shape != (96,) or quantile_values.shape != (96, 3):
        raise ValueError("QiluPulse-96 v1.0 requires exactly 96 point, probability and three-quantile outputs")
    if not np.isfinite(point_values).all() or not np.isfinite(probability).all() or not np.isfinite(quantile_values).all():
        raise ValueError("Normalized model output must be finite")
    if not np.isfinite(normalization_center) or not np.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("normalization_center and positive normalization_scale are required")
    if not np.logical_and(probability >= 0.0, probability <= 1.0).all():
        raise ValueError("negative_probability must be within [0, 1]")
    values = quantile_values * float(normalization_scale) + float(normalization_center)
    values.sort(axis=1)
    target_index = pd.date_range(contract.target_date, periods=96, freq="15min", tz=contract.target_date.tz)
    result = pd.DataFrame(
        {
            "market_date": contract.target_date.strftime("%Y-%m-%d"),
            "period_start": target_index.strftime("%H:%M"),
            "predicted_cny_mwh": point_values * float(normalization_scale) + float(normalization_center),
            "negative_probability": probability,
            "p10_cny_mwh": values[:, 0],
            "p50_cny_mwh": values[:, 1],
            "p90_cny_mwh": values[:, 2],
            "model_id": MODEL_ID,
            "model_version": VERSION,
            "as_of": contract.as_of.isoformat(),
            "realtime_source_endpoint": contract.realtime_source_endpoint.isoformat(),
            "realtime_cutoff": contract.realtime_cutoff.isoformat(),
            "day_ahead_cutoff": contract.day_ahead_cutoff.isoformat(),
            "contract_version": CONTRACT_VERSION,
            "normalization_center": float(normalization_center),
            "normalization_scale": float(normalization_scale),
            "parameter_checksum": str(parameter_checksum),
            "point_postprocess": "raw_qilupulse96_v1",
            "risk_postprocess": "not_applied",
        }
    )
    if not (result["p10_cny_mwh"] <= result["p50_cny_mwh"]).all() or not (result["p50_cny_mwh"] <= result["p90_cny_mwh"]).all():
        raise RuntimeError("QiluPulse-96 v1.0 emitted crossed quantiles")
    return result


def _coerce_as_of_timestamp(value: str | pd.Timestamp) -> str | pd.Timestamp:
    """Accept the public ``YYYY-MM-DD HH:MM Asia/Shanghai`` spelling reliably."""
    if isinstance(value, str) and value.rstrip().endswith("Asia/Shanghai"):
        local = value.rsplit("Asia/Shanghai", maxsplit=1)[0].strip()
        return pd.Timestamp(local).tz_localize("Asia/Shanghai")
    return value
