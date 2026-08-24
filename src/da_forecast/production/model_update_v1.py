"""Promotion and audit helpers for manually requested realtime-only updates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle


def daily_block_bootstrap_delta(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Paired day bootstrap for audit reporting; it never gates manual promotion."""
    if draws < 1:
        raise ValueError("draws must be positive")
    required = {"market_date", "predicted_cny_mwh", "actual_cny_mwh"}
    if not required.issubset(baseline) or not required.issubset(candidate):
        raise ValueError("Bootstrap inputs require market_date, predicted_cny_mwh and actual_cny_mwh")
    base_daily = (baseline["predicted_cny_mwh"] - baseline["actual_cny_mwh"]).abs().groupby(baseline["market_date"]).mean()
    candidate_daily = (candidate["predicted_cny_mwh"] - candidate["actual_cny_mwh"]).abs().groupby(candidate["market_date"]).mean()
    dates = base_daily.index.intersection(candidate_daily.index).to_numpy()
    if not len(dates):
        raise ValueError("Bootstrap requires at least one shared settled day")
    rng = np.random.default_rng(seed)
    deltas = np.asarray([
        float((candidate_daily.loc[sample] - base_daily.loc[sample]).mean())
        for sample in (rng.choice(dates, size=len(dates), replace=True) for _ in range(draws))
    ])
    return {
        "days": int(len(dates)),
        "draws": int(draws),
        "candidate_minus_base_mae": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def promote_bundle(
    root: str | Path,
    *,
    candidate: QiluPulse96ProductionBundle,
    candidate_path: str | Path,
    target_date: str,
    source_default: dict[str, Any],
    update_id: str,
    operator_note: str,
    validation: dict[str, Any],
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically switch the default pointer after hard bundle checks."""
    root = Path(root)
    candidate_path = Path(candidate_path).resolve()
    if candidate.root is None or candidate.root.resolve() != candidate_path:
        raise ValueError("Candidate bundle root does not match the promotion path")
    reloaded = QiluPulse96ProductionBundle.load(candidate_path)
    if reloaded.parameter_checksum != candidate.parameter_checksum:
        raise ValueError("Candidate parameter checksum changed during promotion")
    if reloaded.manifest_data.get("feature_schema", {}).get("price_features") != "realtime_only":
        raise ValueError("Only realtime-only bundles can be promoted by this workflow")
    old_path = root / source_default["bundle_path"]
    if not old_path.is_dir():
        raise FileNotFoundError(f"Current default bundle is missing: {old_path}")
    update_dir = root / "runs" / "model_updates" / str(target_date) / str(update_id)
    update_dir.mkdir(parents=True, exist_ok=True)
    old_manifest = {
        "default_pointer": source_default,
        "bundle_path": str(old_path.resolve()),
        "parameter_checksum": QiluPulse96ProductionBundle.load(old_path).parameter_checksum,
    }
    (update_dir / "previous_default.json").write_text(json.dumps(old_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "update_id": update_id,
        "target_date": target_date,
        "promotion_status": "promoted_by_operator",
        "operator_note": operator_note,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(old_path.resolve()),
        "source_parameter_checksum": old_manifest["parameter_checksum"],
        "candidate_bundle": str(candidate_path),
        "candidate_parameter_checksum": reloaded.parameter_checksum,
        "candidate_bundle_sha256": reloaded.bundle_sha256,
        "validation_gate": "warning_only",
        "training": training_metadata,
        "validation": validation,
        "rollback_bundle": str(old_path.resolve()),
    }
    _atomic_write_json(update_dir / "update_manifest.json", payload)
    pointer = {
        "model_id": "QiluPulse-96",
        "version": reloaded.manifest_data.get("model_version", candidate_path.name),
        "bundle_path": str(candidate_path.relative_to(root)).replace("\\", "/"),
        "promoted_at": payload["promoted_at"],
        "update_id": update_id,
        "operator_note": operator_note,
        "previous_bundle_path": str(old_path.relative_to(root)).replace("\\", "/"),
        "previous_parameter_checksum": old_manifest["parameter_checksum"],
        "parameter_checksum": reloaded.parameter_checksum,
    }
    _atomic_write_json(root / "artifacts" / "prediction-layer" / "default.json", pointer)
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
