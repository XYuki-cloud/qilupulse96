from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from da_forecast.models import QiluPulse96V1Artifact as PublicArtifact
from da_forecast.models.qilupulse96_v1 import (
    QiluPulse96V1Artifact,
    QiluPulse96V1Spec,
    normalized_output_frame,
)


def _mainline_spec() -> QiluPulse96V1Spec:
    return QiluPulse96V1Spec(
        station_variable_dim=25,
        history_extra_dim=18,
        target_extra_dim=19,
        n_stations=16,
    )


def test_v1_kernel_is_exposed_from_the_models_package() -> None:
    assert PublicArtifact is QiluPulse96V1Artifact


def test_v1_artifact_round_trip_preserves_declared_kernel_contract(tmp_path) -> None:
    spec = _mainline_spec()
    artifact = QiluPulse96V1Artifact.create(spec, training_metadata={"weather_kind": "observed_proxy"})
    path = tmp_path / "qilupulse96_v1.pt"

    artifact.save(path)
    restored = QiluPulse96V1Artifact.load(path)

    assert restored.spec == spec
    assert restored.model_id == "QiluPulse-96"
    assert restored.version == "1.0.0"
    assert restored.parameter_checksum == artifact.parameter_checksum


def test_v1_artifact_manifest_exposes_auditable_contract_without_model_execution() -> None:
    spec = _mainline_spec()
    artifact = QiluPulse96V1Artifact.create(
        spec,
        training_metadata={"weather_kind": "observed_proxy", "train_end": "2025-12-30"},
    )

    manifest = artifact.manifest()

    assert manifest["model_id"] == "QiluPulse-96"
    assert manifest["model_version"] == "1.0.0"
    assert manifest["contract_version"] == "realtime_tminus1_1100_endpoint_v1"
    assert manifest["output_slots"] == 96
    assert manifest["spec"] == {
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
    assert manifest["training_metadata"]["train_end"] == "2025-12-30"
    assert manifest["parameter_checksum"] == artifact.parameter_checksum


def test_v1_artifact_inspection_cli_emits_the_manifest(tmp_path) -> None:
    spec = _mainline_spec()
    artifact_path = tmp_path / "qilupulse96_v1.pt"
    QiluPulse96V1Artifact.create(spec).save(artifact_path)

    completed = subprocess.run(
        [sys.executable, "scripts/inspect_qilupulse96_v1_artifact.py", "--artifact", str(artifact_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(completed.stdout)
    assert manifest["model_id"] == "QiluPulse-96"
    assert manifest["output_slots"] == 96


def test_v1_rejects_a_different_model_capacity_under_the_same_version() -> None:
    with pytest.raises(ValueError, match="fixed mainline topology"):
        QiluPulse96V1Spec(
            station_variable_dim=25,
            history_extra_dim=18,
            target_extra_dim=19,
            n_stations=16,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
        )


def test_v1_normalized_output_frame_enforces_96_slot_tminus1_contract() -> None:
    slots = np.arange(96, dtype=float)
    frame = normalized_output_frame(
        point=np.zeros(96),
        negative_probability=np.linspace(0.0, 1.0, 96),
        quantiles=np.column_stack([np.ones(96), np.zeros(96), -np.ones(96)]),
        normalization_center=300.0,
        normalization_scale=20.0,
        target_date="2026-08-20",
        as_of="2026-08-19 12:00 Asia/Shanghai",
        parameter_checksum="unit-test-checksum",
    )

    assert len(frame) == 96
    assert frame["period_start"].iloc[0] == "00:00"
    assert frame["period_start"].iloc[-1] == "23:45"
    assert frame["contract_version"].eq("realtime_tminus1_1100_endpoint_v1").all()
    assert frame["realtime_cutoff"].eq("2026-08-19T10:45:00+08:00").all()
    assert frame["day_ahead_cutoff"].eq("2026-08-18T23:45:00+08:00").all()
    assert frame["negative_probability"].between(0.0, 1.0).all()
    assert (frame["p10_cny_mwh"] <= frame["p50_cny_mwh"]).all()
    assert (frame["p50_cny_mwh"] <= frame["p90_cny_mwh"]).all()
    assert np.isclose(frame["predicted_cny_mwh"].iloc[0], 300.0)
