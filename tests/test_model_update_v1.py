from __future__ import annotations

import json

import pandas as pd

from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Spec
from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.model_update_v1 import daily_block_bootstrap_delta, promote_bundle
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1


def _bundle(root):
    spec = QiluPulse96V1Spec(station_variable_dim=25, history_extra_dim=14, target_extra_dim=19, n_stations=16)
    bundle = QiluPulse96ProductionBundle(
        spec=spec,
        model=spec.build_model(),
        preprocessing=PreprocessingStateV1.identity(history_extra_dim=14, target_extra_dim=14),
        manifest_data={"model_version": "realtime-only", "feature_schema": {"price_features": "realtime_only"}},
    )
    bundle.save(root)
    return bundle


def test_promote_bundle_keeps_previous_bundle_and_switches_only_default_pointer(tmp_path) -> None:
    old_path = tmp_path / "artifacts" / "prediction-layer" / "bundles" / "old"
    candidate_path = tmp_path / "artifacts" / "prediction-layer" / "bundles" / "candidate"
    old = _bundle(old_path)
    candidate = _bundle(candidate_path)
    default_path = tmp_path / "artifacts" / "prediction-layer" / "default.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(json.dumps({"bundle_path": str(old_path.relative_to(tmp_path))}), encoding="utf-8")

    promoted = promote_bundle(
        tmp_path,
        candidate=candidate,
        candidate_path=candidate_path,
        target_date="2026-08-23",
        source_default=json.loads(default_path.read_text(encoding="utf-8")),
        update_id="update-test",
        operator_note="人工要求日更微调",
        validation={"deployment_gate": "warning_only"},
        training_metadata={"method": "weekly_decay180_realtime_only_v1"},
    )

    pointer = json.loads(default_path.read_text(encoding="utf-8"))
    assert old_path.is_dir() and candidate_path.is_dir()
    assert pointer["bundle_path"] == str(candidate_path.relative_to(tmp_path)).replace("\\", "/")
    assert pointer["previous_parameter_checksum"] == old.parameter_checksum
    assert promoted["promotion_status"] == "promoted_by_operator"
    manifest = tmp_path / "runs" / "model_updates" / "2026-08-23" / "update-test" / "update_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["rollback_bundle"] == str(old_path.resolve())


def test_daily_block_bootstrap_reports_candidate_minus_baseline_mae() -> None:
    base = pd.DataFrame({
        "market_date": ["2026-08-01", "2026-08-02"],
        "predicted_cny_mwh": [10.0, 10.0],
        "actual_cny_mwh": [0.0, 0.0],
    })
    candidate = base.copy()
    candidate["predicted_cny_mwh"] = [5.0, 5.0]

    audit = daily_block_bootstrap_delta(base, candidate, draws=20, seed=7)

    assert audit["days"] == 2
    assert audit["candidate_minus_base_mae"] == -5.0
