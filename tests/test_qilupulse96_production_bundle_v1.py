from pathlib import Path

import numpy as np
import torch

from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Spec
from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1


def test_production_bundle_round_trip_preserves_weights_and_hash(tmp_path):
    spec = QiluPulse96V1Spec(station_variable_dim=25, history_extra_dim=18, target_extra_dim=19, n_stations=16)
    model = spec.build_model()
    bundle = QiluPulse96ProductionBundle(spec, model, PreprocessingStateV1.identity(), {"feature_schema": {}, "station_order": []})
    root = bundle.save(tmp_path / "bundle")
    restored = QiluPulse96ProductionBundle.load(root)
    assert restored.parameter_checksum == bundle.parameter_checksum
    assert restored.bundle_sha256 == bundle.bundle_sha256
    assert np.array_equal(restored.preprocessing.state_features.mean, bundle.preprocessing.state_features.mean)


def test_bundle_rejects_tampered_model_state(tmp_path):
    spec = QiluPulse96V1Spec(station_variable_dim=25, history_extra_dim=18, target_extra_dim=19, n_stations=16)
    bundle = QiluPulse96ProductionBundle(spec, spec.build_model(), PreprocessingStateV1.identity(), {"feature_schema": {}, "station_order": []})
    root = bundle.save(tmp_path / "bundle")
    state = torch.load(root / "model_state.pt", weights_only=True)
    first = next(iter(state))
    state[first] = state[first] + 1.0
    torch.save(state, root / "model_state.pt")
    try:
        QiluPulse96ProductionBundle.load(root)
    except ValueError as exc:
        assert "checksum" in str(exc).lower()
    else:
        raise AssertionError("tampered bundle was accepted")
