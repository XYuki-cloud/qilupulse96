from __future__ import annotations

import numpy as np

from da_forecast.models.adaptive_normalization import RevIN, RobustRecentNormalizer, recent_state_features


def test_revin_round_trip_uses_only_history_statistics() -> None:
    history = np.asarray([-40.0, 10.0, 120.0, 360.0])
    target = np.asarray([-20.0, 80.0, 420.0])
    normalizer = RevIN()
    stats = normalizer.statistics(history)

    normalized = normalizer.normalize(target, stats)
    restored = normalizer.denormalize(normalized, stats)

    assert np.allclose(restored, target)
    future_changed = normalizer.statistics(np.concatenate([history, [99999.0]]))
    assert stats != future_changed


def test_revin_constant_history_has_finite_safe_scale() -> None:
    normalizer = RevIN()
    stats = normalizer.statistics(np.full(8, 100.0))

    assert stats.scale >= normalizer.eps
    assert np.isfinite(normalizer.normalize(np.asarray([100.0]), stats)).all()


def test_robust_normalizer_is_not_pulled_by_a_single_spike() -> None:
    history = np.asarray([100.0, 110.0, 105.0, 95.0, 102.0, 10000.0])
    stats = RobustRecentNormalizer().statistics(history)

    assert 95.0 < stats.center < 115.0
    assert stats.scale < 100.0


def test_recent_state_features_are_causal_and_complete() -> None:
    history = np.asarray([-20.0, 0.0, 100.0, 200.0])
    state = recent_state_features(history)

    assert state.shape == (5,)
    assert state[2] == 0.25
    assert np.isfinite(state).all()
