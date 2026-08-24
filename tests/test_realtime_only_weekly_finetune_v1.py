from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Spec
from da_forecast.production.bundle_v1 import QiluPulse96ProductionBundle
from da_forecast.production.preprocessing_v1 import PreprocessingStateV1
from da_forecast.production.weekly_finetune_v1 import select_realtime_only_finetune_positions


def _market_index(days: int = 500) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=days * 96, freq="15min", tz="Asia/Shanghai")


def test_finetune_uses_365_completed_days_with_last_14_held_out() -> None:
    index = _market_index()
    price = pd.Series(np.arange(len(index), dtype=float), index=index)

    fit, validation = select_realtime_only_finetune_positions(
        price,
        label_cutoff="2026-05-15",
        context_days=90,
    )

    assert len(fit) == 351
    assert len(validation) == 14
    assert index[validation[-1]].normalize() == pd.Timestamp("2026-05-15", tz="Asia/Shanghai")
    assert index[fit[-1]].normalize() < index[validation[0]].normalize()


def test_finetune_rejects_a_partial_label_day_in_the_eligible_window() -> None:
    index = _market_index()
    price = pd.Series(np.arange(len(index), dtype=float), index=index)
    missing = pd.Timestamp("2026-05-14 23:45", tz="Asia/Shanghai")
    price = price.drop(missing)

    with pytest.raises(ValueError, match="Incomplete realtime label day"):
        select_realtime_only_finetune_positions(
            price,
            label_cutoff="2026-05-15",
            context_days=90,
        )


def test_realtime_only_dataset_is_constructed_without_a_scripts_import() -> None:
    from da_forecast.production.weekly_finetune_v1 import (
        RealtimeOnlyFineTuneInputs,
        _metadata_for_bundle,
        build_realtime_only_dataset,
    )

    index = _market_index(100)
    price = pd.Series(np.linspace(-20.0, 400.0, len(index)), index=index)
    spec = QiluPulse96V1Spec(
        station_variable_dim=25,
        history_extra_dim=14,
        target_extra_dim=19,
        n_stations=16,
    )
    bundle = QiluPulse96ProductionBundle(
        spec=spec,
        model=spec.build_model(),
        preprocessing=PreprocessingStateV1.identity(history_extra_dim=14, target_extra_dim=14),
        manifest_data={"feature_schema": {"price_features": "realtime_only"}},
    )
    inputs = RealtimeOnlyFineTuneInputs(
        price=price,
        history_extra=np.zeros((len(index), 14), dtype=np.float32),
        station_weather=np.zeros((len(index), 16, 25), dtype=np.float32),
        target_extra=np.zeros((len(index), 14), dtype=np.float32),
        index=index,
    )
    position = np.asarray([90 * 96 + 96], dtype=int)
    metadata = _metadata_for_bundle(bundle, inputs, position)

    sample = build_realtime_only_dataset(inputs, metadata, position)[0]

    assert sample["history_extra"].shape == (90 * 96, 14)
    assert sample["target_extra"].shape == (96, 19)


def test_weather_history_overlay_takes_precedence_for_the_same_slot() -> None:
    from da_forecast.production.weekly_finetune_v1 import merge_observed_weather_history

    index = pd.date_range("2026-08-19", periods=2, freq="15min", tz="Asia/Shanghai")
    overlay_index = pd.date_range("2026-08-19 00:15", periods=2, freq="15min", tz="Asia/Shanghai")
    base = {"SD_JINAN": pd.DataFrame({"temperature_2m": [20.0, 21.0]}, index=index)}
    overlay = {"SD_JINAN": pd.DataFrame({"temperature_2m": [99.0, 22.0]}, index=overlay_index)}

    merged = merge_observed_weather_history(base, overlay)

    assert merged["SD_JINAN"].loc[index[1], "temperature_2m"] == 99.0
    assert merged["SD_JINAN"].loc[overlay_index[1], "temperature_2m"] == 22.0
