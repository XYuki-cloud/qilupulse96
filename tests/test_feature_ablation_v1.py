from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from da_forecast.production.feature_ablation_v1 import (
    GROUP_VARIANTS,
    ablate_inputs,
    bootstrap_mean_ci,
    calculate_metrics,
    all_variant_names,
    output_delta_metrics,
    paired_metric_summary,
    summarize_variant_frame,
    variable_specs,
)
from da_forecast.production.input_builder_v1 import CausalInputBundle


def _inputs() -> CausalInputBundle:
    return CausalInputBundle(
        target_date=pd.Timestamp("2026-08-08", tz="Asia/Shanghai"),
        realtime_cutoff=pd.Timestamp("2026-08-07 10:45", tz="Asia/Shanghai"),
        day_ahead_cutoff=pd.Timestamp("2026-08-06 23:45", tz="Asia/Shanghai"),
        history_price=np.zeros((8, 1), dtype=np.float32),
        history_extra=np.arange(8 * 14, dtype=np.float32).reshape(8, 14),
        history_station_weather=np.arange(8 * 16 * 25, dtype=np.float32).reshape(8, 16, 25),
        target_extra=np.arange(96 * 19, dtype=np.float32).reshape(96, 19),
        target_station_weather=np.arange(96 * 16 * 25, dtype=np.float32).reshape(96, 16, 25),
        state_features=np.arange(5, dtype=np.float32),
        normalization_center=300.0,
        normalization_scale=50.0,
    )


def test_group_ablation_preserves_shapes_dtypes_and_intended_channels() -> None:
    original = _inputs()

    for name in GROUP_VARIANTS:
        changed = ablate_inputs(original, name)
        assert changed.history_price.shape == original.history_price.shape
        assert changed.history_extra.shape == original.history_extra.shape
        assert changed.history_station_weather.shape == original.history_station_weather.shape
        assert changed.target_extra.shape == original.target_extra.shape
        assert changed.target_station_weather.shape == original.target_station_weather.shape
        assert changed.state_features.shape == original.state_features.shape
        assert changed.target_extra.dtype == np.float32
        assert changed.target_station_weather.dtype == np.float32

    full = ablate_inputs(original, "full")
    np.testing.assert_array_equal(full.history_extra, original.history_extra)
    np.testing.assert_array_equal(full.target_station_weather, original.target_station_weather)

    weather = ablate_inputs(original, "weather_meteorology_off")
    np.testing.assert_array_equal(weather.history_station_weather[..., :18], 0.0)
    np.testing.assert_array_equal(weather.target_station_weather[..., :18], 0.0)
    np.testing.assert_array_equal(weather.target_station_weather[..., 18:], original.target_station_weather[..., 18:])

    calendar = ablate_inputs(original, "calendar_date_off")
    np.testing.assert_array_equal(calendar.history_extra[:, :2], original.history_extra[:, :2])
    np.testing.assert_array_equal(calendar.target_extra[:, :2], original.target_extra[:, :2])
    np.testing.assert_array_equal(calendar.target_extra[:, 14:], original.target_extra[:, 14:])
    np.testing.assert_array_equal(calendar.history_extra[:, 2:], 0.0)
    np.testing.assert_array_equal(calendar.target_extra[:, 2:14], 0.0)

    state = ablate_inputs(original, "price_state_off")
    np.testing.assert_array_equal(state.state_features, 0.0)
    np.testing.assert_array_equal(state.target_extra[:, 14:], 0.0)


def test_single_variable_ablation_changes_only_requested_weather_channel() -> None:
    original = _inputs()
    changed = ablate_inputs(original, "weather:temperature_2m")

    np.testing.assert_array_equal(changed.history_station_weather[..., 0], 0.0)
    np.testing.assert_array_equal(changed.target_station_weather[..., 0], 0.0)
    np.testing.assert_array_equal(changed.history_station_weather[..., 1:], original.history_station_weather[..., 1:])
    np.testing.assert_array_equal(changed.target_station_weather[..., 1:], original.target_station_weather[..., 1:])
    np.testing.assert_array_equal(changed.history_extra, original.history_extra)
    np.testing.assert_array_equal(changed.target_extra, original.target_extra)


def test_single_variable_specs_cover_weather_calendar_and_state_features() -> None:
    specs = variable_specs()
    assert len(specs) == 25 + 14 + 5
    assert {spec.group for spec in specs} == {"weather", "calendar", "state"}
    assert specs[0].name == "weather:temperature_2m"
    assert specs[25].name == "calendar:slot_sin"
    assert specs[-1].name == "state:std"


def test_all_variant_names_are_unique_and_include_group_and_variable_interventions() -> None:
    names = all_variant_names()

    assert len(names) == len(set(names))
    assert names[: len(GROUP_VARIANTS)] == GROUP_VARIANTS
    assert "weather:temperature_2m" in names
    assert "calendar:is_weekend_effective" in names
    assert "state:recent_price_median" in names


def test_calculate_metrics_reports_point_probability_and_interval_quality() -> None:
    frame = pd.DataFrame(
        {
            "actual_cny_mwh": [-10.0, 20.0, 40.0, 60.0],
            "predicted_cny_mwh": [-5.0, 10.0, 50.0, 50.0],
            "negative_probability": [0.8, 0.2, 0.1, 0.05],
            "p10_cny_mwh": [-20.0, 0.0, 20.0, 40.0],
            "p90_cny_mwh": [10.0, 30.0, 60.0, 70.0],
        }
    )

    metrics = calculate_metrics(frame)

    assert metrics["slot_count"] == 4
    assert metrics["mae_cny_mwh"] == 8.75
    assert metrics["rmse_cny_mwh"] == 9.013878188659973
    assert metrics["brier_score"] == 0.023125
    assert metrics["interval_coverage"] == 1.0
    assert metrics["mean_interval_width_cny_mwh"] == 32.5


def test_bootstrap_mean_ci_is_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    first = bootstrap_mean_ci(values, draws=1000, seed=7)
    second = bootstrap_mean_ci(values, draws=1000, seed=7)

    assert first == second
    assert first["sample_count"] == 4
    assert first["draws"] == 1000
    assert first["mean"] == 3.75
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_summarize_variant_frame_returns_overall_and_daily_metrics() -> None:
    frame = pd.DataFrame(
        {
            "market_date": ["2026-08-07", "2026-08-07", "2026-08-08", "2026-08-08"],
            "period_start": ["00:00", "00:15", "00:00", "00:15"],
            "actual_cny_mwh": [10.0, 20.0, 30.0, 40.0],
            "predicted_cny_mwh": [12.0, 18.0, 33.0, 35.0],
            "negative_probability": [0.1, 0.1, 0.1, 0.1],
            "p10_cny_mwh": [0.0, 10.0, 20.0, 30.0],
            "p90_cny_mwh": [20.0, 30.0, 40.0, 50.0],
        }
    )

    summary = summarize_variant_frame(frame)

    assert summary["overall"]["slot_count"] == 4
    assert summary["overall"]["mae_cny_mwh"] == 3.0
    assert list(summary["daily"]["market_date"]) == ["2026-08-07", "2026-08-08"]
    assert list(summary["daily"]["mae_cny_mwh"]) == [2.0, 4.0]


def test_paired_metric_summary_treats_positive_variant_minus_full_as_worse() -> None:
    full = pd.DataFrame(
        {"market_date": ["2026-08-07", "2026-08-08", "2026-08-09"], "mae_cny_mwh": [10.0, 20.0, 30.0]}
    )
    variant = pd.DataFrame(
        {"market_date": ["2026-08-07", "2026-08-08", "2026-08-09"], "mae_cny_mwh": [12.0, 18.0, 35.0]}
    )

    result = paired_metric_summary(full, variant, metric="mae_cny_mwh", draws=1000, seed=7)

    assert result["mean_variant_minus_full"] == 1.6666666666666667
    assert result["worse_days"] == 2
    assert result["better_days"] == 1
    assert result["sample_count"] == 3


def test_feature_ablation_cli_help_is_available_without_runtime_data() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--bundle-path" in result.stdout
    assert "--runtime-root" in result.stdout
    assert "--allow-backend-numeric-drift" in result.stdout


def test_simple_baselines_use_all_prior_complete_days_for_each_validation_day() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    index = pd.date_range("2026-07-01", "2026-08-08 23:45", freq="15min", tz="Asia/Shanghai")
    values = np.repeat(np.arange(index.normalize().nunique(), dtype=float), 96)
    realtime = pd.Series(values, index=index)
    days = [
        pd.Timestamp("2026-08-07", tz="Asia/Shanghai"),
        pd.Timestamp("2026-08-08", tz="Asia/Shanghai"),
    ]

    frames = module._simple_baseline_frames(realtime, days)
    second_day = frames["baseline:last_day_same_slot"][1]

    assert second_day["predicted_cny_mwh"].iloc[0] == 37.0


def test_report_daily_metrics_does_not_duplicate_full_baseline(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_report_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    full = pd.DataFrame(
        {
            "variant": "full",
            "market_date": ["2026-08-07", "2026-08-07", "2026-08-08", "2026-08-08"],
            "period_start": ["00:00", "00:15", "00:00", "00:15"],
            "actual_cny_mwh": [10.0, 20.0, 30.0, 40.0],
            "predicted_cny_mwh": [12.0, 18.0, 33.0, 35.0],
            "negative_probability": [0.1] * 4,
            "p10_cny_mwh": [0.0, 10.0, 20.0, 30.0],
            "p90_cny_mwh": [20.0, 30.0, 40.0, 50.0],
        }
    )
    full_day1 = full.iloc[:2].copy()
    full_day2 = full.iloc[2:].copy()
    variant_day1 = full_day1.copy()
    variant_day1["variant"] = "weather_meteorology_off"
    variant_day1["predicted_cny_mwh"] += 1.0
    variant_day2 = full_day2.copy()
    variant_day2["variant"] = "weather_meteorology_off"
    variant_day2["predicted_cny_mwh"] += 1.0

    module._build_report(
        output_dir=tmp_path,
        manifest={
            "start_date": "2026-08-07",
            "end_date": "2026-08-08",
            "device": "cpu",
            "bundle_parameter_checksum": "parameter",
            "bundle_sha256": "bundle",
            "weather_kind": "observed_proxy",
        },
        variant_frames={
            "full": [full_day1, full_day2],
            "weather_meteorology_off": [variant_day1, variant_day2],
        },
        baseline_frames={},
    )

    daily = pd.read_csv(tmp_path / "daily_metrics.csv")
    assert len(daily.loc[daily["variant"] == "full"]) == 2


def test_output_delta_metrics_rejects_required_columns_missing_from_either_frame() -> None:
    full = pd.DataFrame(
        {
            "predicted_cny_mwh": [1.0],
            "negative_probability": [0.1],
            "p10_cny_mwh": [0.0],
        }
    )
    variant = full.assign(p90_cny_mwh=2.0)

    with pytest.raises(ValueError, match="p90_cny_mwh"):
        output_delta_metrics(full, variant)


def test_weather_history_merge_normalizes_mixed_timezones_to_market_timezone() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_weather_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cached = pd.DataFrame(
        {"temperature_2m": [1.0, 2.0]},
        index=pd.date_range("2026-05-21 16:00", periods=2, freq="h", tz="UTC"),
    )
    history = pd.DataFrame(
        {"temperature_2m": [3.0, 4.0]},
        index=pd.date_range("2026-05-22 00:00", periods=2, freq="15min", tz="Asia/Shanghai"),
    )

    merged = module._merge_weather_history({"SD_TEST": cached}, {"SD_TEST": history})["SD_TEST"]

    assert isinstance(merged.index, pd.DatetimeIndex)
    assert str(merged.index.tz) == "Asia/Shanghai"


def test_observed_panel_prefers_complete_existing_quarter_cache(tmp_path, monkeypatch) -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_cache_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from da_forecast.config import SHANDONG_SPATIAL_STATIONS
    from da_forecast.production.feature_schema_v1 import STATION_COLUMNS
    import da_forecast.sources.spatial_weather_v01 as spatial_weather

    index = pd.date_range("2026-05-22", periods=4, freq="15min", tz="Asia/Shanghai")
    frame = pd.DataFrame(
        {column: np.ones(len(index), dtype=float) for column in STATION_COLUMNS},
        index=index,
    )
    cache_root = tmp_path / "data" / "raw" / "openmeteo_spatial_v01_quarter"
    for station in SHANDONG_SPATIAL_STATIONS:
        destination = cache_root / station.code / "weather.parquet"
        destination.parent.mkdir(parents=True)
        frame.to_parquet(destination)

    def fail_if_rebuilt(**_kwargs):
        raise AssertionError("the offline audit must not rebuild an existing complete quarter cache")

    monkeypatch.setattr(spatial_weather, "load_or_build_observed_spatial_quarters", fail_if_rebuilt)

    panel = module._load_observed_panel(tmp_path)

    assert set(panel) == {station.code for station in SHANDONG_SPATIAL_STATIONS}
    assert panel["SD_JINAN"].index.equals(index)
    np.testing.assert_array_equal(panel["SD_JINAN"][list(STATION_COLUMNS)].to_numpy(), frame.to_numpy())


def test_device_auto_resolves_to_cpu_without_cuda() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_device_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._resolve_device("cpu") == "cpu"
    assert module._resolve_device("auto", cuda_available=False) == "cpu"
    assert module._resolve_device("auto", cuda_available=True) == "cuda"


def test_variable_rows_receive_separate_sensitivity_and_error_contribution_ranks() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_rank_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [
        {
            "variant": "weather:a",
            "output_mean_abs_output_delta_cny_mwh": 2.0,
            "paired_mae_variant_minus_full": 0.1,
        },
        {
            "variant": "weather:b",
            "output_mean_abs_output_delta_cny_mwh": 1.0,
            "paired_mae_variant_minus_full": 0.3,
        },
    ]

    ranked = module._rank_variable_rows(rows)

    assert ranked[0]["output_sensitivity_rank"] == 1
    assert ranked[1]["output_sensitivity_rank"] == 2
    assert ranked[0]["predictive_contribution_rank"] == 2
    assert ranked[1]["predictive_contribution_rank"] == 1


def test_full_parity_numeric_drift_requires_explicit_override() -> None:
    script = Path(__file__).parents[1] / "scripts" / "audit_qilupulse96_feature_ablation.py"
    spec = importlib.util.spec_from_file_location("feature_ablation_parity_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    periods = [f"{index:02d}" for index in range(96)]
    full = pd.DataFrame(
        {
            "period_start": periods,
            "predicted_cny_mwh": [100.001] + [100.0] * 95,
        }
    )
    ledger = pd.DataFrame(
        {
            "market_date": ["2026-08-07"] * 96,
            "period_start": periods,
            "predicted_cny_mwh": [100.0] * 96,
        }
    )

    with pytest.raises(RuntimeError, match="Full replay parity failed"):
        module._check_full_parity(full, ledger, "2026-08-07")

    assert module._check_full_parity(full, ledger, "2026-08-07", allow_numeric_drift=True) == pytest.approx(0.001)
