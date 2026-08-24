"""Causal 180-day decay fine-tuning for realtime-only production bundles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from da_forecast.config import PRICE_COL, TIMEZONE
from da_forecast.features.calendar_v01 import build_calendar_v01
from da_forecast.models.adaptive_normalization import RobustRecentNormalizer, recent_state_features
from da_forecast.models.longseq_v01 import EarlyStopping, multitask_loss
from da_forecast.sources.spatial_weather_v01 import load_or_build_observed_spatial_quarters
from da_forecast.forecasting.rolling_adaptation_v01 import select_weekly_training_positions

from .bundle_v1 import QiluPulse96ProductionBundle
from .data_resolver_v1 import DataResolverV1


CONTEXT_DAYS = 90
MAX_TRAINING_DAYS = 365
VALIDATION_DAYS = 14
HALF_LIFE_DAYS = 180.0
REALTIME_HISTORY_END_OFFSET_SLOTS = 53


@dataclass(frozen=True)
class RealtimeOnlyFineTuneInputs:
    price: pd.Series
    history_extra: np.ndarray
    station_weather: np.ndarray
    target_extra: np.ndarray
    index: pd.DatetimeIndex


@dataclass(frozen=True)
class FineTuneOutcome:
    model: torch.nn.Module
    metadata: dict[str, Any]
    fit_positions: np.ndarray
    validation_positions: np.ndarray


class _RealtimeOnlySequenceDataset(Dataset):
    """Training samples for the realtime-only contract; no day-ahead fields exist here."""

    def __init__(
        self,
        inputs: RealtimeOnlyFineTuneInputs,
        metadata: dict[str, Any],
        positions: np.ndarray,
    ) -> None:
        self.inputs = inputs
        self.metadata = metadata
        self.positions = np.asarray(positions, dtype=int)
        self.context_slots = CONTEXT_DAYS * 96

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, object]:
        position = int(self.positions[item])
        history_end = position - REALTIME_HISTORY_END_OFFSET_SLOTS
        history_start = history_end - self.context_slots + 1
        history_slice = slice(history_start, history_end + 1)
        target_slice = slice(position, position + 96)
        price = self.inputs.price.to_numpy(dtype=np.float32)
        normalizer = self.metadata["normalizer"]
        stats = normalizer.statistics(price[history_slice])
        state = self.metadata["state_scaler"].transform(
            self.metadata["state_features"][position][None, :]
        )[0].astype(np.float32)
        return {
            "target_position": torch.tensor(position, dtype=torch.int64),
            "history_price": torch.from_numpy(normalizer.normalize(price[history_slice], stats).astype(np.float32)[:, None]),
            "history_extra": torch.from_numpy(self.metadata["history_scaled"][history_slice].astype(np.float32)),
            "history_station_weather": torch.from_numpy(self.metadata["station_scaled"][history_slice].astype(np.float32)),
            "target_extra": torch.from_numpy(
                np.column_stack([
                    self.metadata["target_scaled"][target_slice],
                    np.repeat(state[None, :], 96, axis=0),
                ]).astype(np.float32)
            ),
            "target_station_weather": torch.from_numpy(self.metadata["station_scaled"][target_slice].astype(np.float32)),
            "label": torch.from_numpy(normalizer.normalize(price[target_slice], stats).astype(np.float32)),
            "negative": torch.from_numpy((price[target_slice] < 0).astype(np.float32)),
            "normalization_center": float(stats.center),
            "normalization_scale": float(stats.scale),
            "state_features": torch.from_numpy(state),
        }


def build_realtime_only_dataset(
    inputs: RealtimeOnlyFineTuneInputs,
    metadata: dict[str, Any],
    positions: np.ndarray,
) -> Dataset:
    """Build an in-package dataset so fine-tuning does not depend on ``scripts/`` imports."""
    return _RealtimeOnlySequenceDataset(inputs, metadata, positions)


def _move_model_inputs(sample: dict[str, object], device: torch.device, *, batch: bool) -> dict[str, torch.Tensor]:
    keys = (
        "history_price", "history_extra", "history_station_weather", "target_extra",
        "target_station_weather", "state_features",
    )
    values: dict[str, torch.Tensor] = {}
    for key in keys:
        value = sample[key]
        assert isinstance(value, torch.Tensor)
        values[key] = value.to(device) if batch else value.unsqueeze(0).to(device)
    return values


def _local_day(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(TIMEZONE) if stamp.tz is None else stamp.tz_convert(TIMEZONE)
    return stamp.normalize()


def merge_observed_weather_history(
    base_panel: dict[str, pd.DataFrame],
    history_overlay: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Apply the same observed-history precedence used by calibration replay."""
    merged: dict[str, pd.DataFrame] = {}
    for code, base in base_panel.items():
        extra = history_overlay.get(code)
        if extra is None or extra.empty:
            merged[code] = base
            continue
        frame = pd.concat([base, extra]).sort_index()
        merged[code] = frame[~frame.index.duplicated(keep="last")]
    return merged


def _complete_day_position(index: pd.DatetimeIndex, day: pd.Timestamp) -> int:
    expected = pd.date_range(day, periods=96, freq="15min", tz=TIMEZONE)
    available = index[(index >= day) & (index < day + pd.Timedelta("1D"))]
    if not available.equals(expected):
        missing = expected.difference(available)
        raise ValueError(
            f"Incomplete realtime label day {day.date()}: expected 96 settlement slots, "
            f"missing {len(missing)} slot(s)"
        )
    return int(index.get_loc(day))


def select_realtime_only_finetune_positions(
    realtime: pd.Series,
    *,
    label_cutoff: str | pd.Timestamp,
    context_days: int = CONTEXT_DAYS,
    max_training_days: int = MAX_TRAINING_DAYS,
    validation_days: int = VALIDATION_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a complete causal 365-day window for 180-day decay fine-tuning."""
    if realtime.empty:
        raise ValueError("Realtime labels are empty")
    if realtime.index.has_duplicates:
        raise ValueError("Realtime labels contain duplicate timestamps")
    index = pd.DatetimeIndex(realtime.index)
    index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    index = index.sort_values()
    cutoff = _local_day(label_cutoff)
    earliest_target = index.min().normalize() + pd.Timedelta(f"{int(context_days) + 1}D")
    if cutoff < earliest_target:
        raise ValueError(f"label_cutoff {cutoff.date()} does not have {context_days} days of price history")
    dates = pd.date_range(earliest_target, cutoff, freq="D", tz=TIMEZONE)
    if len(dates) < max_training_days:
        raise ValueError(
            f"Need {max_training_days} complete target days through {cutoff.date()}, found {len(dates)}"
        )
    selected_dates = dates[-max_training_days:]
    positions = np.asarray([_complete_day_position(index, day) for day in selected_dates], dtype=int)
    return select_weekly_training_positions(
        positions,
        index,
        label_cutoff=cutoff,
        max_training_days=max_training_days,
        validation_days=validation_days,
    )


def load_realtime_only_finetune_inputs(
    root: str | Path,
    *,
    label_cutoff: str | pd.Timestamp,
    manual_workbook: str | Path | None = None,
) -> RealtimeOnlyFineTuneInputs:
    """Load effective realtime labels and observed weather without day-ahead data."""
    root = Path(root)
    cutoff_end = _local_day(label_cutoff) + pd.Timedelta(hours=23, minutes=45)
    realtime = DataResolverV1(root, manual_workbook=manual_workbook).load_price("realtime").sort_index()
    realtime = realtime.loc[:cutoff_end]
    if realtime.empty:
        raise ValueError(f"No realtime labels available through {cutoff_end.isoformat()}")
    index = pd.DatetimeIndex(realtime.index)
    index = index.tz_localize(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    realtime.index = index
    calendar = build_calendar_v01(index, reference_dir=root / "data" / "reference" / "calendar")
    panel = load_or_build_observed_spatial_quarters(cache_dir=root / "data" / "raw")
    history_root = root / "data" / "raw" / "weather_history_v1"
    if history_root.is_dir():
        history_overlay = {
            code: pd.read_parquet(history_root / code / "weather.parquet")
            for code in panel
            if (history_root / code / "weather.parquet").is_file()
        }
        panel = merge_observed_weather_history(panel, history_overlay)
    columns = tuple(
        "temperature_2m relative_humidity_2m apparent_temperature precipitation rain cloud_cover "
        "cloud_cover_low cloud_cover_mid cloud_cover_high shortwave_radiation direct_radiation "
        "diffuse_radiation direct_normal_irradiance wind_speed_10m wind_direction_10m wind_speed_100m "
        "wind_direction_100m wind_gusts_10m solar_elevation solar_azimuth_sin solar_azimuth_cos is_daylight "
        "clear_sky_ghi shortwave_clear_sky_index shortwave_radiation_ramp_15m".split()
    )
    arrays: list[np.ndarray] = []
    for code in sorted(panel):
        frame = panel[code].reindex(index)
        if any(column not in frame for column in columns) or frame[list(columns)].isna().any().any():
            raise ValueError(f"Incomplete observed weather for fine-tuning station {code}")
        arrays.append(frame[list(columns)].to_numpy(dtype=np.float32))
    return RealtimeOnlyFineTuneInputs(
        price=realtime,
        history_extra=calendar.to_numpy(dtype=np.float32),
        station_weather=np.stack(arrays, axis=1),
        target_extra=calendar.to_numpy(dtype=np.float32),
        index=index,
    )


def _metadata_for_bundle(
    bundle: QiluPulse96ProductionBundle,
    inputs: RealtimeOnlyFineTuneInputs,
    positions: np.ndarray,
) -> dict[str, Any]:
    if bundle.spec.history_extra_dim != inputs.history_extra.shape[1]:
        raise ValueError("Fine-tuning requires a realtime-only bundle with 14 calendar history features")
    price_raw = inputs.price.to_numpy(dtype=np.float32)
    context_slots = CONTEXT_DAYS * 96
    state_features = {
        int(position): recent_state_features(
            price_raw[
                int(position) - REALTIME_HISTORY_END_OFFSET_SLOTS - context_slots + 1:
                int(position) - REALTIME_HISTORY_END_OFFSET_SLOTS + 1
            ]
        )
        for position in np.asarray(positions, dtype=int)
    }
    return {
        "price_scaler": bundle.preprocessing.price,
        "price_global": None,
        "history_scaled": bundle.preprocessing.history_extra.transform(inputs.history_extra),
        "target_scaled": bundle.preprocessing.target_extra.transform(inputs.target_extra),
        "station_scaled": bundle.preprocessing.station_weather.transform(inputs.station_weather),
        "normalizer": RobustRecentNormalizer(
            eps=float(bundle.preprocessing.robust_normalizer.get("eps", 1e-4))
        ),
        "state_features": state_features,
        "state_scaler": bundle.preprocessing.state_features,
    }


def fine_tune_realtime_only_bundle(
    bundle: QiluPulse96ProductionBundle,
    inputs: RealtimeOnlyFineTuneInputs,
    *,
    label_cutoff: str | pd.Timestamp,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 16,
    seed: int = 7,
    device: str | torch.device | None = None,
    progress: Callable[[str], None] | None = None,
) -> FineTuneOutcome:
    """Warm-start one 180-day-decay update without changing the source bundle."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience and batch_size must be positive")
    if bundle.manifest_data.get("feature_schema", {}).get("price_features") != "realtime_only":
        raise ValueError("Fine-tuning refuses bundles that use day-ahead prices")
    fit_positions, validation_positions = select_realtime_only_finetune_positions(
        inputs.price,
        label_cutoff=label_cutoff,
    )
    selected = np.concatenate([fit_positions, validation_positions])
    metadata = _metadata_for_bundle(bundle, inputs, selected)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = deepcopy(bundle.model).to(resolved_device)
    context_slots = CONTEXT_DAYS * 96
    train_set = build_realtime_only_dataset(inputs, metadata, fit_positions)
    validation_set = build_realtime_only_dataset(inputs, metadata, validation_positions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=resolved_device.type == "cuda")
    stopper = EarlyStopping(patience=patience)
    best_state: dict[str, torch.Tensor] | None = None
    trace: list[dict[str, float | int | bool]] = []
    latest_position = int(fit_positions[-1])
    started = time.perf_counter()
    if progress is not None:
        progress(
            f"fine_tune_start device={resolved_device.type} train_days={len(fit_positions)} "
            f"validation_days={len(validation_positions)} batch_size={batch_size}"
        )
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=resolved_device.type == "cuda")
        for batch_number, batch in enumerate(train_loader, start=1):
            tensors = {key: value.to(resolved_device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            ages = (latest_position - tensors["target_position"].float()) / 96.0
            weights = torch.exp(-np.log(2.0) * ages / HALF_LIFE_DAYS)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=resolved_device.type == "cuda"):
                output = model(**_move_model_inputs(tensors, resolved_device, batch=True))
                loss = multitask_loss(output, tensors["label"], negative_labels=tensors["negative"], sample_weights=weights)
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            if progress is not None and (batch_number == 1 or batch_number % 5 == 0 or batch_number == len(train_loader)):
                progress(f"fine_tune_train epoch={epoch}/{epochs} batch={batch_number}/{len(train_loader)}")
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in DataLoader(validation_set, batch_size=batch_size, pin_memory=resolved_device.type == "cuda"):
                tensors = {key: value.to(resolved_device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
                with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=resolved_device.type == "cuda"):
                    output = model(**_move_model_inputs(tensors, resolved_device, batch=True))
                    validation_losses.append(float(multitask_loss(output, tensors["label"], negative_labels=tensors["negative"])))
        validation_loss = float(np.mean(validation_losses))
        should_stop = stopper.update(epoch=epoch, validation_loss=validation_loss)
        trace.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "is_best": stopper.last_improved,
        })
        if stopper.last_improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress is not None:
            progress(
                f"fine_tune_epoch epoch={epoch}/{epochs} train_loss={float(np.mean(train_losses)):.6f} "
                f"validation_loss={validation_loss:.6f} best={stopper.last_improved}"
            )
        if should_stop:
            break
    if best_state is None:
        raise RuntimeError("Fine-tuning did not record a validation checkpoint")
    model.load_state_dict(best_state)
    outcome_metadata = {
        "method": "weekly_decay180_realtime_only_v1",
        "label_cutoff": _local_day(label_cutoff).date().isoformat(),
        "max_training_days": MAX_TRAINING_DAYS,
        "validation_days": VALIDATION_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "fit_start": inputs.index[int(fit_positions[0])].date().isoformat(),
        "fit_end": inputs.index[int(fit_positions[-1])].date().isoformat(),
        "validation_start": inputs.index[int(validation_positions[0])].date().isoformat(),
        "validation_end": inputs.index[int(validation_positions[-1])].date().isoformat(),
        "epochs_requested": epochs,
        "epochs_completed": epoch,
        "best_epoch": stopper.best_epoch,
        "best_validation_loss": stopper.best_validation_loss,
        "training_seconds": time.perf_counter() - started,
        "device": resolved_device.type,
        "trace": trace,
        "source_bundle_parameter_checksum": bundle.parameter_checksum,
        "candidate_parameter_checksum": sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
        ).hexdigest(),
    }
    return FineTuneOutcome(model=model.cpu(), metadata=outcome_metadata, fit_positions=fit_positions, validation_positions=validation_positions)


def evaluate_realtime_only_bundle(
    bundle: QiluPulse96ProductionBundle,
    inputs: RealtimeOnlyFineTuneInputs,
    positions: np.ndarray,
    *,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Replay held-out dates with observed-proxy weather for a like-for-like candidate check."""
    metadata = _metadata_for_bundle(bundle, inputs, positions)
    dataset = build_realtime_only_dataset(inputs, metadata, positions)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = bundle.model.to(resolved_device)
    model.eval()
    rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for sample in dataset:
            output = model(**_move_model_inputs(sample, resolved_device, batch=False))
            center = float(sample["normalization_center"])
            scale = float(sample["normalization_scale"])
            quantiles = output["quantiles"].detach().cpu().numpy()[0] * scale + center
            quantiles.sort(axis=1)
            position = int(sample["target_position"])
            target_index = inputs.index[position:position + 96]
            actual = inputs.price.iloc[position:position + 96].to_numpy(dtype=float)
            rows.append(pd.DataFrame({
                "market_date": target_index.strftime("%Y-%m-%d"),
                "period_start": target_index.strftime("%H:%M"),
                "actual_cny_mwh": actual,
                "predicted_cny_mwh": output["point"].detach().cpu().numpy()[0] * scale + center,
                "negative_probability": output["negative_probability"].detach().cpu().numpy()[0],
                "p10_cny_mwh": quantiles[:, 0],
                "p50_cny_mwh": quantiles[:, 1],
                "p90_cny_mwh": quantiles[:, 2],
                "parameter_checksum": bundle.parameter_checksum,
            }))
    return pd.concat(rows, ignore_index=True)


def forecast_metrics(detail: pd.DataFrame) -> dict[str, float]:
    """Metrics used only for the fixed 14-day candidate deployment gate."""
    error = detail["predicted_cny_mwh"] - detail["actual_cny_mwh"]
    negative = detail["actual_cny_mwh"] < 0
    predicted_negative = detail["negative_probability"] >= 0.5
    true_positive = int((negative & predicted_negative).sum())
    false_positive = int((~negative & predicted_negative).sum())
    false_negative = int((negative & ~predicted_negative).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "samples": float(len(detail)),
        "days": float(detail["market_date"].nunique()),
        "mae_cny_mwh": float(error.abs().mean()),
        "rmse_cny_mwh": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
        "bias_cny_mwh": float(error.mean()),
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "p10_p90_coverage": float(detail["actual_cny_mwh"].between(detail["p10_cny_mwh"], detail["p90_cny_mwh"]).mean()),
    }


def save_finetuned_bundle(
    source: QiluPulse96ProductionBundle,
    outcome: FineTuneOutcome,
    destination: str | Path,
) -> QiluPulse96ProductionBundle:
    """Persist a candidate bundle with unchanged realtime-only preprocessing."""
    manifest = deepcopy(source.manifest_data)
    training = dict(manifest.get("training_metadata", {}))
    training["fine_tune"] = outcome.metadata
    manifest["training_metadata"] = training
    manifest["model_version"] = f"{manifest.get('model_version', '1.1.0-realtime-only')}-weekly-decay180"
    candidate = QiluPulse96ProductionBundle(
        spec=source.spec,
        model=outcome.model,
        preprocessing=source.preprocessing,
        manifest_data=manifest,
    )
    candidate.save(destination)
    return candidate
