"""Causal weekly training-set selection for the AdaLN drift experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from da_forecast.config import TIMEZONE


def select_weekly_training_positions(
    target_positions: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    label_cutoff: str | pd.Timestamp,
    max_training_days: int = 365,
    validation_days: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causally eligible fit/validation daily anchors ending at ``label_cutoff``."""
    if max_training_days <= validation_days or validation_days < 1:
        raise ValueError("max_training_days must exceed a positive validation_days")
    cutoff = pd.Timestamp(label_cutoff)
    cutoff = cutoff.tz_localize(TIMEZONE) if cutoff.tz is None else cutoff.tz_convert(TIMEZONE)
    positions = np.asarray(target_positions, dtype=int)
    dates = index[positions].normalize()
    eligible = positions[dates <= cutoff.normalize()]
    selected = eligible[-max_training_days:]
    if len(selected) < max_training_days:
        raise ValueError(
            f"Need {max_training_days} complete target days through {cutoff.date()}, found {len(selected)}"
        )
    return selected[:-validation_days], selected[-validation_days:]


def validate_initial_train_end(
    initial_train_end: str | pd.Timestamp, test_start: str | pd.Timestamp,
) -> None:
    """Reject labels that were unavailable at the first target's decision time."""
    initial_end = pd.Timestamp(initial_train_end)
    initial_end = initial_end.tz_localize(TIMEZONE) if initial_end.tz is None else initial_end.tz_convert(TIMEZONE)
    first_target = pd.Timestamp(test_start)
    first_target = first_target.tz_localize(TIMEZONE) if first_target.tz is None else first_target.tz_convert(TIMEZONE)
    if initial_end.normalize() > (first_target.normalize() - pd.Timedelta(days=2)):
        raise ValueError("Initial training labels must end no later than the first target's T-2 day")
