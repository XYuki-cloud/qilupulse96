"""Strictly causal calibration backed by the production prediction ledger."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from da_forecast.models.online_calibration_v01 import CausalResidualCalibrator


def load_settled_ledger_history(
    root: str | Path,
    *,
    actual_prices: pd.Series,
    bundle_parameter_checksum: str | None = None,
) -> pd.DataFrame:
    """Read prior official forecasts and join labels only after settlement."""
    root = Path(root)
    details: list[pd.DataFrame] = []
    for metadata_path in sorted((root / "runs" / "predictions").glob("*/run_metadata.json")):
        metadata = pd.read_json(metadata_path, typ="series")
        if metadata.get("publish_status") != "official_published":
            continue
        if bundle_parameter_checksum is not None and str(metadata.get("parameter_checksum", "")) != str(bundle_parameter_checksum):
            continue
        detail_path = metadata_path.parent / "prediction_detail.csv"
        if detail_path.is_file():
            details.append(pd.read_csv(detail_path))
    if not details:
        return pd.DataFrame()
    history = pd.concat(details, ignore_index=True)
    timestamps = pd.to_datetime(history["market_date"] + " " + history["period_start"]).dt.tz_localize("Asia/Shanghai")
    lookup = actual_prices.copy()
    lookup.index = lookup.index.tz_localize("Asia/Shanghai") if lookup.index.tz is None else lookup.index.tz_convert("Asia/Shanghai")
    history["actual_cny_mwh"] = lookup.reindex(timestamps).to_numpy()
    return history.dropna(subset=["actual_cny_mwh"]).copy()


def calibrate_final(prediction: pd.DataFrame, *, history: pd.DataFrame, target_date: str | pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    if history.empty:
        history = pd.DataFrame(columns=[
            "market_date", "period_start", "actual_cny_mwh", "predicted_cny_mwh", "p10_cny_mwh", "p50_cny_mwh", "p90_cny_mwh",
        ])
    return CausalResidualCalibrator(enable_interval=True).calibrate(prediction, history, target_date=target_date)
