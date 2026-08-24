#!/usr/bin/env python3
"""Create, evaluate and save one realtime-only 180-day-decay fine-tune candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def _root(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _daily_bootstrap_delta(base: pd.DataFrame, candidate: pd.DataFrame, *, draws: int = 2000, seed: int = 7) -> dict[str, float | int]:
    base_daily = (base["predicted_cny_mwh"] - base["actual_cny_mwh"]).abs().groupby(base["market_date"]).mean()
    candidate_daily = (candidate["predicted_cny_mwh"] - candidate["actual_cny_mwh"]).abs().groupby(candidate["market_date"]).mean()
    dates = base_daily.index.intersection(candidate_daily.index).to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.asarray([
        float((candidate_daily.loc[sample] - base_daily.loc[sample]).mean())
        for sample in (rng.choice(dates, size=len(dates), replace=True) for _ in range(draws))
    ])
    return {
        "days": int(len(dates)),
        "draws": draws,
        "candidate_minus_base_mae": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=_root, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True, help="Explicit source bundle to fine-tune")
    parser.add_argument("--label-cutoff", required=True, help="Last complete settlement day allowed as a training label")
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    bundle_path = args.bundle_path.expanduser()
    if not bundle_path.is_absolute():
        bundle_path = args.root / bundle_path
    bundle_path = bundle_path.resolve()
    sys.path.insert(0, str(args.root / "src"))

    from da_forecast.production.workflow_v1 import ProductionWorkflow
    from da_forecast.production.weekly_finetune_v1 import (
        evaluate_realtime_only_bundle,
        fine_tune_realtime_only_bundle,
        forecast_metrics,
        load_realtime_only_finetune_inputs,
        save_finetuned_bundle,
    )

    source = ProductionWorkflow(args.root).resolve_bundle(bundle_path)
    inputs = load_realtime_only_finetune_inputs(args.root, label_cutoff=args.label_cutoff)
    def progress(message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat()} {message}"
        print(line, flush=True)
        if args.log_file is not None:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            with args.log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    outcome = fine_tune_realtime_only_bundle(
        source,
        inputs,
        label_cutoff=args.label_cutoff,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        progress=progress,
    )
    candidate = save_finetuned_bundle(source, outcome, args.output_bundle)
    evaluation_positions = outcome.validation_positions
    baseline_detail = evaluate_realtime_only_bundle(source, inputs, evaluation_positions)
    candidate_detail = evaluate_realtime_only_bundle(candidate, inputs, evaluation_positions)
    baseline_metrics = forecast_metrics(baseline_detail)
    candidate_metrics = forecast_metrics(candidate_detail)
    bootstrap = _daily_bootstrap_delta(baseline_detail, candidate_detail)
    accepted = bool(
        candidate_metrics["mae_cny_mwh"] < baseline_metrics["mae_cny_mwh"]
        and bootstrap["ci95_high"] <= 0.0
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    baseline_detail.to_csv(args.report.with_name("baseline_validation_detail.csv"), index=False)
    candidate_detail.to_csv(args.report.with_name("candidate_validation_detail.csv"), index=False)
    payload = {
        "source_bundle": str(source.root),
        "candidate_bundle": str(candidate.root),
        "source_parameter_checksum": source.parameter_checksum,
        "candidate_parameter_checksum": candidate.parameter_checksum,
        "fine_tune": outcome.metadata,
        "validation": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "daily_block_bootstrap": bootstrap,
            "deployment_gate": {
                "candidate_mae_strictly_lower": candidate_metrics["mae_cny_mwh"] < baseline_metrics["mae_cny_mwh"],
                "bootstrap_ci95_upper_not_positive": bootstrap["ci95_high"] <= 0.0,
                "accepted": accepted,
            },
        },
        "data_contract": {
            "price_features": "realtime_only",
            "day_ahead_read": False,
            "weather_kind": "observed_proxy",
            "validation_days": 14,
        },
    }
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
