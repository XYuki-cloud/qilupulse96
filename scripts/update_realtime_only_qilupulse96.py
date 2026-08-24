#!/usr/bin/env python3
"""Manually fine-tune, audit and promote the realtime-only QiluPulse-96 bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import uuid
from zoneinfo import ZoneInfo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, help="Ignored local data/output root; defaults to <root>/.private-runtime")
    parser.add_argument("--bundle-path", type=Path, required=True, help="Explicit current bundle to update")
    parser.add_argument("--manual-workbook", type=Path, help="Explicit realtime workbook; relative paths use --root")
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--operator-note", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    runtime_root = (args.runtime_root.expanduser() if args.runtime_root is not None else root / ".private-runtime").resolve()
    bundle_path = args.bundle_path.expanduser()
    if not bundle_path.is_absolute():
        bundle_path = root / bundle_path
    bundle_path = bundle_path.resolve()
    manual_workbook = args.manual_workbook.expanduser() if args.manual_workbook is not None else None
    if manual_workbook is not None and not manual_workbook.is_absolute():
        manual_workbook = (root / manual_workbook).resolve()
    sys.path.insert(0, str(root / "src"))
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the default manual update path, but torch.cuda.is_available() is false")
    from da_forecast.production.model_update_v1 import daily_block_bootstrap_delta, promote_bundle
    from da_forecast.production.weekly_finetune_v1 import (
        evaluate_realtime_only_bundle,
        fine_tune_realtime_only_bundle,
        forecast_metrics,
        load_realtime_only_finetune_inputs,
        save_finetuned_bundle,
    )
    from da_forecast.production.workflow_v1 import ProductionWorkflow

    target = datetime.fromisoformat(args.target_date).date()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    local_as_of = as_of.replace(tzinfo=ZoneInfo("Asia/Shanghai")) if as_of.tzinfo is None else as_of.astimezone(ZoneInfo("Asia/Shanghai"))
    if local_as_of.date() != target - timedelta(days=1) or local_as_of.hour < 12:
        raise SystemExit("--as-of must be target-date minus one day at 12:00 Asia/Shanghai or later")
    label_cutoff = target - timedelta(days=2)
    source = ProductionWorkflow(
        root,
        runtime_root=runtime_root,
        manual_workbook=manual_workbook,
    ).resolve_bundle(bundle_path)
    source_default = None
    if not args.no_promote:
        try:
            source_pointer_path = source.root.resolve().relative_to(runtime_root)
        except ValueError as exc:
            raise SystemExit(
                "--bundle-path must be under --runtime-root when promotion is requested"
            ) from exc
        source_default = {"bundle_path": source_pointer_path.as_posix()}
    update_id = f"update_{target.isoformat()}_{uuid.uuid4().hex[:10]}"
    update_dir = runtime_root / "runs" / "model_updates" / args.target_date / update_id
    update_dir.mkdir(parents=True, exist_ok=False)
    output_bundle = runtime_root / "artifacts" / "prediction-layer" / "bundles" / (
        f"QiluPulse-96-{label_cutoff.isoformat()}-{update_id}"
    )
    inputs = load_realtime_only_finetune_inputs(
        runtime_root,
        label_cutoff=label_cutoff.isoformat(),
        manual_workbook=manual_workbook,
    )

    def progress(message: str) -> None:
        print(message, flush=True)
        with (update_dir / "training.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    outcome = fine_tune_realtime_only_bundle(
        source,
        inputs,
        label_cutoff=label_cutoff.isoformat(),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        progress=progress,
    )
    candidate = save_finetuned_bundle(source, outcome, output_bundle)
    validation_positions = outcome.validation_positions
    baseline = evaluate_realtime_only_bundle(source, inputs, validation_positions, device=args.device)
    candidate_detail = evaluate_realtime_only_bundle(candidate, inputs, validation_positions, device=args.device)
    validation = {
        "baseline": forecast_metrics(baseline),
        "candidate": forecast_metrics(candidate_detail),
        "daily_block_bootstrap": daily_block_bootstrap_delta(baseline, candidate_detail),
        "deployment_gate": "warning_only",
    }
    baseline.to_csv(update_dir / "baseline_validation_detail.csv", index=False, encoding="utf-8-sig")
    candidate_detail.to_csv(update_dir / "candidate_validation_detail.csv", index=False, encoding="utf-8-sig")
    (update_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (update_dir / "validation_summary.md").write_text(
        "\n".join([
            f"# QiluPulse-96 微调更新（{args.target_date}）",
            "",
            f"- 标签截止：`{label_cutoff.isoformat()}`",
            f"- 设备：`{args.device}`",
            f"- 默认模型 MAE：`{validation['baseline']['mae_cny_mwh']:.2f}`",
            f"- 新模型 MAE：`{validation['candidate']['mae_cny_mwh']:.2f}`",
            "- 晋升规则：人工触发后硬校验通过即晋升，质量指标仅作提示。",
            "",
        ]),
        encoding="utf-8",
    )
    if args.no_promote:
        manifest = {
            "update_id": update_id,
            "promotion_status": "not_requested",
            "candidate_bundle": str(output_bundle),
            "source_parameter_checksum": source.parameter_checksum,
            "candidate_parameter_checksum": candidate.parameter_checksum,
            "operator_note": args.operator_note,
            "training": outcome.metadata,
            "validation": validation,
        }
        (update_dir / "update_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        promotion_status = "not_requested"
    else:
        promote_bundle(
            runtime_root,
            candidate=candidate,
            candidate_path=output_bundle,
            target_date=args.target_date,
            source_default=source_default,
            update_id=update_id,
            operator_note=args.operator_note,
            validation=validation,
            training_metadata=outcome.metadata,
        )
        promotion_status = "promoted_by_operator"
    print(json.dumps({
        "update_id": update_id,
        "target_date": args.target_date,
        "label_cutoff": label_cutoff.isoformat(),
        "candidate_bundle": str(output_bundle),
        "candidate_parameter_checksum": candidate.parameter_checksum,
        "promotion_status": promotion_status,
        "update_dir": str(update_dir),
        "validation": validation,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
