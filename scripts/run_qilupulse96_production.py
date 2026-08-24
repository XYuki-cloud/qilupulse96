"""Run the production realtime-only QiluPulse-96 workflow without a GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _root(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=_root, required=True)
    parser.add_argument(
        "--runtime-root",
        type=_root,
        help="Ignored local data/output root. Defaults to <root>/.private-runtime.",
    )
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--bundle-path",
        type=Path,
        required=True,
        help="Explicit authorized bundle directory. Does not use or update a default pointer.",
    )
    parser.add_argument(
        "--manual-workbook",
        type=Path,
        help="Explicit operator realtime workbook; relative paths are resolved from --root.",
    )
    parser.add_argument(
        "--weather-source",
        choices=("existing", "fetch"),
        default="existing",
        help="Use archived snapshots only (default) or explicitly allow API acquisition.",
    )
    args = parser.parse_args(argv)
    runtime_root = args.runtime_root or (args.root / ".private-runtime").resolve()
    bundle_path = args.bundle_path.expanduser()
    if not bundle_path.is_absolute():
        bundle_path = args.root / bundle_path
    bundle_path = bundle_path.resolve()
    manual_workbook = args.manual_workbook.expanduser() if args.manual_workbook is not None else None
    if manual_workbook is not None and not manual_workbook.is_absolute():
        manual_workbook = (args.root / manual_workbook).resolve()
    sys.path.insert(0, str(args.root / "src"))
    from da_forecast.production.workflow_v1 import ProductionWorkflow

    messages: list[str] = []
    workflow = ProductionWorkflow(
        args.root,
        runtime_root=runtime_root,
        manual_workbook=manual_workbook,
        weather_source=args.weather_source,
    )
    try:
        result = workflow.run_prediction_draft(
            target_date=args.target_date,
            as_of=args.as_of,
            fetch_weather=True,
            progress=messages.append,
            bundle_path=bundle_path,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        message = str(exc)
        lowered = message.lower()
        if "target forecast snapshot missing" in lowered:
            reason = "target weather snapshot missing"
        elif "issued_at mismatch" in lowered:
            reason = "target weather snapshot timestamp mismatch"
        else:
            reason = "production input or model unavailable"
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": reason,
                    "error_type": type(exc).__name__,
                    "message": message,
                    "progress": messages,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    json_path, markdown_path = workflow.write_report(
        result, target_date=args.target_date, as_of=args.as_of
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "publish_status": result.publish_status,
                "detail_path": str(result.detail_path),
                "metadata_path": str(runtime_root / "runs" / "predictions" / result.run_id / "run_metadata.json"),
                "summary_json": str(json_path),
                "summary_markdown": str(markdown_path),
                "final_prediction_png": str(workflow.last_final_artifacts.plot_path) if workflow.last_final_artifacts else None,
                "final_prediction_xlsx": str(workflow.last_final_artifacts.excel_path) if workflow.last_final_artifacts else None,
                "final_report_markdown": str(workflow.last_final_artifacts.markdown_path) if workflow.last_final_artifacts else None,
                "final_report_json": str(workflow.last_final_artifacts.report_json_path) if workflow.last_final_artifacts else None,
                "whitebox_explanation_json": str(workflow.last_final_artifacts.explanation_json_path) if workflow.last_final_artifacts else None,
                "explanation_status": workflow.last_final_artifacts.explanation_status if workflow.last_final_artifacts else None,
                "ai_interpretation_status": "pending" if workflow.last_final_artifacts else None,
                "progress": messages,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
