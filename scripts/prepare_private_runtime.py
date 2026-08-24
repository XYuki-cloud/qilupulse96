"""Stage authorized local inputs into the ignored QiluPulse-96 runtime tree.

The public repository contains the code contract only.  This command copies
operator-provided data and an explicitly selected bundle into a separate,
ignored runtime root so a production run never needs to read the old archive
or the user's external data directory directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _resolve(value: Path, *, base: Path) -> Path:
    return (value if value.is_absolute() else base / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(_sha256(item).encode("ascii"))
        count += 1
        total += item.stat().st_size
    return digest.hexdigest(), count, total


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _copy_file(
    source: Path,
    destination: Path,
    *,
    label: str,
    runtime_root: Path,
    records: list[dict[str, object]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "label": label,
            "destination": destination.relative_to(runtime_root).as_posix(),
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    )


def _copy_directory(
    source: Path,
    destination: Path,
    *,
    label: str,
    runtime_root: Path,
    records: list[dict[str, object]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    tree_hash, file_count, total_bytes = _tree_sha256(destination)
    records.append(
        {
            "label": label,
            "destination": destination.relative_to(runtime_root).as_posix(),
            "tree_sha256": tree_hash,
            "file_count": file_count,
            "bytes": total_bytes,
        }
    )


def prepare_runtime(
    *,
    public_root: Path,
    runtime_root: Path,
    archive_root: Path,
    manual_workbook: Path,
    bundle_path: Path,
) -> dict[str, object]:
    """Copy the minimum local inputs needed by the realtime-only workflow."""
    public_root = public_root.resolve()
    runtime_root = runtime_root.resolve()
    archive_root = archive_root.resolve()
    manual_workbook = _require_file(manual_workbook.resolve(), "manual workbook")
    bundle_path = _require_directory(bundle_path.resolve(), "bundle directory")
    _require_file(bundle_path / "manifest.json", "bundle manifest")
    _require_file(bundle_path / "model_state.pt", "bundle model weights")

    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    parameter_checksum = str(manifest.get("parameter_checksum") or "")
    if len(parameter_checksum) != 64:
        raise ValueError("bundle manifest does not contain a SHA-256 parameter_checksum")

    records: list[dict[str, object]] = []
    _copy_file(
        manual_workbook,
        runtime_root / "data" / "manual_realtime_prices.xlsx",
        label="operator_manual_realtime_workbook",
        runtime_root=runtime_root,
        records=records,
    )

    bundle_destination = runtime_root / "artifacts" / "prediction-layer" / "bundles" / bundle_path.name
    _copy_directory(
        bundle_path,
        bundle_destination,
        label="authorized_prediction_bundle",
        runtime_root=runtime_root,
        records=records,
    )

    archive_data = archive_root / "data"
    _copy_file(
        _require_file(
            archive_data / "bootstrap" / "curated" / "shandong_all_network" / "SD" / "realtime_prices_15min.parquet",
            "curated realtime price parquet",
        ),
        runtime_root / "data" / "bootstrap" / "curated" / "shandong_all_network" / "SD" / "realtime_prices_15min.parquet",
        label="historical_realtime_prices",
        runtime_root=runtime_root,
        records=records,
    )
    _copy_directory(
        _require_directory(archive_data / "reference" / "calendar", "calendar reference"),
        runtime_root / "data" / "reference" / "calendar",
        label="calendar_reference",
        runtime_root=runtime_root,
        records=records,
    )

    calibration_source = archive_data / "calibration" / "realtime_only" / parameter_checksum
    _copy_directory(
        _require_directory(calibration_source, "checksum-scoped calibration ledger"),
        runtime_root / "data" / "calibration" / "realtime_only" / parameter_checksum,
        label="checksum_scoped_calibration_ledger",
        runtime_root=runtime_root,
        records=records,
    )

    _copy_directory(
        _require_directory(archive_data / "raw" / "weather_history_v1", "historical weather cache"),
        runtime_root / "data" / "raw" / "weather_history_v1",
        label="historical_weather_cache",
        runtime_root=runtime_root,
        records=records,
    )
    _copy_directory(
        _require_directory(archive_data / "raw" / "openmeteo_spatial_v01_quarter", "quarterly spatial weather cache"),
        runtime_root / "data" / "raw" / "openmeteo_spatial_v01_quarter",
        label="quarterly_spatial_weather_cache",
        runtime_root=runtime_root,
        records=records,
    )
    _copy_directory(
        _require_directory(archive_data / "raw" / "weather_forecasts", "archived weather snapshots"),
        runtime_root / "data" / "raw" / "weather_forecasts",
        label="archived_weather_forecast_snapshots",
        runtime_root=runtime_root,
        records=records,
    )

    runtime_root.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_root / "runtime_manifest.json"
    payload: dict[str, object] = {
        "schema_version": "private_runtime_manifest_v1",
        "public_root": public_root.name,
        "status": "prepared",
        "manual_workbook": records[0],
        "bundle": {
            "name": bundle_path.name,
            "parameter_checksum": parameter_checksum,
            "destination": bundle_destination.relative_to(runtime_root).as_posix(),
        },
        "records": records,
        "weather_contract": {
            "mode": "existing",
            "required_target_as_of": "T-1 12:00 Asia/Shanghai",
            "network_fallback": False,
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"runtime_root": str(runtime_root), "manifest": str(manifest_path), "records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=_path, required=True)
    parser.add_argument("--runtime-root", type=_path, required=True)
    parser.add_argument("--archive-root", type=_path, required=True)
    parser.add_argument("--manual-workbook", type=_path, required=True)
    parser.add_argument("--bundle-path", type=_path, required=True)
    args = parser.parse_args(argv)

    public_root = args.public_root.resolve()
    runtime_root = _resolve(args.runtime_root, base=public_root)
    archive_root = args.archive_root.resolve()
    manual_workbook = _resolve(args.manual_workbook, base=public_root)
    bundle_path = _resolve(args.bundle_path, base=archive_root)
    result = prepare_runtime(
        public_root=public_root,
        runtime_root=runtime_root,
        archive_root=archive_root,
        manual_workbook=manual_workbook,
        bundle_path=bundle_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
