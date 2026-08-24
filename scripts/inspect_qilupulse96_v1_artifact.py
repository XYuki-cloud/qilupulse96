#!/usr/bin/env python3
"""Print the audited manifest of a QiluPulse-96 v1.0 kernel artifact.

This is intentionally inspection-only.  It loads a serialized model bundle to
validate its topology and checksum, but does not construct features, request
weather, train a model, or produce a market forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from da_forecast.models.qilupulse96_v1 import QiluPulse96V1Artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a QiluPulse-96 v1.0 kernel artifact")
    parser.add_argument("--artifact", required=True, type=Path, help="Path to a .pt QiluPulse-96 v1.0 artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = QiluPulse96V1Artifact.load(args.artifact)
    print(json.dumps(artifact.manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
