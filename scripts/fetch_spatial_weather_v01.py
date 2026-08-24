#!/usr/bin/env python3
"""Cache the v0.1 16-city detailed Open-Meteo Archive weather panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, TIMEZONE
from da_forecast.sources.spatial_weather_v01 import fetch_observed_spatial_weather


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-08-01")
    parser.add_argument("--end", default="2026-08-14", help="Exclusive Shanghai date")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz=TIMEZONE)
    end = pd.Timestamp(args.end, tz=TIMEZONE)
    if end <= start:
        raise ValueError("--end must be after --start")
    panel = fetch_observed_spatial_weather(start, end, cache_dir=args.raw_dir)
    first = next(iter(panel.values()))
    print(f"Cached 16-city detailed observed weather: {first.index.min()} -> {first.index.max()} ({len(first)} hours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
