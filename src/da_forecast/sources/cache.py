"""Small, deterministic Parquet cache used by public data adapters.

The cache deliberately has no knowledge of a particular market or provider.
Callers choose the source, logical zone, and data type that form the relative
path below::

    <base_dir>/<source>/<zone>/<datatype>.parquet

This module stores only caller-provided frames; it never downloads data and
never discovers files outside ``base_dir``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    """Read, write, and merge Parquet frames under one explicit directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def _path(self, source: str, zone: str, datatype: str) -> Path:
        """Return the cache path for one logical data stream."""
        return self.base_dir / source / zone / f"{datatype}.parquet"

    def save(self, source: str, zone: str, datatype: str, frame: pd.DataFrame) -> None:
        """Persist ``frame`` and create only the required parent directory."""
        path = self._path(source, zone, datatype)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    def load(self, source: str, zone: str, datatype: str) -> pd.DataFrame | None:
        """Load one frame, or return ``None`` when no cache entry exists."""
        path = self._path(source, zone, datatype)
        if not path.exists():
            return None

        frame = pd.read_parquet(path)
        if (
            isinstance(frame.index, pd.DatetimeIndex)
            and frame.index.freq is None
            and len(frame) >= 3
        ):
            inferred = pd.infer_freq(frame.index)
            if inferred:
                frame.index.freq = pd.tseries.frequencies.to_offset(inferred)
        return frame

    def merge(
        self,
        source: str,
        zone: str,
        datatype: str,
        new_frame: pd.DataFrame,
    ) -> None:
        """Merge ``new_frame`` by index, with new values winning overlaps."""
        existing = self.load(source, zone, datatype)
        if existing is None:
            self.save(source, zone, datatype, new_frame)
            return

        retained = existing.loc[~existing.index.isin(new_frame.index)]
        combined = pd.concat([retained, new_frame]).sort_index()
        self.save(source, zone, datatype, combined)

    def get_cached_range(
        self,
        source: str,
        zone: str,
        datatype: str,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return the minimum and maximum index values for a cached frame."""
        frame = self.load(source, zone, datatype)
        if frame is None or frame.empty:
            return None
        return frame.index.min(), frame.index.max()
