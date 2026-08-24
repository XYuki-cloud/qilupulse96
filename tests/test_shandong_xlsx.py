"""Tests for the Shandong matrix-Excel -> long-CSV converter."""

import numpy as np
import pandas as pd
import pytest

from import_shandong_xlsx import melt_matrix, _parse_date_header


def _fake_matrix(days, hours, values, title="test"):
    """Build a matrix like the real workbook: row0 title, row1 header, rows hours."""
    rows = [[title] + [None] * (len(days) + 1)]
    header = ["\u65f6\u95f4"] + days + ["\u5e73\u5747"]
    rows.append(header)
    for h, vals in zip(hours, values):
        rows.append([f"{h}\u65f6"] + list(vals) + [None])
    return pd.DataFrame(rows)


class TestMeltMatrix:
    def test_basic_melt(self):
        days = ["05-01(\u4e94)", "05-02(\u516d)"]
        hours = [1, 2]
        values = [[100.0, 200.0], [110.0, 210.0]]
        df = _fake_matrix(days, hours, values)
        out = melt_matrix(df, year=2026, value_col="price")
        assert len(out) == 4
        assert out["price"].tolist() == [100.0, 110.0, 200.0, 210.0]
        assert out["\u65f6\u95f4"].iloc[0] == pd.Timestamp("2026-05-01 00:00")
        assert out["\u65f6\u95f4"].iloc[2] == pd.Timestamp("2026-05-02 00:00")
        assert out["\u65f6\u95f4"].iloc[3] == pd.Timestamp("2026-05-02 01:00")

    def test_weekday_validation_passes(self):
        # 2026-05-01 is Friday (\u4e94)
        assert _parse_date_header("05-01(\u4e94)", 2026).weekday() == 4

    def test_weekday_validation_fails_on_wrong_year(self):
        # 2024-05-01 is Wednesday, so label (\u4e94) must fail
        with pytest.raises(ValueError, match="wrong year"):
            _parse_date_header("05-01(\u4e94)", 2024)

    def test_missing_values_dropped(self):
        # 2 days, 1 hour; second day value is NaN -> dropped
        days = ["05-01(\u4e94)", "05-02(\u516d)"]
        hours = [1]
        values = [[100.0, np.nan]]
        df = _fake_matrix(days, hours, values)
        out = melt_matrix(df, year=2026, value_col="price")
        assert len(out) == 1
        assert out["price"].tolist() == [100.0]
