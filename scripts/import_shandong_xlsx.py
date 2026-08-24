#!/usr/bin/env python3
"""Convert a Shandong "matrix" Excel (days x 24 hours) into a long CSV.

Workbook structure per sheet (e.g. "日前出清电价"):
  row 0      : title, e.g. "日前出清电价(元/兆瓦时)（92天对比）"
  row 1      : headers -- 时间 | 05-01(五) ... 07-31(五) | 平均
  rows 2..25 : hours -- 1时 ... 24时
  last column: 平均 (ignored)

Usage:
    uv run python scripts/import_shandong_xlsx.py --input path/to/hourly_matrix.xlsx
    uv run python scripts/import_shandong_xlsx.py --input path/to/input.xlsx --year 2026 --sheet 日前出清电价
    uv run python scripts/import_shandong_xlsx.py --input path/to/input.xlsx --out data/shandong_csv/prices_shandong_da.csv
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Chinese weekday labels: 一=Monday ... 日=Sunday
CN_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}

DEFAULT_OUT = Path(__file__).parent.parent / "data" / "shandong_csv" / "prices_shandong_da.csv"


def _parse_date_header(label: str, year: int) -> date:
    """Parse '05-01(五)' -> date(2026,5,1), validating the weekday label."""
    mm_dd = str(label).split("(")[0].strip()
    d = datetime.strptime(mm_dd, "%m-%d").date().replace(year=year)
    if "(" in str(label):
        wd_label = str(label).split("(")[1][0]
        if wd_label in CN_WEEKDAY:
            expected = CN_WEEKDAY[wd_label]
            actual = d.weekday()
            if actual != expected:
                raise ValueError(
                    f"Header {label!r} says weekday={wd_label} but {d} is "
                    f"weekday {actual} in {year} -- wrong year?"
                )
    return d


def melt_matrix(df: pd.DataFrame, year: int, value_col: str) -> pd.DataFrame:
    """Melt a matrix sheet into a long DataFrame [时间, value_col]."""
    header = df.iloc[1, :]
    hour_labels = df.iloc[2:, 0].astype(str).tolist()
    day_labels = [str(x) for x in header[1:-1]]  # drop 时间 and 平均

    days = [_parse_date_header(x, year) for x in day_labels]
    hours = []
    for lab in hour_labels:
        h = int(str(lab).replace("时", "").strip())
        hours.append(h - 1)  # 1时 -> 00:00

    records = []
    for col_i, day in enumerate(days):
        values = pd.to_numeric(df.iloc[2:, 1 + col_i], errors="coerce")
        for row_i, h in enumerate(hours):
            ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=h)
            records.append((ts, values.iloc[row_i]))

    out = pd.DataFrame(records, columns=["时间", value_col])
    out = out.sort_values("时间").reset_index(drop=True)
    n_missing = int(out[value_col].isna().sum())
    if n_missing:
        print(f"  WARNING: {n_missing} missing values will be dropped")
        out = out.dropna(subset=[value_col])
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert Shandong matrix Excel to long CSV.")
    parser.add_argument("--input", required=True, help="Path to the .xlsx matrix file")
    parser.add_argument("--year", type=int, default=2026, help="Calendar year of the data (validated by weekday labels)")
    parser.add_argument("--sheet", default="日前出清电价", help="Sheet name to convert")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path")
    parser.add_argument("--value-col", default="出清价", help="Value column name in the output CSV")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    df = pd.read_excel(args.input, sheet_name=args.sheet, header=None)
    print(f"Sheet: {args.sheet}  (shape {df.shape})")
    out = melt_matrix(df, args.year, args.value_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(out)} rows -> {out_path}")
    print(f"  period: {out['时间'].min()} .. {out['时间'].max()}")
    print(f"  {args.value_col}: min={out[args.value_col].min():.2f} max={out[args.value_col].max():.2f} "
          f"mean={out[args.value_col].mean():.2f} neg={(out[args.value_col] < 0).sum()}")


if __name__ == "__main__":
    main()
