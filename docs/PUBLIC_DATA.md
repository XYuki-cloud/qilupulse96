# Public research data package

This repository contains four maintainer-reviewed Excel derivatives under
[`data/public/`](../data/public/). They are provided to make the workbook
parser, date/time contracts, feature adapters, and historical research path
reproducible. The package is not a production data feed, a forecast service, a
trading signal, or a guarantee of data completeness.

## Files and coverage

| File | Coverage | Contents |
| --- | --- | --- |
| `shandong_market_2024_public.xlsx` | 2024-01-01 to 2024-12-31 | Real-time clearing, day-ahead clearing, and D+1 disclosure fields |
| `shandong_market_2025_public.xlsx` | 2025-01-01 to 2025-12-31 | Real-time clearing, day-ahead clearing, and D+1 disclosure fields |
| `shandong_market_2026-01-01_2026-08-15_public.xlsx` | Workbook rows: 2026-01-01 to 2026-08-15 | Real-time clearing, day-ahead clearing, and D+1 disclosure fields |
| `manual_realtime_prices_2026-08-13_2026-08-22_public.xlsx` | 2026-08-13 to 2026-08-22 | Manual real-time price records |

The three market workbooks contain only these sheets and fields:

- `实时出清数据`: `目标日期`, `时刻`, `实时出清电价`;
- `日前出清数据`: `目标日期`, `时刻`, `日前出清电价`;
- `日前披露数据`: `当前日期`, `目标日期`, `时刻`, `相隔天数`, and the
  eight forecast fields used by the public Shandong adapter.

The manual workbook contains `日期`, `时间`, and `实时电价`. Prices are in
CNY/MWh. Forecast fields with an MW suffix are in MW; `负荷率预测` is a
dimensionless rate. Source end-point labels such as `24:00` are retained where
needed by the existing parser contract.

The manual workbook also contains a duplicate normalized `2026-08-21 00:00`
slot caused by its source time layout. The parser keeps the first row and emits
a warning; the condition is recorded in the manifest and is not silently
treated as independent observations.

The source workbook sheet `实际披露数据` and fields not required by the public
adapter are intentionally excluded. The files in `data/public` are derived
minimal-field copies, not byte-for-byte copies of the source workbooks.

The 2026 workbook contains complete blank placeholder rows for real-time
prices on 2026-08-14 and 2026-08-15, and for day-ahead prices on 2026-08-15.
The importer preserves the disclosure rows but excludes those blank price days
from the canonical price parquet, following the existing parser contract. This
is a known source-data gap, not an imputed value.

## Sanitization and manifest

The public copies were written as new workbooks. The transformation:

1. selected only the allowlisted sheets and fields;
2. normalized date and time cells to portable text values;
3. removed creator, last-modified-by, custom properties, comments, hyperlinks,
   hidden sheets, formulas, external links, and embedded objects; and
4. recorded file size, SHA-256, sheet names, fields, row counts, and coverage
   in [`data/public/MANIFEST.json`](../data/public/MANIFEST.json).

The manifest contains no source path, workbook author, operator identity, or
private runtime detail. The original source files remain outside this Git
repository and are not modified by the public-data workflow.

## Import into a private runtime

The importer validates the exact four-file package, verifies the manifest and
workbook boundary, checks complete 96-slot days and D+1 disclosure dates, and
rejects conflicting overlaps. It does not call a weather API or load a model:

```powershell
uv run python scripts/ingest_public_shandong_workbooks.py `
  --input-dir data/public `
  --runtime-root .private-runtime
```

For a read-only package check:

```powershell
uv run python scripts/ingest_public_shandong_workbooks.py `
  --input-dir data/public `
  --check-only
```

Canonical parquet outputs are written only below the ignored runtime root:

```text
.private-runtime/data/raw/shandong_all_network/SD/
├── realtime_prices_15min.parquet
├── day_ahead_prices_15min.parquet
├── day_ahead_disclosure.parquet
└── manual_realtime_prices_15min.parquet
```

The generated parquet files are local runtime products and are not tracked by
Git. Weather snapshots, calendar confirmations, calibration ledgers, model
bundles, and prediction reports are separate inputs and are not supplied by
this package.

## Rights and limitations

The data package is separate from the Apache-2.0 license applied to the
original code. Redistribution is permitted only under the data provider's
applicable terms and the maintainer's release authorization; Apache-2.0 does
not grant rights to market data. Users must independently verify provenance,
retention, redistribution, market-rule, and research-use obligations before
using the files.

The public data is intended for parser and research reproducibility. It does
not establish model accuracy, live-market coverage, production suitability,
trading performance, or revenue outcomes. The absence of weather and model
artifacts means that the repository cannot reproduce a production forecast
from these four workbooks alone.
