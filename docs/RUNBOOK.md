# Public runbook

This runbook describes the public interface only. It does not provide private
market files or a production model bundle.

## Install

```powershell
uv sync --locked --dev
uv run python -c "import da_forecast; print(da_forecast.__version__)"
```

## Test and compile

```powershell
uv run pytest -q
uv run python -m compileall -q src scripts
```

## Build synthetic input data

```powershell
uv run python scripts/generate_demo_data.py --days 90 --seed 7
```

Synthetic data is for adapter and test development. It does not provide an
authorized model and must not be reported as a market backtest.

## Prepare a private runtime

Keep authorized inputs and generated output outside the public source state.
The preparation command copies only explicitly selected local assets and
records their hashes in `runtime_manifest.json`:

```powershell
uv run python scripts/prepare_private_runtime.py `
  --public-root . `
  --runtime-root .private-runtime `
  --archive-root path/to/private/archive `
  --manual-workbook path/to/authorized/realtime-workbook.xlsx `
  --bundle-path artifacts/prediction-layer/bundles/authorized-bundle
```

The original workbook, archive, and source data are not modified. The runtime
root is ignored by Git and may contain prices, weather caches, calibration
ledgers, bundle weights, prediction ledgers, and reports.

## Run with an authorized bundle

```powershell
uv run python scripts/run_qilupulse96_production.py `
  --root . `
  --runtime-root .private-runtime `
  --target-date YYYY-MM-DD `
  --as-of YYYY-MM-DDT12:00:00+08:00 `
  --bundle-path .private-runtime/artifacts/prediction-layer/bundles/authorized-bundle `
  --manual-workbook .private-runtime/data/manual_realtime_prices.xlsx `
  --weather-source existing
```

`existing` is the strict offline weather mode. It requires a complete local
history cache and exactly 16 target-day forecast snapshots whose issue time is
the target day minus one day at 12:00 in `Asia/Shanghai`. It does not call the
current weather API when that snapshot is absent. Use `--weather-source fetch`
only when a separate, documented acquisition step is explicitly authorized.

The bundle directory must contain the manifest, state, preprocessing, feature
schema, station schema, calibration configuration, and checksum files expected
by `QiluPulse96ProductionBundle.load`.

## Acquire a historical target forecast issue

If the exact target-day issue is missing, an operator may run a separately
authorized acquisition step with Open-Meteo's Single Runs API. The API model
initialisation time is not the same field as the QiluPulse business `as-of`
time; keep both values in the runtime manifest and review their availability
before using the result:

```powershell
uv run python scripts/fetch_qilupulse96_weather_snapshot.py `
  --runtime-root .private-runtime `
  --target-date YYYY-MM-DD `
  --as-of YYYY-MM-DDT12:00:00+08:00 `
  --model-run YYYY-MM-DDTHH:MM:00Z `
  --model ecmwf_ifs
```

The command requires a complete 24-hour target day for all 16 stations, stores
the raw API response, and writes only to the ignored runtime. It does not
change the production runner's strict `existing` behavior. A model run must be
earlier than the business contract and satisfy the script's conservative
availability check; choosing a run merely because it can be downloaded later
does not prove that it was available at the historical decision time.

If calibration replays a date earlier than the target date, complete the wider
observed-history cache separately:

```powershell
uv run python scripts/complete_private_weather_history.py `
  --runtime-root .private-runtime `
  --start-date YYYY-MM-DD `
  --end-date YYYY-MM-DD
```

Here `--end-date` is exclusive. The command uses the Open-Meteo Archive API,
merges into `weather_history_v1`, and preserves any wider cache already staged.
It must not be used to fill a target forecast or to replace a missing exact
forecast issue with observed data.

## Build or adapt a bundle

The bundle builder requires an explicit checkpoint, training-data snapshot
hash, calendar-reference hash, and both market/weather roots. It will not
silently replace missing production preprocessing with identity transforms:

```powershell
uv run python scripts/build_qilupulse96_production_bundle_v1.py `
  --checkpoint path/to/authorized/checkpoint.pt `
  --output path/to/new/bundle `
  --training-data-snapshot-hash SHA256_OR_AUDIT_ID `
  --calendar-reference-hash SHA256_OR_AUDIT_ID `
  --market-data-root path/to/authorized/market-data `
  --weather-root path/to/authorized/weather-data `
  --realtime-only
```

For loader tests only, replace the two data roots with
`--synthetic-preprocessing`. The resulting manifest is marked
`synthetic_test_only` and must not be presented as a production bundle.

## Read a result

```powershell
uv run python scripts/inspect_qilupulse_result.py --root .private-runtime --target-date YYYY-MM-DD
```

If the inspector reports `blocked`, stop and fix the listed input or audit
condition. Do not rerun a model simply because a report is pending.

When the exact target weather issue is unavailable, the production runner must
return a blocked response with reason `target weather snapshot missing`; it
must not substitute an older issue or create a partial forecast.

## Data-provider obligations

Before using real data, the operator must document the source terms, issue
time, target-date availability, timezone, snapshot/version identity, and
retention permission. Store those records outside the public source tree.
