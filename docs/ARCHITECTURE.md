# Architecture

## Runtime flow

```text
code root + ignored private runtime root
        |
        v
authorized market/weather inputs
        |
        v
data resolver and readiness checks
        |
        +--> weather snapshot/provenance archive
        |
        +--> causal input builder
                    |
                    v
             QiluPulse-96 bundle
                    |
                    v
             96-slot raw inference
                    |
                    v
             bias/interval calibration
                    |
                    v
        auditable prediction ledger and report package
```

## Source layout

- `src/da_forecast/models/`: model topology, normalization, and calibrated output.
- `src/da_forecast/forecasting/`: business-time visibility contracts.
- `src/da_forecast/features/`: deterministic calendar features.
- `src/da_forecast/sources/`: parquet, Excel, Open-Meteo, and provenance adapters.
- `src/da_forecast/production/`: bundle, readiness, inference, calibration,
  reporting, and workflow orchestration.
- `src/da_forecast/system/`: prediction registry and read-only explanations.
- `scripts/`: explicit command-line entry points.
- `tests/`: unit, contract, bundle, and synthetic workflow tests.

## Public versus private state

The checkout is the code root. An operational run should use a separate
ignored runtime root, conventionally `.private-runtime/`. Its `data/`,
`artifacts/`, `runs/`, `logs/`, and `output/` directories hold local state and
must not be copied into a public commit. `ProductionWorkflow(root)` remains
compatible with the historical single-root test interface; production callers
should pass `runtime_root` explicitly.

The runner also accepts an explicit manual workbook path. It never needs to
discover an operator workbook by walking a machine-specific parent directory
when that argument is supplied.

## Weather source contract

`WeatherRuntimeV1(weather_source="existing")` is the reproducible offline
mode. It requires a complete local history cache and one exact JSON snapshot per
Shandong station for the target date, all issued at `T-1 12:00 Asia/Shanghai`.
Missing or mismatched snapshots are blocking conditions. The mode does not
fall back to the current Open-Meteo response. Network acquisition is a separate
explicit mode and must be documented by the operator.

The public adapter also exposes a separate historical Single Runs acquisition
path. It records the weather model's UTC initialization time in the raw request
while the runtime archive records the QiluPulse business `as-of` contract
independently. Historical observed-cache requests validate internal hourly
coverage rather than relying only on minimum and maximum timestamps. Weather
completion merges the required causal window into an existing private cache so
calibration replay cannot accidentally trim a wider history.

## Contract invariants

- one target date produces exactly 96 fifteen-minute rows;
- target-date labels and unavailable future data are rejected;
- quantiles remain ordered `P10 <= P50 <= P90`;
- final reports require active post-processing and a complete ledger;
- model and bundle checksums are stored with the result metadata;
- explanation output is read-only with respect to prediction values.
