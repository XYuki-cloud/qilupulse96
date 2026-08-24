# QiluPulse-96 public repository rules

## Scope

- This repository contains the public Shandong QiluPulse-96 production subset.
- Do not add the former European research tree, private market files, local
  weather snapshots, production runs, or model checkpoints.
- Keep code under the checkout root and keep operational data under the
  ignored `.private-runtime/` root or another explicitly supplied runtime
  directory. Do not make production code discover personal parent directories.

## Data and model boundaries

- Never commit secrets, API keys, cookies, certificates, user workbooks,
  raw market data, weather snapshots, calibration ledgers, or generated runs.
- A production model bundle is an external input unless its ownership and
  redistribution evidence has been recorded in `docs/MODEL_RELEASE.md`.
- Missing data or weights must produce an explicit blocked/error state; never
  fabricate a successful forecast.
- The default CLI weather mode is `--weather-source existing`. It requires the
  exact target-day forecast issue at `T-1 12:00 Asia/Shanghai`, all 16 cities,
  and a complete local history cache. It must not silently call the current
  weather API or substitute another issue.

## Forecast contract

- The target is 96 fifteen-minute slots in `Asia/Shanghai`.
- The decision time is the target day minus one day at 12:00 or later.
- The realtime endpoint, target-day visibility boundary, calibration status,
  parameter checksum, and source hashes must remain auditable.
- Production commands must pass `--runtime-root` and, when applicable,
  `--manual-workbook` explicitly.
- Explanations may describe a result but must not change prediction values.

## Verification

Before proposing a public release, run:

```powershell
uv run pytest -q
uv run python -m compileall -q src scripts
git diff --check
```

Also inspect `git status --short --untracked-files=all`, scan for secrets and
absolute local paths, and verify that no ignored data or weights are staged.
