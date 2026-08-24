# Code and data provenance ledger

This file is an explicit release gate for the curated QiluPulse-96 source
tree. A clean Git history and a new copyright header do not, by themselves,
prove that every retained line is owned by `XYuki`. For this release, the
maintainer has confirmed the publication authority for the retained source and
authorized aggregate experiment summary; no identity document or private proof
is stored in this repository.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `REWRITTEN_FOR_PUBLIC_BOUNDARY` | Rewritten or newly authored for this staging tree; still subject to maintainer ownership confirmation. |
| `OWNER_CONFIRMATION_REQUIRED` | Retained from the local implementation and useful to the public closure, but local Git history is not sufficient evidence of redistribution rights. |
| `MAINTAINER_CONFIRMED` | The maintainer confirmed publication authority for this release; third-party, data-provider, and model-weight terms remain separate gates. |
| `PUBLIC_RESEARCH_DERIVATIVE` | A maintainer-authorized, field-minimized data derivative is included for research; the data provider's redistribution terms remain a separate gate from the code license. |
| `THIRD_PARTY` | Distributed under the upstream package or source license; Apache-2.0 does not replace those terms. |
| `EXCLUDED` | Intentionally absent from the public tree. |

## Source ledger

| Public path or group | Status | Evidence / release action |
| --- | --- | --- |
| `src/da_forecast/config.py`, package `__init__.py` files | `REWRITTEN_FOR_PUBLIC_BOUNDARY` | European constants and exports were removed; maintainer publication confirmation is recorded for this release. |
| `src/da_forecast/sources/cache.py` | `REWRITTEN_FOR_PUBLIC_BOUNDARY` | Rewritten as a provider-neutral cache with an explicit base directory; maintainer publication confirmation is recorded for this release. |
| `src/da_forecast/sources/openmeteo.py` | `REWRITTEN_FOR_PUBLIC_BOUNDARY` | Shandong-only station map and public API contract; European coordinates and wording were removed; maintainer publication confirmation is recorded for this release. |
| `src/da_forecast/features/`, `src/da_forecast/forecasting/` | `MAINTAINER_CONFIRMED` | Required by the public closure. The maintainer confirmed that the retained implementation is authored or authorized for redistribution. |
| `src/da_forecast/models/` | `MAINTAINER_CONFIRMED` | Includes the QiluPulse-96 topology, normalization, long-context, and calibration implementation. The maintainer confirmed source publication authority; weight and training-data rights remain separate, and weights stay excluded. |
| `src/da_forecast/production/` | `MAINTAINER_CONFIRMED` | The maintainer confirmed publication authority for the retained workflow. The code license does not grant rights to runtime data or model artifacts. |
| `src/da_forecast/production/feature_ablation_v1.py`, `scripts/audit_qilupulse96_feature_ablation.py`, `tests/test_feature_ablation_v1.py`, `docs/FEATURE_ABLATION.md` | `REWRITTEN_FOR_PUBLIC_BOUNDARY` | Offline audit masks, metrics, CLI, tests, and method documentation; maintainer publication confirmation is recorded for this release. |
| `src/da_forecast/sources/manual_realtime_xlsx.py`, `shandong_market_xlsx.py`, `spatial_weather_v01.py`, `weather_provenance.py` | `MAINTAINER_CONFIRMED` | Shandong-specific adapters retained for the public workflow; the maintainer confirmed source publication authority, while provider terms govern runtime data. |
| `src/da_forecast/system/` | `MAINTAINER_CONFIRMED` | The maintainer confirmed publication authority for the retained public system and prediction interfaces. |
| `scripts/`, `tests/`, `.github/`, and public documentation | `MAINTAINER_CONFIRMED` | Public-facing wrappers, tests, boundary guard, and documentation were curated for this release; maintainer publication confirmation is recorded. |
| `data/public/*.xlsx` and `data/public/MANIFEST.json` | `PUBLIC_RESEARCH_DERIVATIVE` | Four field-minimized Shandong research copies and a machine-independent manifest are included; the source workbooks, provider terms, and raw fields remain outside the public tree. See `docs/PUBLIC_DATA.md`. |
| `data/raw/`, `data/bootstrap/`, `data/calibration/`, `artifacts/`, `runs/`, `logs/`, `output/` | `EXCLUDED` | Private runtime inputs, local ledgers, checkpoints, and generated results are not part of the public Git tree. |
| Old European modules, notebooks, GUI/EXE/Streamlit archive, research outputs | `EXCLUDED` | Not copied into the clean public history. |
| Production `.pt`, `.pth`, `.ckpt`, `.onnx`, `.npz`, and `.safetensors` files | `EXCLUDED` | No model artifact is included until ownership, data provenance, and redistribution permission are evidenced. |

## Maintainer publication confirmation

For this release, the maintainer confirmed outside the generated runtime data
that:

1. whether the code was written by `XYuki` or by an authorized contributor;
2. whether any older code was copied or adapted, including source URL/revision
   and license;
3. whether the code depends on a third-party notice that must be retained;
4. whether the associated input data, feature engineering, and model weights
   may be redistributed; and
5. the retained aggregate experiment summary may be disclosed in rounded form;
6. the confirmation does not include or require publishing private identity
   documents, contracts, personal contact details, or raw data.

The confirmation is recorded as a maintainer release decision dated
`2026-08-24`. It does not override third-party licenses, market/weather
provider terms, model-weight restrictions, or applicable law.

## Data and model boundary

Market and weather providers may impose terms that are separate from the code
license. The four files under `data/public/` are derived research copies, not a
grant of rights to any source data beyond the maintainer's release decision.
Synthetic fixtures are generated by repository scripts and are not a license
for real data. Model weights remain excluded by default; see
`docs/MODEL_RELEASE.md` for the separate artifact gate.

The old private repository history is not part of this public Git history. Its
existence in a private working tree is not a redistribution grant for the
QiluPulse-96 source tree.
