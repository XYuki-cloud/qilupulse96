# QiluPulse-96 Feature Ablation Audit

本文件定义天气、日历和近期价格状态变量的离线消融审计。它描述的是模型输入
依赖性测试，不是因果识别，也不是新的训练或模型选择流程。

## Scope and fixed contract

The audit is intentionally fixed to the settled validation window
`2026-08-07` through `2026-08-20`. It:

- loads one explicitly selected, checksum-validated QiluPulse-96 bundle;
- uses the same causal realtime cutoff, `T-1 10:45 Asia/Shanghai`;
- uses the existing `observed_proxy` weather panel and calendar reference;
- runs raw model inference only;
- does not calibrate, publish, promote, retrain, or call a weather API;
- writes all private inputs and generated results below `runtime-root`.

The `full` replay is compared slot by slot with the checksum-scoped calibration
ledger. A maximum point difference greater than `1e-4 CNY/MWh` blocks the
experiment. This gate prevents a silently different input or numerical runtime
from being presented as a feature result.

Run it with an authorized local runtime:

```powershell
uv run python scripts/audit_qilupulse96_feature_ablation.py `
  --root . `
  --runtime-root .private-runtime `
  --bundle-path .private-runtime/artifacts/prediction-layer/bundles/authorized-bundle `
  --manual-workbook .private-runtime/data/manual_realtime_prices.xlsx `
  --start-date 2026-08-07 `
  --end-date 2026-08-20 `
  --device auto
```

`--device auto` resolves to CUDA when available and otherwise to CPU. The
reported device is part of the private experiment manifest; backend changes
can affect strict replay parity at small numerical scales.

The default command is strict. If the selected backend is known to differ from
the backend that produced the historical ledger, an explicit exploratory run
may use `--allow-backend-numeric-drift`. This never relaxes the default gate:
the manifest and report are marked `exploratory_backend_numeric_drift`, and
the result must not be described as a parity-certified replay.

## Interventions

Inputs are already standardized by the production preprocessing state. An
intervention sets selected standardized values to zero, which represents the
training mean without changing tensor shape or dtype.

Group variants:

| Variant | Intervention |
| --- | --- |
| `full` | All production inputs retained. |
| `weather_meteorology_off` | Zero the 18 meteorology variables; retain seven solar/geometry variables. |
| `weather_solar_geometry_off` | Zero the seven solar/geometry variables; retain 18 meteorology variables. |
| `weather_all_off` | Zero all 25 weather variables; pressure test only. |
| `calendar_date_off` | Retain `slot_sin` and `slot_cos`; zero the 12 date/property columns. |
| `calendar_all_off` | Zero all 14 calendar columns; distribution-shift pressure test only. |
| `price_state_off` | Zero the five state variables in both state input locations. |

Single-variable variants use the names `weather:<column>`,
`calendar:<column>`, and `state:<column>`. A weather intervention is applied
to every station in both history and target-day tensors. Calendar interventions
are applied to history and target-day calendar tensors. State interventions
are applied to the standalone state vector and the repeated target extension.

## Measurements

For each day and variant, the audit records MAE, RMSE, mean bias, within-day
correlation, adjacent-slot direction accuracy, negative-probability Brier
score, P10--P90 coverage, and mean interval width. It also records point,
negative-probability, and interval output deltas relative to `full`.

Daily paired differences are defined as `variant - full`. Positive MAE or RMSE
differences mean that the intervention worsened the error on average. The
audit reports worse/better day counts and a 10,000-draw, seed-7 bootstrap
interval over the 14 daily differences. The interval is descriptive because
the sample is small; it is not a standalone hypothesis test.

The report also includes five causal baselines where the required history is
available: a 28-day flat mean, a 28-day flat median, a 28-day same-slot mean,
a 28-day same-slot median, and the previous complete day's same-slot price.

## Current local evidence (exploratory, not a benchmark)

The current private run covered 14 days and 1,344 slots. It used the
`observed_proxy` weather panel, CPU inference, the `T-1 10:45
Asia/Shanghai` realtime cutoff, raw inference without recalibration, and
10,000 bootstrap draws with seed `7`. The public summary below is rounded; the
raw CSV and JSON outputs remain private. The disclosed values are authorized
aggregate summaries only: they contain no row-level prices, raw workbooks,
checksums, model outputs, run identifiers, or report files. Remove them if the
underlying data or derived-result permission changes.

| Evaluation item | Result |
| --- | ---: |
| Full frozen model MAE | 60.82 CNY/MWh |
| Full frozen model RMSE | 80.94 CNY/MWh |
| Full frozen model mean bias | +19.33 CNY/MWh |
| Full frozen model within-day correlation | 0.804 |
| 28-day fixed-mean MAE | 71.82 CNY/MWh |
| Previous-day same-slot MAE | 70.92 CNY/MWh |
| 28-day same-slot-mean MAE | 92.19 CNY/MWh |

| Group intervention | MAE | Paired MAE change | 95% bootstrap interval |
| --- | ---: | ---: | ---: |
| `weather_meteorology_off` | 114.09 | +53.27 | [+25.99, +80.56] |
| `weather_solar_geometry_off` | 71.99 | +11.17 | [-0.41, +24.82] |
| `weather_all_off` | 98.31 | +37.49 | [+10.97, +70.70] |
| `calendar_date_off` | 60.91 | +0.09 | [-6.66, +7.77] |
| `calendar_all_off` (pressure test) | 59.44 | -1.38 | [-7.88, +6.05] |
| `price_state_off` | 68.04 | +7.23 | [+2.28, +12.79] |

Selected single-variable interventions with the largest paired MAE changes
were `weather:apparent_temperature` (+30.66 CNY/MWh),
`weather:wind_speed_100m` (+24.41 CNY/MWh),
`weather:temperature_2m` (+15.41 CNY/MWh), and
`state:recent_price_median` (+4.88 CNY/MWh). Output sensitivity and predictive
contribution are different quantities: a variable can change the output
without improving validation error, and this audit does not identify causal
effects.

The full replay is currently marked `exploratory_backend_numeric_drift`. The
strict replay threshold is `1e-4 CNY/MWh`; the historical ledger was generated
with CUDA, while the current replay used CPU, and the maximum observed backend
drift was approximately `7.55e-4 CNY/MWh`. The results therefore are not a
strict parity certification. Fourteen days are insufficient for claims about
long-term, cross-season, or production performance.

## Interpretation boundary

Large output sensitivity indicates that the fitted model uses a variable or
variable group. It does not demonstrate that the variable improves forecasting.
Evidence for predictive contribution requires a systematic deterioration in
paired validation error after the intervention. Conversely, a small output
change is evidence of low sensitivity under this frozen bundle and validation
window, not proof that the underlying physical variable is unimportant.

Because the weather panel is `observed_proxy`, these results do not establish
performance under a production weather forecast. The 14-day window also does
not support a general claim about all seasons, market regimes, or future model
versions. The audit does not modify any private prediction, bundle, or runtime
artifact.

## Private outputs

Results are written only to:

```text
.private-runtime/runs/feature_ablation/
└── 2026-08-07_2026-08-20_<parameter_checksum>/
    ├── feature_ablation_report.md
    ├── feature_ablation_report.json
    ├── group_summary.csv
    ├── variable_sensitivity.csv
    ├── daily_metrics.csv
    ├── prediction_deltas.csv
    └── experiment_manifest.json
```

These files contain real prices, model outputs, checksums, and runtime
metadata. They are ignored local artifacts and must not be committed or
uploaded with the public source repository.

## 中文说明

该审计固定使用 2026-08-07 至 2026-08-20 的已结算验证日，读取授权 bundle、
`observed_proxy` 历史天气、日历参考和 `T-1 10:45 Asia/Shanghai` 之前的实时价格。
它只做未校准的原始推理，不训练、不校准、不发布、不晋升模型，也不调用天气 API。

`full` 必须与同 checksum 的校准账本逐点一致，最大点差超过
`1e-4 CNY/MWh` 就阻断。消融将已标准化输入置零，表示替换为训练均值，同时保持
张量形状不变。分组测试覆盖气象、太阳几何、日历日期属性、全部日历变量和近期价格
状态；逐变量测试覆盖 25 个天气变量、14 个日历变量和 5 个状态变量。

默认命令严格阻断回放漂移。如果明确知道当前推理后端与生成历史账本的后端不同，才
可以额外使用 `--allow-backend-numeric-drift` 生成探索性结果；报告必须标记为
`exploratory_backend_numeric_drift`，不能称为通过严格 parity 认证的回放。

输出变化只能证明模型对变量敏感；只有消融后按日误差出现稳定、可重复的变差，才可
称为该变量在当前冻结模型和验证窗口内具有预测贡献。结果不能证明天气或日历与价格
之间的因果关系，也不能代表真实 Forecast 天气条件下的生产效果。报告和原始结果只
能保留在 `.private-runtime`，不进入公开 Git。

### 当前本地结果（探索性证据，不是 benchmark）

当前私有实验覆盖 14 天、1,344 个时段，使用 `observed_proxy` 历史天气、CPU
推理、`T-1 10:45 Asia/Shanghai` 实时价格截止点、未重新校准的原始推理，以及随机
种子 `7` 的 10,000 次 bootstrap。公开文档中的数字已四舍五入，原始 CSV、JSON、
真实价格、模型输出和 checksum 仍只保留在私有 runtime。

完整冻结模型的 MAE 为 60.82 CNY/MWh，RMSE 为 80.94 CNY/MWh，平均偏差为
+19.33 CNY/MWh，日内相关系数为 0.804。对照基线的 MAE 为：28 天固定均值
71.82 CNY/MWh、上一日同槽位 70.92 CNY/MWh、28 天同槽位均值 92.19 CNY/MWh。

关闭气象变量后的 MAE 为 114.09 CNY/MWh（相对完整模型 +53.27），关闭全部天气
变量为 98.31（+37.49），关闭近期价格状态变量为 68.04（+7.23），关闭日期类
日历属性并保留时段编码为 60.91（+0.09）。这说明当前冻结模型对天气和近期价格
状态存在输入依赖，而当前 14 天窗口没有显示出稳定的日期类日历收益；它不能证明
天气或日历与价格之间存在因果关系。

该结果标记为 `exploratory_backend_numeric_drift`。严格 parity 阈值为
`1e-4 CNY/MWh`，历史账本由 CUDA 生成，本次回放使用 CPU，观测到的最大后端差异
约为 `7.55e-4 CNY/MWh`。因此它不属于严格 parity 认证结果，也不足以支持长期、
跨季节或生产级结论。
