<div align="center">

# QiluPulse-96

**面向山东电力市场的 96 时段电价预测 Python 工作流。**

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

> [!WARNING]
> **实验状态：** QiluPulse-96 是研究与工程原型，当前不具备生产使用条件。
> 不得将它用于交易、财务、运营或监管决策。测试通过以及下方列出的本地结果，
> 都不代表已经验证了真实市场准确率。

QiluPulse-96 提供目标日电价预测工作流的源码和命令行接口。工作流接收经过
授权的市场与天气输入，使用明确的决策时间截止点，加载指定的模型 bundle，
并生成可供检查的结果元数据。

本仓库是源码预览版本，包含 4 个经过字段最小化和元数据清理的 Excel 研究副本，位于
[`data/public/`](data/public/)，用于复现解析器和历史研究流程；但不包含原始完整工作簿、
生产天气快照、私有凭据或生产模型权重。安装成功和测试通过，只能说明软件接口可以运行，
不能据此推断预测准确率或生产部署条件已经具备。

实际运行时，应将代码目录与被 Git 忽略的本地 runtime 目录分开。runtime 目录
用于存放操作者有权使用的输入、模型 bundle、缓存、校准账本和生成报告；这些
内容不属于公开源码树。

## 概览

| 项目 | 当前约定 |
| --- | --- |
| 市场范围 | 山东电力市场 |
| 预测输出 | 一个目标日的 96 个 15 分钟时段值 |
| 决策时间截止点 | `T-1 12:00 Asia/Shanghai` |
| Python 发行包 | `qilupulse96` |
| Python 导入路径 | `da_forecast` |
| 支持环境 | Python 3.11 或更新版本；锁定开发环境使用 [uv](https://docs.astral.sh/uv/) |
| 发布状态 | 实验性源码预览；未通过生产使用资格审查 |
| 本仓库提供 | 源码接口、4 个清理后的研究工作簿、合成 fixture、测试和文档 |
| 本仓库不提供 | 原始完整工作簿、私有运行状态、天气快照和生产模型权重 |

## 当前实验性证据

以下结果来自一次固定模型 bundle 的本地审计，验证区间为
`2026-08-07` 至 `2026-08-20`，共 14 天、1,344 个 15 分钟时段。审计使用
`observed_proxy` 历史天气、`T-1 10:45 Asia/Shanghai` 的因果实时价格截止点，
只执行原始推理，不重新校准，并使用随机种子 `7` 做 10,000 次 bootstrap 重采样。
这些结果属于探索性本地证据，不是公开 benchmark，也不是对未来表现的保证。

| 评估项目 | 结果（除特别说明外，单位为 CNY/MWh） |
| --- | ---: |
| 完整冻结模型 MAE | 60.82 |
| 完整冻结模型 RMSE | 80.94 |
| 完整冻结模型平均偏差 | +19.33 |
| 完整冻结模型日内相关系数 | 0.804 |
| 28 天固定均值基线 MAE | 71.82 |
| 上一日同槽位基线 MAE | 70.92 |
| 28 天同槽位均值基线 MAE | 92.19 |

以上数字是经维护者授权公开的四舍五入聚合统计，不包含行级价格、原始工作簿、
哈希、模型输出、运行标识或报告文件。它们只能作为实验性证据；如果底层数据或
派生结果的再披露权限发生变化，应将这些数字从公开文档中移除。

消融结果显示模型依赖部分输入组，但不能据此建立因果关系：

| 消融设置 | MAE | 相对完整模型变化 |
| --- | ---: | ---: |
| 关闭气象变量 | 114.09 | +53.27 |
| 关闭全部天气变量 | 98.31 | +37.49 |
| 关闭近期价格状态变量 | 68.04 | +7.23 |
| 关闭日期类日历属性（保留时段编码） | 60.91 | +0.09 |

在这个较短窗口内，冻结模型并不是只输出均值；关闭天气变量会明显改变其验证误差。
但关闭日期类日历属性没有显示出稳定的预测收益。上述结果只描述当前冻结模型对输入
的依赖，不能证明天气或日历变量对市场价格存在因果影响，也不能外推到其他季节或
其他市场状态。

本次回放标记为 `exploratory_backend_numeric_drift`：历史参考结果由 CUDA 后端生成，
本次审计使用 CPU 后端。严格 parity 阈值为 `1e-4 CNY/MWh`，观测到的最大后端差异
约为 `7.55e-4 CNY/MWh`，因此该审计不是通过严格 parity 认证的复现结果。

## 项目范围

本仓库包含预测软件本身及其运行边界，不包含数据订阅服务或可直接下载的
训练模型。整个项目仍处于实验阶段；命令行和包名中的“生产”仅为兼容既有接口而保留，
不代表项目已经通过生产资格审查。主要内容包括：

- 面向授权市场和天气数据的输入适配器与就绪检查；
- 目标日特征构建所需的时间和可见性契约；
- QiluPulse-96 模型及其 bundle 格式；
- 推理、校准、输出校验和结果报告接口；
- 命令行入口、契约测试和合成 fixture。

为了兼容现有生产脚本，公开 Python 包的导入路径保留为 `da_forecast`，
发行包名称为 `qilupulse96`。

## 当前契约

当前源码和测试为以下约束提供接口或断言：

- 一个目标日生成 96 个 15 分钟记录；
- 拒绝使用目标日标签或决策时间之后才可获得的数据；
- 使用区间输出时，保持 `P10 <= P50 <= P90` 的顺序；
- 将模型和 bundle checksum 写入结果元数据；
- 生产结果需要满足规定的后处理和账本状态；
- 解释输出不会修改预测值。

这些内容属于软件契约和校验目标。仓库测试通过，不等于真实市场准确率已经
得到验证，也不等于已经完成独立历史数据审计。

## 适用范围与非目标

本仓库适用于：

- 审查源码和公开接口；
- 开发市场/天气适配器的合成数据流程；
- 对公开工作流执行可复现测试；
- 接入使用者有权使用和再分发的市场、天气及模型输入。

本仓库不提供：

- 市场数据订阅服务或有稳定性保证的数据下载服务；
- 携带生产 checkpoint、无需额外输入即可运行的预测结果；
- 交易表现、收益或基准排名的声明；
- 财务、交易、监管或生产运营建议。

## 工作流

```mermaid
flowchart LR
    A[授权的市场与天气输入] --> B[就绪与截止点检查]
    B --> C[因果特征构建]
    C --> D[指定的 QiluPulse-96 bundle]
    D --> E[96 时段推理]
    E --> F[校准与输出校验]
    F --> G[结果元数据与报告包]
```

该流程用于记录输入截止点、模型身份、校准状态和结果元数据。它不能替代部署
操作者对数据来源条款、数据质量和当地市场规则进行检查。

## 仓库边界

| 包含 | 默认不包含 |
| --- | --- |
| 模型结构和公开 Python 接口 | 原始完整工作簿以及私有市场/天气输入 |
| 生产工作流、就绪检查、推理、校准和报告 | 生产天气快照和私有账本 |
| 公开市场/天气适配器与来源接口 | 生产 checkpoint 或模型权重 |
| `data/public/` 下的 4 个清理后研究工作簿 | 其他工作簿、原始字段和仅供内部使用的数据 |
| bundle manifest 和 checksum 校验 | API key、cookie、证书或本地配置 |
| 合成数据工具和契约测试 | 本地运行、日志、报告和生成结果 |
| CLI 入口与 CI 边界检查 | 再分发权尚未确认的代码或产物 |

## 安装与验证

锁定的开发环境使用 Python 3.11 或更新版本，以及 `uv`：

```powershell
uv sync --locked --dev
uv run pytest -q
uv run python -m compileall -q src scripts
uv run python -c "import da_forecast; import da_forecast.production; print(da_forecast.__version__)"
```

生成供适配器开发使用的合成数据：

```powershell
uv run python scripts/generate_demo_data.py --days 90 --seed 7
```

生成的文件默认被 Git 忽略。它们只用于验证软件接口，不是市场数据集、生产
模型，也不能作为真实市场表现的证据。

## 公开研究数据包

仓库内的 `data/public/` 包含 4 个派生工作簿：覆盖 2024 年、2025 年、工作簿行覆盖
2026-01-01 至 2026-08-15 的 3 个市场数据文件，以及覆盖 2026-08-13 至 2026-08-22 的人工实时电价文件。
它们只保留公开山东适配器所需字段。原工作簿中的 `实际披露数据` 工作表、原始作者元数据
和非必要字段均已排除。这些文件用于研究和复现，不是官方数据接口，也不意味着具备生产
数据资格；准确哈希和结构记录在 [`data/public/MANIFEST.json`](data/public/MANIFEST.json)。

只检查包结构，不写入 runtime：

```powershell
uv run python scripts/ingest_public_shandong_workbooks.py `
  --input-dir data/public `
  --check-only
```

将其导入被 Git 忽略的本地 runtime，用于解析和历史研究：

```powershell
uv run python scripts/ingest_public_shandong_workbooks.py `
  --input-dir data/public `
  --runtime-root .private-runtime
```

导入器只会在 `.private-runtime/data/raw/shandong_all_network/SD/` 下写入 canonical
parquet，不会获取天气、创建模型 bundle 或运行预测。2026 文件中 8 月 14—15 日的实时
价格、8 月 15 日的日前价格是完整空白占位行；导入器按现有解析契约排除这些空白价格日，
不会插值或伪造数值。字段、单位、转换规则和再分发限制见
[`docs/PUBLIC_DATA.md`](docs/PUBLIC_DATA.md)。

## 准备本地 private runtime

准备脚本会把明确指定的本地输入复制到被忽略的 runtime 目录，不会修改源归档，
也不会覆盖原始输入文件：

```powershell
uv run python scripts/prepare_private_runtime.py `
  --public-root . `
  --runtime-root .private-runtime `
  --archive-root path/to/private/archive `
  --manual-workbook path/to/authorized/realtime-workbook.xlsx `
  --bundle-path artifacts/prediction-layer/bundles/authorized-bundle
```

生成的布局属于本地运行状态，不是公开发布布局：

```text
.private-runtime/
├── data/                 # 授权的价格、日历、天气和校准数据
├── artifacts/            # 指定的 bundle 及其校验信息
├── runs/                 # 预测账本和报告包
└── runtime_manifest.json # 复制记录与哈希记录
```

准备脚本会记录复制文件和目录的哈希。使用 runtime 做运行决策前，应先检查
manifest 以及模型 bundle 的来源和授权记录。

## 使用授权输入运行

生产入口要求显式提供代码根目录、独立 runtime 根目录、目标日期、`as-of` 时间、
模型 bundle 和实时价格工作簿。默认的 `existing` 天气模式是离线模式：它要求
16 个城市都存在在精确 `T-1 12:00 Asia/Shanghai` 发布时间生成的快照，同时要求
本地历史天气缓存完整。它不会用其他发布时间的快照替代，也不会静默调用当前天气
接口。

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

使用只读检查器读取结果包：

```powershell
uv run python scripts/inspect_qilupulse_result.py `
  --root .private-runtime `
  --target-date YYYY-MM-DD
```

如果目标天气快照缺失，运行器会返回类似
`{"status":"blocked","reason":"target weather snapshot missing"}` 的阻断结果，
不会生成预测。缺失必要数据、日历确认、校准或模型输入时同样必须阻断。公开接口
不使用贡献者机器上的默认 bundle 路径。bundle 构建要求提供训练、日历、市场数据
和天气数据的明确来源，完整命令约定见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md)。

### 经授权的历史天气补齐

严格生产入口在精确快照缺失时不会静默替换输入。操作者可以在完成数据使用授权
后，单独调用 Open-Meteo Single Runs API 获取某次历史模型运行：

```powershell
uv run python scripts/fetch_qilupulse96_weather_snapshot.py `
  --runtime-root .private-runtime `
  --target-date YYYY-MM-DD `
  --as-of YYYY-MM-DDT12:00:00+08:00 `
  --model-run YYYY-MM-DDTHH:MM:00Z `
  --model ecmwf_ifs
```

这里 `--as-of` 是 QiluPulse 的业务决策时间契约，`--model-run` 是天气模型的
UTC 初始化时间，两者不是同一个字段。脚本会记录两种时间，检查 16 个城市是否都
有完整的目标日 24 小时数据，保留原始响应，并且只写入 `.private-runtime`。后来
能够下载某次模型运行，不等于已经证明它在历史决策时刻可用。

如果校准需要回放早于目标日的日期，可以单独补齐更宽的历史天气缓存：

```powershell
uv run python scripts/complete_private_weather_history.py `
  --runtime-root .private-runtime `
  --start-date YYYY-MM-DD `
  --end-date YYYY-MM-DD
```

其中结束日期不包含在内。该命令使用历史 Archive API，将实况/历史天气合并到
`weather_history_v1`，并保留已经存在的更宽缓存；它不能用来替代目标日的精确天气
预报快照。

## 数据、模型与许可证边界

使用真实输入前，操作者应记录数据来源条款、发布时间、目标日可用性、时区、
快照或版本身份，以及保留和再分发许可。运行数据和这些记录应保存在源码树之外。

生产模型权重默认不发布。单独发布模型产物前，需要提供权属、训练数据来源、派生
特征来源、再分发许可、manifest、checksum 和用途说明等证据。具体要求见
[`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md)。

Apache-2.0 仅适用于维护者确认属于原创或已经完成合法再许可的代码。它不会替代
第三方或继承材料的许可证与 NOTICE 要求。再分发衍生版本前，请阅读
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

不要提交真实输入、用户提供的工作簿、私有配置、凭据、校准账本、模型 checkpoint
或生成报告。`.gitignore` 不能替代对 staged 文件清单的人工审查。

## 文档

| 文档 | 说明 |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 运行流程、源码布局和契约不变量 |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | 安装、合成数据、bundle 和结果检查命令 |
| [`docs/PUBLIC_DATA.md`](docs/PUBLIC_DATA.md) | 公开工作簿字段、清理、导入和数据边界 |
| [`docs/FEATURE_ABLATION.md`](docs/FEATURE_ABLATION.md) | 天气、日历和近期价格状态的离线依赖审计 |
| [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md) | 发布模型权重前需要具备的证据 |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | 代码、数据、依赖和再分发来源记录 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 贡献与代码审查要求 |
| [`SECURITY.md`](SECURITY.md) | 私有数据和漏洞报告边界 |

## 限制

本项目是实验性的研究与工程实现，不适合生产部署。当前模型尚未经过足够长时间、
跨季节以及覆盖多种市场状态的独立验证。合成 fixture 和上面的本地审计都依赖特定
输入，不能证明未来表现或真实市场准确率。市场规则、天气服务、数据格式和来源条款
可能变化；未来的使用者需要自行检查数据质量、时间契约、数据提供方条款、当地市场
规则和必要的运营审批。

命令行入口和 `production` 包名只是接口兼容名称，不表示项目已经具备生产资格。
本项目不构成财务、交易、监管或生产运营建议，任何预测结果都不保证实际市场结果。

## 许可证

Copyright 2026 XYuki.

本项目采用 Apache License 2.0。详见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)
和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
