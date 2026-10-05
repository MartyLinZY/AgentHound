# 数据目录说明

本目录保存实验数据说明、元信息，以及本机准备的原始和处理后数据。当前项目数据流程使用 **AgenTracer** 和 **RootSE**。

## 目录结构

```text
data/
├── raw/          # 原始数据，只读；不纳入 Git
│   ├── agentracer/tracertraj-code-test.parquet
│   └── rootse/    # RootSE 发布文件、README 和 manifest.json
├── processed/    # AgenTracer 适配后的 inputs.jsonl / labels.jsonl；不纳入 Git
├── splits/       # 本地证据分层和样本清单；不纳入 Git
└── metadata/     # 数据来源、版本、规模、许可及审计信息
```

原始数据、标签和运行产物不随仓库共享。仓库只保留数据使用说明和 `metadata/datasets.json`。运行代码使用上述仓库相对路径，不依赖个人机器上的绝对路径。

## 当前数据集

| 数据集 | 本地路径 | 样本规模 | 当前用途 |
|---|---|---:|---|
| AgenTracer coding test | `data/raw/agentracer/tracertraj-code-test.parquet` | 127 条 | 按 A/B/C 证据强度分层的可行性评估；test 子集，不用于训练或调参 |
| RootSE | `data/raw/rootse/data/` | 102 条、5,268 步 | 人工标注的最早决定性错误步骤评估，支持依赖图与第三通道实验 |

详细来源、版本、校验信息和许可记录见 [`metadata/datasets.json`](metadata/datasets.json)。

### AgenTracer

当前文件来自 AgenTracer coding test 子集，字段包含任务、轨迹历史和根因标签。轨迹证据审计分层为：

| 分层 | 数量 | 说明 |
|---|---:|---|
| A | 9 | 存在明确的硬失败回传 |
| B | 31 | 有 agent 对失败的文字陈述，作为弱证据单独报告 |
| C | 87 | 没有可用的负向观察，作为 trace-only 对照 |

由于它是 test 子集，不能用标签选择权重、规则阈值或角色先验。排序器只接收适配后的 `inputs.jsonl`；根因步骤、agent 和原因写入独立的 `labels.jsonl`，只由评估阶段读取。运行 `scripts/run_experiment.py` 会从原始 parquet 生成输入和标签、审计证据、运行排名并生成指标和案例报告。

### RootSE

RootSE 是公开发布的 repository-level coding-agent 轨迹集，包含 102 条人工标注轨迹，合计 5,268 步。原始数据由 agent、模型和实例目录下的 JSON 文件组成，另有数据集 README 与 `manifest.json`。适配器从 `original_traj` 读取步骤，并将 `failure_id`、`failure_reason` 保存在标签对象中，与排序输入隔离。

RootSE 的依赖图包含确定性的文件状态依赖和受约束的 LLM 依赖提议；提议须通过证据校验后才能成为图边。完整实验还需配置模型 API。运行入口为 `scripts/run_rootse_experiment.py`，排名、状态和请求缓存写入被忽略的 `outputs/rootse/`。

## 数据隔离与许可

- 原始数据只读保存，不把本地数据副本、处理结果、标签或缓存强制加入版本库。
- 排序输入和评估标签分开存储。排序阶段不得读取 `ground_truth`、`mistake_step`、`mistake_agent`、`mistake_reason`、RootSE 的 `failure_id` 或 `failure_reason`。
- 失败观察只能来自轨迹中记录的工具输出、异常、测试结果等负向信息；不得从根因标签反向生成证据。
- AgenTracer 当前文件名含 `test`，报告时须注明其 test 子集属性。
- RootSE 上游数据集 README 声明 MIT；引用或分发前仍应核对上游版本及其说明。
- `.gitignore` 排除原始数据、处理结果和本地运行输出。共享代码仓库不作为数据集分发渠道。
