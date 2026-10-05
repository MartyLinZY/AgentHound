# LogHound-Agent-Debug

LogHound-Agent-Debug 对 coding-agent 的失败执行轨迹进行步骤级根因排序：给定任务和一条已保存的轨迹，为每个执行步骤计算可疑度并输出排名。项目关注离线分析和可审计证据，不在排序时读取根因标签。

## 当前实现

- **三路静态排序**：任务/失败观察与步骤文本的 TF-IDF 相关性、可解释的失败线索分数，以及基于轨迹依赖图的全图 PageRank 结构分数。
- **AgenTracer**：适配本地 127 条 coding test 轨迹，并按证据强度分为 A 硬失败（9 条）、B 弱失败陈述（31 条）和 C 无失败证据（87 条）。这是 test 子集；不能据此训练参数或规则。
- **RootSE**：适配 102 条、共 5,268 个步骤的 repository-level 轨迹。依赖图结合可确定的文件状态依赖和受约束的 LLM 依赖抽取；LLM 提议只在验证证据后进入图。当前完整本地运行使用 semantic/clue/path 权重 `0.50/0.05/0.45`，Top-1 为 `10/102`，Top-3 为 `21/102`。这是一组数据集上的实验结果，不代表跨数据集性能保证。

## 环境与安装

项目要求 Python 3.11 或更高版本。仓库使用 `pyproject.toml` 声明核心依赖；`requirements.txt` 另列数据集读取和测试所需依赖。按项目约定，以下命令在 Conda `mybook` 环境中运行：

```bash
conda run -n mybook python -m pip install -r requirements.txt
```

## 数据

数据文件不随仓库提交。请按 [`data/README.md`](data/README.md) 放置数据，并在使用前核对各数据集的来源与许可。

默认路径：

```text
data/raw/agentracer/tracertraj-code-test.parquet
data/raw/rootse/                      # RootSE 原始轨迹与标签
```

处理结果写入 `data/processed/`；实验产物写入 `outputs/`。这些本地数据和产物均由 `.gitignore` 排除。输入和标签分别保存为 `inputs.jsonl` 与 `labels.jsonl`；**排序阶段只读取输入，评估阶段才读取标签**。不要将数据集、标签、实验缓存、`.env` 或服务密钥提交到共享仓库。

## 运行 AgenTracer 实验

一条命令完成数据转换、证据审计、排序、指标计算和案例报告：

```bash
conda run -n mybook python scripts/run_experiment.py
```

若要单独检查预处理数据或证据分层，可分别运行：

```bash
conda run -n mybook python scripts/prepare_data.py
conda run -n mybook python scripts/audit_data.py
```

`run_experiment.py` 本身会从原始数据开始完成整套流程；如需在已有排名上单独重算评估或案例报告：

```bash
conda run -n mybook python scripts/evaluate.py
conda run -n mybook python scripts/case_report.py
```

## 运行 RootSE 实验

先将 RootSE 数据放在 `data/raw/rootse/`。完整流程需要设置 `API_KEY` 和 `MODEL_NAME`；可选 `API_ENDPOINT` 用于指定兼容接口。凭据应保存在本地环境或被 Git 忽略的 `.env` 中，不要提交。

```bash
conda run -n mybook python scripts/run_rootse_experiment.py --workers 1
```

只使用已有缓存、不发送 API 请求：

```bash
conda run -n mybook python scripts/run_rootse_experiment.py --cache-only
```

运行状态、排名和缓存默认写入 `outputs/rootse/`。若任务未全部完成，状态文件会标明未完成轨迹；不应把不完整运行当作最终评测结果。

## 项目结构

```text
configs/                       排序参数与线索规则
data/                          数据说明、元数据和本地数据目录
scripts/                       数据准备、实验、评估和报告入口
src/loghound_agent_debug/      schema、适配器、证据、图与排序器
tests/                         单元与数据隔离测试
outputs/                       本地实验结果（不提交）
```
