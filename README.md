# LogHound-Agent-Debug

LogHound-Agent-Debug 对 coding-agent 的失败执行轨迹进行步骤级根因排序：给定任务和一条已保存的轨迹，为每个执行步骤计算可疑度并输出排名。项目关注离线分析和可审计证据，不在排序时读取根因标签。

## 当前实现

- **三路静态排序**：RootSE 使用语义相关性、决策时点错误（misstep）和依赖图结构分数；AgenTracer 保留原线索分数。RootSE misstep 通过决策时点引文校验后替换 clue 槽位。
- **AgenTracer**：适配本地 127 条 coding test 轨迹，并按证据强度分为 A 硬失败（9 条）、B 弱失败陈述（31 条）和 C 无失败证据（87 条）。这是 test 子集；不能据此训练参数或规则。
- **RootSE**：适配 102 条、共 5,268 个步骤的 repository-level 轨迹。第三通道依赖缓存已重建并覆盖全量轨迹，且与第二通道 misstep 缓存分开保存。第二通道在每条轨迹最多 100 次请求后，91/102 条完成排名；另 11 条触及预算上限。在三路结果共有的 91 条轨迹上，misstep 融合 Top-1/Top-3/MRR 为 `7/91`、`29/91`、`0.2459`，旧线索融合为 `12/91`、`25/91`、`0.2581`。这是未完成全量的探索性比较，不代表跨数据集性能保证。

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

只使用已有的两通道缓存、不发送 API 请求：

```bash
conda run -n mybook python scripts/run_rootse_experiment.py --cache-only --misstep-cache-only
```

第三通道依赖缓存与第二通道 misstep 缓存分别保存在 `outputs/rootse/dependency_cache/` 和 `outputs/rootse/misstep_cache/`。首次运行或缓存被删除后，需配置模型 API 并重新生成缺失缓存；`--skip-misstep` 用于运行原 clue 融合消融。

若运行中断，可添加 `--resume` 复用已有排名并只处理尚无排名的轨迹；`--resume-failures` 仅重试仍有 misstep 请求预算的失败轨迹。

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
