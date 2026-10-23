# LogHound-Agent-Debug

LogHound-Agent-Debug 对 coding-agent 的失败执行轨迹进行步骤级根因排序：给定任务和一条已保存的轨迹，为每个执行步骤计算可疑度并输出排名。项目关注离线分析和可审计证据，不在排序时读取根因标签。

## 当前版本：PPT 对照后的代码清理

本目录是 `codex/ppt-code-clean` 分支的开发版本；冻结实验仍在相邻 `AgentHound` 目录，原始预测、图和推理源码未被覆盖。本轮修复状态操作解析、RootSE 图入口和离线审计脚本，不运行新模型推理。

- 三通道排序：TF-IDF 语义、规则失败线索、反向 PageRank，权重为 `0.50/0.05/0.45`。
- RootSE 必须显式提供证据依赖图；不会因漏传图而回退到通用的同 agent/时间顺序边。
- 规则状态追踪：支持结构化及字符串编辑器调用，复制/移动、局部修改/追加与覆盖分开处理；未知脚本或不确定复合命令会使旧状态失效。它仍是有限的观察推断，不是仓库内容快照或完整 shell 解释器。
- 锚点回溯：规则锚点、外部测试实体检索、personalized PageRank、Top-5 支持路径。没有 LLM 锚点裁判、已验证恢复或反事实重放。
- 完整对照与修复记录：[PPT_CODE_AUDIT.md](PPT_CODE_AUDIT.md)。

### 实验数字的来源

9.29 PPT 的 `15/102、26/102` 属于历史汇报，缺少对应原始产物，尚未复现。2026-10-06 冻结全量实验中新锚点回溯为 `10/102、21/102`，旧同图全局为 `9/102、28/102`；这些数字属于清理前的冻结协议，**不是本清理版本的新成绩**。本轮只验证固定原图上的排名行为及状态解析正确性，不以结果调权重。

## 环境与安装

项目要求 Python 3.11 或更高版本。仓库使用 `pyproject.toml` 声明核心依赖；`requirements.txt` 另列数据集读取和测试所需依赖。可以使用独立 Python 环境；下文保留已有 Conda 入口供安装了 `mybook` 的环境使用：

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

## RootSE 离线迁移实验

无 API 的四组消融、两种输入协议：

```bash
python scripts/run_rootse_offline.py --root data/raw/rootse --out-dir outputs/rootse-anchor-run1
```

`--root` 同时支持包含 `data/` 的 Hugging Face 下载目录，以及官方 TrajAudit 的 `RootSE/` 目录，两种布局生成相同轨迹 ID。输出目录须为空，避免覆盖旧结果。

默认只用规则状态图。使用原始入口生成的已验证 LLM 缓存时，加 `--graph-source cache --cache-dir outputs/rootse/llm_cache`，模型配置须与原缓存一致。缓存模式也不发送 API 请求；缓存缺失会标记运行未完成并终止，不退化成部分集合评估。原始构图费用不包含在本次缓存读取耗时中。

若本机没有 `mybook`，可在独立 Python 3.11+ 环境安装：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . pytest
.venv/bin/python -m pytest -q
```

## 项目结构

```text
configs/                       排序参数与线索规则
data/                          数据说明、元数据和本地数据目录
scripts/                       数据准备、实验、评估和报告入口
src/loghound_agent_debug/      schema、适配器、证据、图与排序器
tests/                         单元与数据隔离测试
outputs/                       本地实验结果（不提交）
```

## 复现本轮代码审计（无 API）

已有冻结实验位于相邻 `AgentHound` 时：

```bash
bash scripts/reproduce_ppt_code_audit.sh
```

入口安装锁定依赖并运行测试，再扫描全102条/5268步骤，核对816组固定图排名及原始state/排名哈希。产物写入 `outputs/ppt-code-audit/`。本入口不读取密钥、不发送模型请求、不覆盖冻结产物。

新构图实验默认目录改为 `rootse-migration-state-v2-deepseek-v41-flash-32k`，使用前需另行冻结并运行；不得把旧结果直接归给修复后的规则图。
