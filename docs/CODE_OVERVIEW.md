# scripts、src 与 tests 代码说明

本文档介绍项目中 `scripts/`、`src/` 和 `tests/` 三个目录的职责、主要文件以及它们之间的调用关系。项目的目标是：输入一条已经失败的 coding-agent 执行轨迹，对其中的 decision step 进行根因可疑度排序。

## 1. 总体结构

```text
原始 AgenTracer Parquet
          │
          ▼
scripts/prepare_data.py
          │
          ├── data/processed/agentracer/inputs.jsonl
          └── data/processed/agentracer/labels.jsonl
                    │
                    ▼
         src/loghound_agent_debug/
          │  适配、证据抽取、图传播、排序
          ▼
outputs/rankings/rankings.jsonl
          │
          ├── scripts/evaluate.py     → outputs/metrics/
          └── scripts/case_report.py  → outputs/cases/
```

核心排序器只读取 `inputs.jsonl`，不读取根因标签；标签只在评估和案例报告阶段加载。这样可以避免排序过程意外使用 `mistake_step`、`ground_truth` 等真实推理时不可获得的信息。

## 2. `scripts/`：实验与数据处理入口

`scripts/` 中的文件是面向实验运行的薄封装。它们负责读取数据、调用 `src/loghound_agent_debug` 中的库函数，并将结果写入 JSONL、JSON、CSV 或 Markdown 文件；算法逻辑主要位于 `src/`。

| 文件 | 作用 | 默认输入 | 主要输出 |
|---|---|---|---|
| `prepare_data.py` | 将 AgenTracer Parquet 转换为统一输入和独立标签 | `data/raw/agentracer/tracertraj-code-test.parquet` | `data/processed/agentracer/inputs.jsonl`、`labels.jsonl` |
| `audit_data.py` | 检查输入/标签数量、轨迹 ID、step ID、根因 step 映射，并划分证据等级 | 处理后的两个 JSONL、`configs/clue_rules.yaml` | `data/splits/evidence_split.json` |
| `run_experiment.py` | 一次性执行数据准备、证据审计、排序、评估和 A 档案例报告 | 原始 Parquet、配置和规则文件 | `data/processed/`、`data/splits/`、`outputs/` |
| `evaluate.py` | 将排序结果与独立标签对齐，计算逐轨迹和汇总指标 | `outputs/rankings/rankings.jsonl`、`labels.jsonl` | `outputs/metrics/per_trajectory.csv`、`summary.json` |
| `case_report.py` | 筛选 A-tier 样本，生成案例卡片、错误分类和 Markdown 报告 | `inputs.jsonl`、`labels.jsonl`、排序结果 | `outputs/cases/a_tier_cases.json`、`case_report.md`、`error_categories.json` |
| `view_parquet.py` | 以可读形式查看 Parquet 前 N 条记录，便于数据勘查 | 任意 Parquet 文件 | 终端输出，不写入项目文件 |

### 2.1 推荐执行顺序

只准备数据时：

```bash
conda run -n mybook python scripts/prepare_data.py
```

只做数据审计时：

```bash
conda run -n mybook python scripts/audit_data.py
```

运行完整实验：

```bash
conda run -n mybook python scripts/run_experiment.py
```

完整实验脚本默认会完成以下工作：

1. 调用 AgenTracer adapter，将原始轨迹转换为 canonical schema；
2. 将输入与标签分别写入两个 JSONL 文件；
3. 根据 clue 规则抽取证据并写入 A/B/C evidence split；
4. 对每条轨迹运行静态 LogHound 排序器；
5. 计算 Hit@1、Hit@3、Hit@5、MRR、平均排名和中位排名；
6. 为 A-tier 样本生成案例卡片和错误分类。

参数可通过 `--raw`、`--processed`、`--rankings`、`--config` 和 `--rules` 覆盖。脚本使用相对路径写入输出，因此应从项目根目录执行。

## 3. `src/loghound_agent_debug/`：核心库

`src/` 使用标准 Python `src layout`。`pyproject.toml` 将 `src` 配置为包搜索路径，并注册了命令行入口 `loghound-agent-debug`。

### 3.1 数据模型与输入边界

| 文件 | 主要对象/职责 |
|---|---|
| `schema.py` | 定义 `FailureObservation`、`Step`、`TrajectoryInput` 和 `Label`；提供 JSON 可序列化及反序列化函数 |
| `adapters/agentracer.py` | 读取 AgenTracer Parquet，解析 history、命令、Terminal output、artifact 和失败观察，生成 `TrajectoryInput` 与 `Label` |
| `cli.py` | 提供 `loghound-agent-debug <inputs>` 命令，调用排序器批量处理 JSONL |

`TrajectoryInput` 是排序器的唯一输入对象，包含任务信息、失败观察和所有执行步骤；`Label` 单独保存根因 step、agent、原因及标注来源。适配器会保留原始 `source_step_id`，并将一个 history 元素作为一个候选 decision step，不因文本清洗重新编号。

### 3.2 特征与失败证据

| 文件 | 作用 |
|---|---|
| `evidence.py` | 加载 YAML clue 规则，扫描结构化的 `decision_text`、`action_text` 和 `observation_text`，输出命中规则、证据等级以及失败聚类统计；不生成图 PageRank 种子 |
| `features/semantic.py` | 在单条轨迹内部计算任务/失败观察与每个 step 的 TF-IDF cosine 相关性，并提供 min-max 归一化 |
| `features/specificity.py` | 将 step 归一化为模板，计算跨轨迹模板特异性；当前主协议默认使用 `singleton`，即每个 step 的特异性为 1 |

证据等级含义如下：

- **A**：存在 terminal failure 或工具/环境失败的硬证据；
- **B**：存在 agent 明确陈述失败的弱证据；
- **C**：没有可用的负向证据，状态为 `no_evidence`。

重复出现的同一失败会按错误签名聚类，同时记录 `failed_occurrences` 和 `failed_clusters`；聚类仅用于诊断，不影响 PageRank。

### 3.3 图构建与根因排序

| 文件 | 作用 |
|---|---|
| `graph.py` | 根据时间、同 agent、artifact 和显式消息关系构建有向轨迹图；过滤纯时间边后，在反向依赖投票图上运行全图 PageRank |
| `rankers/loghound_static.py` | 组合 semantic、clue 和 path 三个通道，归一化并生成稳定排序结果，同时记录图统计、输入 hash 和配置 hash |

排序过程可以概括为：

```text
每条轨迹
  ├─ semantic：step 与 task/failure observation 的文本相关性
  ├─ clue：step 上的显式失败规则命中
  └─ path：反向依赖投票图上的全图 PageRank 分数
             │
             ▼
     轨迹内归一化 + 固定权重融合
             │
             ▼
       按 score 降序、ordinal 升序稳定排名
```

主配置中的默认融合权重为 semantic `0.12`、clue `1.0`、path `1.0`。semantic 和 path 做轨迹内 min-max；clue 保留规则文件中的绝对权重。没有证据时关闭 clue，但仍运行不依赖失败证据的 PageRank。

### 3.4 评估与报告

| 文件 | 作用 |
|---|---|
| `evaluate.py` | 将排名与 `Label` 对齐，计算每条轨迹的首个根因排名、Hit@K 和 MRR，并按轨迹长度汇总 |
| `reporting.py` | 生成 A-tier case cards，保留 clue 命中和 PageRank 图统计，并归类“无排名证据”“未进入 Top-5”等错误 |

评估指标包括 `Hit@1`、`Hit@3`、`Hit@5`、`MRR`、`Mean Rank` 和 `Median Rank`，并按短轨迹（`T<=10`）、中等轨迹（`10<T<=30`）和长轨迹（`T>30`）提供长度分桶结果。

## 4. `tests/`：行为与协议测试

测试使用 `pytest`，配置位于 `pyproject.toml`；测试运行时会自动将 `src` 加入 Python 路径。

| 文件 | 覆盖内容 |
|---|---|
| `test_adapters.py` | 验证命令与 Terminal output 保留在同一个 decision step；验证 `*_attacked_solution.py` 文件名归一化为 `<SOLUTION_FILE>` |
| `test_graph_direction.py` | 验证普通 PageRank 在反向依赖图中按结构重要性偏向上游依赖，且忽略纯时间边 |
| `test_scoring.py` | 验证常量通道的 min-max 结果为全零；验证无证据轨迹仍运行 PageRank，且 PageRank 不受 clue 影响 |
| `test_metrics.py` | 验证平均排名、中位排名和样本数等汇总指标 |
| `test_no_label_leakage.py` | 验证 ranker API 不接收 label，并且实现不引用 `ground_truth` 或 `mistake_step`；同时验证可在没有标签的情况下完成排序 |

运行全部测试：

```bash
conda run -n mybook pytest
```

运行单个测试文件：

```bash
conda run -n mybook pytest tests/test_no_label_leakage.py
```

## 5. 输出文件速查

| 输出 | 内容 |
|---|---|
| `data/processed/agentracer/inputs.jsonl` | 排序器可见的 canonical 输入 |
| `data/processed/agentracer/labels.jsonl` | 仅评估与报告使用的根因标签 |
| `data/splits/evidence_split.json` | 每条轨迹的 A/B/C 证据等级、clue 命中步骤和失败统计 |
| `outputs/rankings/rankings.jsonl` | 每条轨迹的完整 step 排名、分数组件、证据和图统计 |
| `outputs/metrics/summary.json` | 全体及 A/B/C 子集指标 |
| `outputs/metrics/per_trajectory.csv` | 逐轨迹评估明细 |
| `outputs/cases/case_report.md` | A-tier 案例的可读报告 |

`outputs/` 通常被 git 忽略，适合保存本地实验产物；配置和规则版本会通过结果中的 hash/version 字段保留运行可追溯性。

## 6. 修改代码时的边界

1. 新数据集应优先新增 `adapters/<dataset>.py`，转换为既有 canonical schema，不要让排序器直接依赖原始数据列名。
2. 新特征应作为独立模块接入 ranker，并在结果的 `components` 或 `graph_stats` 中保留可审计信息。
3. 评估逻辑可以读取标签，但排序器不能读取标签；新增测试时应继续保持这一边界。
4. 若改变 clue 规则、融合权重或图参数，应同步检查配置 hash、证据分档和现有行为测试。
5. 运行脚本和测试统一使用项目约定的 `mybook` Conda 环境。
