# src/loghound_agent_debug 源码模块详解

> 本文档对 `src/` 下每个模块/单文件的职责、核心数据结构与实现逻辑逐一说明。
> 项目概览：LogHound-Agent-Debug 是一个**基于静态证据的 agent 调试根因排序系统**——
> 输入一条 agent（如 coding agent）失败轨迹，自动识别失败症状，沿"依赖图"反向传播，
> 融合语义相似度，最终对轨迹中每一步"是根因的疑似程度"做排序（Hit@K / MRR 评估）。

---

## 整体架构与数据流

```
parquet 原始导出 (AgenTracer)
   │  adapters/agentracer.py  (schema 适配: 轨迹+标签物理分离)
   ▼
inputs.jsonl (无标签)          labels.jsonl (仅标注, 物理分离)
   │  cli.py / rankers/loghound_static.py
   ▼
evidence.py  ──> 特征features/ ──> graph.py ──> ranker 融合排序
   │               (semantic)      (PPR)
   ▼
outputs/rankings/rankings.jsonl
   │  evaluate.py (scripts/) / reporting.py (case 报告)
   ▼
Hit@K、MRR、A/B/C 分档汇总、A档 case 卡片
```

**关键设计原则**（贯穿所有模块）：
1. **标签与推理物理分离**：`TrajectoryInput` 中故意不包含任何根因标注，ranker 不可能"偷看" ground truth。
2. **轨迹内局部计算**：IDF 归一化、minmax 归一化都只在单条轨迹内做，避免跨样本统计泄漏到主协议。
3. **可审计性**：每一步输出都带 `source_step_id`、`evidence` 说明、config 哈希，保证结果可复现可追查。

---

## 1. `schema.py` — 规范化的数据模型

**职责**：定义全项目的"规范对象"（canonical objects）与序列化/反序列化，是所有模块之间的统一接口。

### 核心 dataclass（全部 `frozen`，不可变）

| 类 | 字段 | 含义 |
|---|---|---|
| `FailureObservation` | `source_step_id`, `kind`, `content`, `provenance` | 一条面向 ranker 可见的失败观察（如 traceback 文本、agent 自述失败） |
| `Step` | `ordinal`（轨迹内序号）, `source_step_id`, `source_event_ids`, `agent`, `role`, `raw_content`, `decision_text`, `action_text`, `observation_text`, `tool_name`, `artifacts_read/written` | 规范化的"候选单元"：一次 agent 决策及其附带效果 |
| `TrajectoryInput` | `schema_version`, `dataset`, `trajectory_id`, `task`, `domain`, `outcome`, `failure_observations`, `steps` | **无标签**的轨迹表示，是 ranker/特征模块的唯一输入 |
| `Label` | `trajectory_id`, `root_cause_steps`, `root_cause_agents`, `root_cause_modules`, `root_cause_types`, `reason`, `annotation_method`, `provenance` | 仅标注信息，只供评估/报告使用 |

### 实现逻辑要点
- **`_asdict` 递归序列化**：把 dataclass 递归转成 JSON 友好的 dict（处理嵌套 dataclass / list / dict，`list[dict]` 默认的 `asdict` 不会递归展开嵌套的 dataclass，这里做了手动递归）。
- **`input_from_dict`**：反序列化时对可选元数据给默认值（`schema_version="0.1.0"`、`domain="coding"`、`outcome="failure"`），保证旧数据也能加载。
- **`label_from_dict`**：把 root cause 的 step/agent 等 ID 统一 `str()` 归一化，防止类型不一致。
- **关键设计**：`Label` 与 `TrajectoryInput` 是两个独立的类，前者绝不混入后者，这是"物理分离"在类型层面的落地。

---

## 2. `adapters/agentracer.py` — 数据集适配器

**职责**：把本地 AgenTracer coding-test 的 parquet 导出转换成内部 schema，**同时产出**无标签输入和物理分离的标签。

### 顶层 API
- `load_agentracer(path)` → `(list[TrajectoryInput], list[Label])`：读 parquet、校验必需列、逐行转换、**保持行序对齐**（inputs 与 labels 的 JSONL 一一对应）。

### 实现逻辑（按处理管线）

1. **`_plain`**：标量值转文本；`None → ""`、`bytes → utf-8`，避免把 null 字面量当真实内容。
2. **`_parse_history`**：解析 `history` 列——支持 NumPy 数组/列表/Python 字符串化列表（`ast.literal_eval`，安全不执行代码）/严格 JSON，统一成 `list[dict]`。
3. **`_json_commands` / `_command_text`**：从被 ```` ```json ```` 围栏包住的命令结构中抽取结构化命令，摊平成确定性的 action 文本（`name args` 用 `" ; "` 连接；dict/list 参数用 `sort_keys` 序列化保证哈希/测试可复现）。
4. **`_observation_text`**：**只**抽取 `Terminal output:` 显式区块内的命令/输出对，排除任务描述与提示词文本——防止把 prompt 误当成观察到的失败。
5. **`_decision_text`**：去除命令/输出脚手架（代码块、Terminal 段、`thinking:` 前缀、编排脚手架标题如 `# Past Experience`、`<experience>` 标签），保留可审计的推理文本。
6. **`_normalise_solution_filename`**：把注入的 `*_attacked_solution.py` 文件名掩码成 `<SOLUTION_FILE>`——否则文件名本身会成为"词法捷径"（作弊特征）。
7. **`_artifacts`**：用路径正则从文本中抽取文件类产物，并做掩码、去重。
8. **`parse_history_entry`**：一条原始 history 记录 → 一个 `Step`。关键点：
   - `artifacts_read` 只有 action 文本含显式读写动词（read/open/cat/grep…）才赋值，写另有 `write/edit/create/|>` 等动词——**启发式**，避免构件边过载；
   - `source_step_id` 保留原始 `step` 字段（可能是非连续整数/字符串），`ordinal` 只是内部图索引。
9. **`_row_to_input`**：扫描失败观察——优先结构化观察里的负向正则（Traceback、AssertionError…），否则用决策文本里的"忏悔"正则（`previous attempt ... failed` 等）生成 `FailureObservation`。最后组装 `TrajectoryInput` 和 `Label`。
   - 注意：`Label` 字段（`mistake_step/mistake_agent/mistake_reason`）**在这里被特意创建，但只写进 labels.jsonl，绝不嵌入 TrajectoryInput**。

---

## 3. `evidence.py` — 证据感知的失败/线索提取

**职责**：从轨迹中识别"失败证据"并按证据强弱给轨迹打 **A/B/C 档**；产出 ranker 用的线索分数、以及用于图传播的去重"症状种子"。

（已在对话中详细讲解，此处为完整版摘要）

### 数据结构
- `ClueHit`：`step_ordinal`、`rule_names`、`category`、`strength`、`signature`（聚类签名）。
- `EvidenceResult`：`tier`、`status`、`clue_scores`（每步线索强度向量）、`hits`（明细）、`symptom_seeds`、`failed_occurrences` / `failed_clusters`。

### 实现逻辑
1. **`load_clue_rules`**：YAML 声明式规则（`name/category/weight/fields/patterns`），加载时预编译正则存到 `compiled`。
2. **`_rule_matches`**：只扫描规则 `fields` 显式指定的步骤字段，拼接后做大小写不敏感（`re.I|re.S`）正则匹配，**限制扫描范围防止任务指令被误判**。
3. **`extract_evidence`**：
   - 每步取命中规则中 **weight 最大**的作为强度（防长报错叠加放大）；
   - `_error_signature` 粗粒度归一化错误（小写、数字→`<N>`、提取错误类型），SHA1 哈希得聚类 key；
   - 同签名（同类反复失败）**只保留一个最强/最早种子**，用于下游图传播"一种症状只传播一次"。
4. **`_tier`**：出现 `terminal_failure` / `tool_env_failure` → **A**；`negative_confession` → **B**；否则 **C**（`status="no_evidence"`）。**A/B 是"有硬失败证据"，C 是"无证据"**。
5. 计数区分：`failed_occurrences`（原始命中）vs `failed_clusters`（独立症状数）。

---

## 4. `graph.py` — 轨迹依赖图与反向个性化 PageRank（PPR）

**职责**：把轨迹建成有向依赖图（节点=step 序号），然后从"症状种子"出发沿**反向边**跑个性化 PageRank，得到"最可能是失败源头"的图上分数——这是"路径通道"（path channel）的核心。

### 数据结构
- `Edge`：`source`（先序）、`target`（后序）、`weight`、`relation`。
- `Graph`：`nodes` + `edges`。

### 1) `build_graph` — 建边策略（保守、可解释的四类边，按权重区分）

边方向含义：`source` *影响* `target`（source 在 target 之前发生）。

| relation | 权重 | 建边规则 |
|---|---|---|
| `artifact` | 1.0 | 某步**写**了构件 A，之后任何步**读/写** A → 前者指向后者（先索引所有 writer，再连后面的读写者） |
| `message` | 0.8 | 决策文本中出现 `to <agent名>`（正则 `(?:\bto\s+...`），且之后有该 agent 的步 → **保守**：只连显式具名收件人的第一处 |
| `same_agent` | 0.5 | 同一 agent 的相邻（沿回看的最近同 agent）步骤 |
| `temporal` | 0.2 | 相邻 ordinal 之间（`i → i+1`），保证无其它关系的步也有时序上下文；更强的语义边会**覆盖**它（`_add` 保留同起点终点中权重最大的） |

- `_add`：去 self-loop、去重（同源同目标只保留最强关系）。

### 2) `ppr_from_source_ids` — 反向 PPR 迭代

1. 种子向量：`symptom_seeds`（含 `source_step_id` + `strength`）通过 `source_step_id → ordinal` 映射表落到概率向量，**归一化**使不同轨迹的种子强度可比；
2. 建**反向图**（边 `target → source`），悬空节点（无入边者）加 self-loop 保证概率守恒；
3. 迭代式反向传播：`next_ranks[p] = (1-α)·seed[p] + α·Σ_{n→p} ranks[n]·w/normalizer`，即"每个节点的概率按出边权重分配回其前驱"——**往源头方向扩散**；
4. 以 L1 距离 < tolerance 判定收敛，返回 `(ranks, converged, iterations)`。

- `reverse_pagerank` 是同逻辑的变体（调用方已持有 ordinal 种子），两者共用同一套迭代核心。

---

## 5. `features/semantic.py` — 语义通道（确定性 TF-IDF）

**职责**：无模型、无跨轨迹状态的轻量语义打分——每个 step 与"任务意图 + 失败文本"的相似度。

### 实现逻辑
1. **`semantic_document`**：拼装每步文档 = `agent + role + decision + action + observation + artifacts`。特别注意：
   - 编排脚手架会整段重复任务内容（`Question: ...`、`Task ID:`），**从 decision 中剔除**，防止"任务文本本身"成为文档级捷径。
2. **`_tokens`**：小写分词（`<SOLUTION_FILE>` 作为词元保留）、去停止词表（the/and/for/…通用流程词）。
3. **`scores`**：
   - 对每条轨迹**内部**构造 IDF（`log((n+1)/(df+1))+1`）——刻意避免跨样本统计泄漏；
   - 两个查询分量分别算 cosine：`task_score`（vs 任务文本）与 `failure_score`（vs 失败文本）；
   - 最终 `task_weight·task + (1-task_weight)·failure`（默认 `query_task_weight=0.4`）。理由：一步可能"与任务相关"但"与错误不相关"，两分量需独立加权。
4. **`minmax`**：单轨迹内 min-max 归一化（常量通道归 0），供 ranker 使用。

---

## 6. `features/specificity.py` — 模板特异性通道（主协议中**默认关闭**）

**职责**：跨轨迹统计"步骤模板"的特异性（越罕见越可疑），但设计上**主协议不启用**（`mode="singleton"` 时返回中性值 1.0），以免跨样本转导统计污染主协议的可审计性。

### 实现逻辑
- **`template`**：把步骤（role/agent/tool/action）小写化并掩码：URL/路径→`<path>`、长 hex→`<id>`、数字→`<n>`，去停止词，生成稳定模板。
- **`corpus_statistics`**：诊断用——统计模板总数、单例模板比例、特异性（IDF）均值/方差。
- **`scores_for`**：`mode=="singleton"` 或无数集时返回 `[1.0]*n`（中性值，不改变主协议分数）；否则返回每步 `log((n+1)/(df+1))`。

---

## 7. `rankers/loghound_static.py` — 三通道注册排序器（项目核心）

**职责**：汇集语义、线索、路径三通道做加权融合，输出每步的嫌疑度排序，并附带完整审计信息。`cli.py` 调用的正是这里的 `rank_file`。

### 整体打分公式

```
score_raw(step) = semantic_w · minmax(语义分数)
                + clue_w     · clue_scores（保持规则固定量纲, 不再归一化）
                + path_w     · minmax(PPR分数 × specificity)
suspiciousness = score_raw / denominator          (denominator 默认 = 三权之和 2.12)
```

### 实现逻辑（`rank_trajectory`）
1. **证据**：`extract_evidence` → `EvidenceResult`（tier/status/clue_scores/symptom_seeds）。
2. **语义**：以 `failure_observations` 拼接文本为查询，算每步语义分数，minmax 归一化。
3. **图**：`build_graph` + `ppr_from_source_ids`（源 = 去重后的 symptom seeds）。
4. **特异性**：默认 `[1.0]*n`（乘性作用于 PPR 分数，即默认路径通道纯粹是轨迹局部的）。
5. **融合**：从 config 读权重（默认 `semantic=0.12, clue=1.0, path=1.0`）。
6. **排序**：按 `score_raw` 降序，**平票时 ordinal 小的优先** → 决定性的稳定复现。
7. **审计信息**：
   - `evidence` 列表由 `_evidence_for_step` 生成（terminal/tool/confession 命中、读改写复用构件、失败证据上游等可解释信号）；
   - `components` 拆分三通道归一的分数；
   - `graph_stats`（边数、按 relation 统计、PPR 收敛情况、失败出现/聚类数）；
   - **`config_hash` / `input_hash`**（SHA256 of 规范化 JSON）——每个输出记录精确的配置与输入版本。

### `rank_file`
读 JSONL input → 逐条 `rank_trajectory` → 写 `rankings.jsonl`（sort_keys 保证确定性）。`input_from_dict` 延迟 import，使直接调用 `rank_trajectory` 的库用户不背 JSON 适配器依赖。

---

## 8. `evaluate.py` — 步骤级指标与证据档汇总

**职责**：把 rankings（预测）对 labels（真值）计算标准检索指标。`scripts/evaluate.py` 调用的核心。

### 实现逻辑
- **`evaluate_rows`（公共 API）**：对每个 trajectory——
  - 建 `source_step_id → rank` 映射，取所有 root cause 步骤的排名；
  - `first_root_rank` = 根因步骤的最小排名（未命中则 `步骤总数+1`）；
  - 指标：`hit_at_1/3/5`（布尔转 int）、`mrr = 1/first_root_rank`；
  - 返回 `summarize_rows` 的汇总（而不是明细列表）。
- **`summarize_rows`**：分组求 **Hit@1 / Hit@3 / Hit@5 / MRR / Mean Rank / Median Rank**，内部再按轨迹长度分桶（`T<=10` / `10<T<=30` / `T>30`，递归且不重复加 by_length）。
- **`rows_from_results`**：为报告层提供的明细行版本（同样的计算路径，把列表字段 `","` 连接便于写 CSV）。

---

## 9. `reporting.py` — 案例卡片与可审计错误分类

**职责**：给 A 档（硬证据）轨迹生成可解释的 case 卡片，统计"简单、可审计"的失败类别，渲染 Markdown 报告。被 `scripts/case_report.py` 使用。

### 实现逻辑
- **`nearest_seed_distance`**：建图后把影响边**当作无向**，从每个 root 做 **BFS**，求"根因 ↔ 症状种子"的最短图距离（多个 root/seed 取最小）。用于诊断"根因在结构上是否与症状相连"（而非传播）。
- **`make_case_cards`**：只处理 `evidence_tier == "A"` 的轨迹（B/C 保留在聚合指标里，不进人工 case 审查协议）；每卡含 top-3、根因排名/构件/证据明细、最近种子距离、`symptom_seeds`。
- **`error_categories`**：三个可审计错误计数——根因无排名证据（root_evidence 为空）、根因与种子图无路径（距离为 None）、根因排在 top-5 之外。
- **`render_case_report`**：渲染成 Markdown（`# AgenTracer A-tier case report` + 每 case 一节 + 错误类别统计）。

---

## 10. `cli.py` — 排名命令行入口

**职责**：对已经 prepare 好的 inputs JSONL 跑批量排序（`python -m loghound_agent_debug.cli` 或 console script）。

```bash
python src/loghound_agent_debug/cli.py inputs.jsonl \
    --config configs/default.yaml --rules configs/clue_rules.yaml \
    --output outputs/rankings/rankings.jsonl
```

实现极薄：解析参数 → 委托 `rank_file` → 打印统计。核心算法都在库层，保证可测试性。

---

## 11. 配置文件（`configs/`）与包入口

### `configs/clue_rules.yaml` — 线索规则（evidence 通道的参数）
| 规则 | 类别 | 权重 | 扫描字段 | 示例模式 |
|---|---|---|---|---|
| terminal_failure | terminal_failure | 1.0 | observation_text | Traceback / AssertionError / NameError / IndentationError / `FAILED (errors=` |
| tool_environment_failure | tool_env_failure | 0.6 | observation_text | command not found / No such file / Permission denied / ModuleNotFoundError |
| negative_confession | negative_confession | 0.4 | decision_text | `previous attempt.{0,120}failed` / `failed due to ... error` / `test cases ... failed` |
| weak_termination | weak_termination | 0.3 | observation_text | （patterns 空，预留） |

注意 A/B 档判定只认前两类 + confession，`weak_termination` 不参与分档、也不成为传播种子。

### `configs/default.yaml` — 排序配置
```yaml
semantic.query_task_weight: 0.4   # 语义通道中任务 vs 失败的权重
graph.alpha: 0.85                 # PPR 阻尼因子
graph.tolerance / max_iterations: 1e-6 / 1000
fusion.semantic / clue / path: 0.12 / 1.0 / 1.0   # 三通道融合权重
fusion.denominator: 2.12          # 归一化分母（= 三权之和）
specificity.mode: "singleton"     # 特异性通道默认关闭
```

### 包入口
- `loghound_agent_debug/__init__.py`：版本号 `0.1.0`。
- `features/__init__.py`、`rankers/__init__.py`、`adapters/__init__.py`：仅命名空间说明，无逻辑。

---

## 附：模块间依赖一览

```
cli.py ──────────────────> rankers/loghound_static.py
                             ├──> evidence.py (线索/分档/种子)
                             ├──> features/semantic.py (语义+minmax)
                             ├──> features/specificity.py (默认中性)
                             ├──> graph.py (建图+反向PPR)
                             └──> schema.py (input_to_dict/input_from_dict)
adapters/agentracer.py ──> schema.py
evaluate.py ─────────────> schema.py (Label)
reporting.py ────────────> graph.py, schema.py
```

**一句话总结**：`schema.py` 定义规范对象；`adapters/` 把原始数据对齐到 schema（标签物理分离）；`evidence.py` 找失败硬证据并分级；`graph.py` 建模依赖并沿反向 PPR 往源头扩散；`features/` 提供语义/特异性通道；`rankers/loghound_static.py` 三通道融合出嫌疑排序；`evaluate.py` 与 `reporting.py` 分别在文档里评测、在人工审查里解释结果；`cli.py` 只是批量运行的入口。