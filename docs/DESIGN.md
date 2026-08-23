# LogHound-Agent-Debug 设计方案

## 1. 研究目标与边界

### 1.1 目标

给定一条已经失败的 coding-agent 执行轨迹，对其中所有候选执行步计算可疑度并降序排列。输出不只是一个根因步，而是一份与 LogHound 方法级排名同构的 step-level 排名，使 Top-K、MRR 和排序消融实验都可直接开展。

一条轨迹记为：

$$
\tau = (q, x_0, x_1, \ldots, x_{T-1}, f)
$$

其中 $q$ 是任务，$x_t$ 是第 $t$ 个执行步，$f$ 是轨迹中可观察到的失败结果。排序器输出：

$$
\pi(\tau) = [(t_1, s_{t_1}), (t_2, s_{t_2}), \ldots, (t_T, s_{t_T})]
$$

其中 $s_t \in [0,1]$，并满足 $s_{t_1} \ge s_{t_2} \ge \cdots \ge s_{t_T}$。

### 1.2 初版的“静态”含义

初版只读取已经保存的轨迹，不执行以下操作：

- 不重放环境；
- 不替换动作做反事实实验；
- 不重新运行测试或工具；
- 不要求访问产生轨迹时的代码仓库；
- 不调用 LLM 生成根因结论。

这与 LogHound 不依赖目标程序动态复现的出发点一致。AgenTracer 的反事实/故障注入只用于生成标签，不进入待评估排序器。

静态不等于“可以凭空制造失败证据”。排序器只能使用轨迹中已经记录的负向工具结果、测试输出、异常或终止状态。如果数据只有“这条轨迹最终失败”的 episode-level 标记，却没有任何 step-level 失败观察，则该条轨迹不满足 evidence-aware 主协议的输入资格；不能用 `mistake_reason` 或 `ground_truth` 补成失败证据。

### 1.3 初版明确不做的内容

- 不训练新的大模型；
- 不直接迁移 LogHound 的 Java AST、CodeQL 或方法调用图实现；
- 不把 AgentDebug 的 ALFWorld、GAIA、WebShop 数据称作 coding-agent 数据；
- 不在 AgenTracer 的官方 test 文件上学习权重；
- 不提前为 SWE-bench Pro 大规模标注。

## 2. 为什么不是机械地“方法换成步骤”

LogHound 的三个主要信号不能逐字段照搬，但可以迁移其证据融合结构：

| LogHound | Agent Debug 迁移 | 保留的思想 |
|---|---|---|
| 缺陷报告与源码方法的 VSM 相似度 | 任务/失败证据与执行步内容的文本相关性 | 信息检索产生候选可疑度 |
| 日志与异常栈中的方法 | 错误消息、失败测试、无效工具调用、终止症状所关联的步骤 | 运行线索比普通文本更直接 |
| 静态调用图上的路径重建与 PageRank | 时间、消息、agent、文件和工具依赖构成的轨迹图；从症状反向传播 | 用结构关系把表面失败追溯到上游步骤 |
| 测试覆盖率抑制广泛执行的方法 | 跨轨迹模板频率抑制常见编排/样板步骤 | 常见但信息量低的节点不应占据榜首 |
| 多路归一化加权 | step-level 多路归一化加权 | 生成统一可疑度排名 |

最大差异是：LogHound 需要从不完整日志恢复执行路径，而 agent 轨迹已经给出实际顺序。因此这里不再“重建执行序列”，而是从序列上补建依赖边，并沿依赖边反向寻找失败的上游来源。

## 3. 数据设计

### 3.1 数据集角色

| 数据 | 初版角色 | 原因 |
|---|---|---|
| AgenTracer coding test，127 条 | 主可行性数据源；按证据分档：9 条硬失败（A 档，主验证）、31 条弱证据（B 档，稳健性）、87 条 trace-only（C 档，对照） | 与 coding 任务最接近，已有单根因 step/agent 标签；step-level 失败观察部分存在（A/B 档共 40 条），且混合了注入故障痕迹，需按 3.1.1 分档 |
| <br />                | <br />  | <br />                                    |
| SWE-bench Pro                | 第二阶段外部测试集                                                              | 最接近最终科研目标，但需要先定义并完成人工标注                                                                    |

本地 AgenTracer 完整论文规模大于当前 127 条文件，但当前工作区只有 coding test 子集。设计和实验报告必须区分"论文数据规模"与"本地实际可用规模"。

#### 3.1.1 本地 127 条的实测证据分布（P0 审计结论，写代码前必须固化）

对 `content` 全文扫描（正则见 3.3）所得，作为后续脚本与报告的基线数字：

| 档位 | 判定条件 | 条数 | 说明 |
|---|---|---|---|
| A. 硬失败证据 | terminal 回传含 Traceback / AssertionError / FAILED / 异常栈 | 9 | `Filter_56113_I`、`Package_9114_I`、`Leetcode_7390_I`、`Leetcode_9277_I`、`Algorithm_35399_I`、`Apps_6606_I`、`Codeforces_7286_I`，以及仅 `command not found` 的 `Filter_17660_I`、`Codeforces_00027102_I` |
| B. 弱陈述证据 | 无回传，但 agent 文本明确陈述"previous attempt ... failed / failed due to syntax error"等 | 31 | 高度模板化（见 4.4 风险），多为注入故障的痕迹；不能与 A 档等同 |
| C. 无证据 | 无任何负向观察，正常走完流水线后 `Plan.finish_current_task`+`end` 收尾 | 87 | trace-only 对照 |

A+B 合并为 evidence-aware 合格集（40 条），但**主协议只以 A 档为强证据样本，B 档单独报告**。这与 4.2、5.5 和 P0 完成标准联动：不能只报"40 条 evidence-aware"，必须给出 9/31/87 三档拆分。

两个已核实的混淆检查结论，写 adapter 时直接使用：

- `*_attacked_solution.py` 文件名首现步均为 step 0（68/68）：文件名在任务正文里直接给出，**不作为泄漏源**，但语义通道需把文件名规范化（见 3.3），避免文件名 token 干扰 IDF；
- 首次写 solution 文件的 step 与标签根因步吻合 0/68："写代码首步"不是根因的可靠代理，位置/角色基线与图传播仍是必要成分。

### 3.2 统一输入 schema

建议 `inputs.jsonl` 每行结构如下：

```json
{
  "schema_version": "0.1.0",
  "dataset": "agentracer",
  "trajectory_id": "Algorithm_34582_I",
  "task": "...",
  "domain": "coding",
  "outcome": "failure",
  "failure_observations": [
    {"source_step_id": "22", "kind": "test_failure", "content": "...", "provenance": "trace"}
  ],
  "steps": [
    {
      "ordinal": 0,
      "source_step_id": "0",
      "source_event_ids": ["0"],
      "agent": "Team Leader",
      "role": "assistant",
      "raw_content": "...",
      "decision_text": "...",
      "action_text": null,
      "observation_text": null,
      "tool_name": null,
      "artifacts_read": [],
      "artifacts_written": []
    }
  ]
}
```

标签放在独立的 `labels.jsonl`：

```json
{
  "trajectory_id": "Algorithm_34582_I",
  "root_cause_steps": ["3"],
  "root_cause_agents": ["Product Manager"],
  "root_cause_modules": [],
  "root_cause_types": [],
  "reason": "...",
  "annotation_method": "fault_injection",
  "provenance": "agentracer"
}
```

`root_cause_steps` 从一开始就设计为数组，以兼容 AgentDebug 的最小根因集合以及未来 SWE-bench Pro 可能出现的多根因情况。`failure_observations` 是轨迹输入中已经存在的观察；规则抽取后的 `symptom_evidence` 属于中间结果，不能和标签一起写入原始 `inputs.jsonl`。

### 3.3 步骤边界

适配时必须遵守两个原则：

1. 保留原始 `source_step_id`，绝不因清洗或模块抽取而重新编号；
2. 排名单元是一个 agent decision step。初版 AgenTracer 中一个 history 元素对应一个 decision step；未来数据中可把一个 assistant 决策和其工具结果合并为一个 decision step，并在 `source_event_ids` 中保留原始事件映射；
3. 一条记录中的 thinking、command、message 不再拆成多个候选，但要解析为 `decision_text`、`action_text`、`observation_text`，并保留 `raw_content` 供审计。

可以把步骤内部进一步解析为 action、observation、tool 和 artifact 字段，但这些只是特征，不改变评估单元。未来适配 SWE-bench Pro 时，再额外建立"assistant 决策 + 工具结果"的 canonical decision 映射，并保留原始 event 到 decision 的映射表。

#### 3.3.1 content 的四段结构与回传解析（实测格式）

一条 thinking 帧的 `content` 按顺序混合四段内容：系统模板行（`# Past Experience` / `# Tool State` / `# Current Plan` 等）→ 任务复述/思考 → ```json 命令数组 → `Terminal output:` 回传块。回传块格式实测为：

```text
Terminal output: [command]: pwd  
[command output] : /data/home_data/R1-Tracer/MetaGPT/workspace
```

解析规则（P0 固化）：

1. `decision_text`：模板行剥离后的思考部分（保留任务复述作为理解意图的上下文）；
2. `action_text`：json 数组中的 `command_name` + `args`；
3. `observation_text`：用正则 `\[command\]:\s*(.*?)\n\[command output\]\s*:\s*(.*?)(?=\n\[command\]|$)` 提取全部 `(命令, 输出)` 对拼接；663 处回传中 614 处为 `pwd`（无信息但真实回传），非 pwd 约 50 处；
4. 命令 JSON 与回传位于**同一个 decision step**，因此工具调用边退化为节点内 observation，不跨节点建 self-loop（与 4.5 的约定一致，此处由实测确认）；
5. 系统模板行、被引用的任务正文、`attacked` 文件名均不进 `decision_text` 的语义文档；文件名统一替换为占位符 `<SOLUTION_FILE>`，防止文件名 token 干扰 IDF（见 3.1.1）。

### 3.4 AgenTracer 已确认的风险

- `history` 是 NumPy `ndarray`，不是 JSON 字符串；适配器要同时兼容数组和字符串化列表。
- 127 条标签都能正确映射，但轨迹长度从 2 到 151，不能只报告未归一化平均排名。
- Engineer 占根因 agent 的 72.4%，step 4 占根因步的 33.1%，存在强角色和位置先验。
- `ground_truth` 和 `mistake_reason` 在真实推理时不可获得，必须物理隔离。
- 文件名为 test；未经额外训练集支持，不可据此调权。
- 本地 AgenTracer Parquet 的 last-step 多数是"完成任务"的编排消息，但**并非完全没有 step-level 失败观察**：`content` 内嵌 663 处 `Terminal output` 回传（3.3.1），其中仅 9 条轨迹含真实硬失败、31 条仅 agent 陈述式弱证据、87 条无证据（3.1.1）。因此 evidence-aware 合格集只有 9 条强样本，**不得将 40/127 说成主验证集规模**；
- 大量弱证据高度模板化（"Since the previous attempt to create the file with content failed, ..."），本质是注入故障的痕迹而非自然测试失败，若直接当 clue seed 会在错误步骤散布分数（4.4 处理）；
- 同一条失败在轨迹内可能重复数十次（如 `Filter_56113_I` 相同 `AssertionError` 出现 10 次），若按 4.5 的 seed 归一化 ∑s=1 会把种子强度稀释到多个重复步骤，需要失败去重（4.5 处理）。

## 4. 初版排序算法

### 4.1 总体流水线

```mermaid
flowchart LR
    A[失败轨迹] --> B[数据适配与步骤保真]
    B --> C[失败观察与证据抽取]
    B --> D[轨迹依赖图构建]
    B --> E[步骤文本与线索特征]
    C --> F[文本相关性分数]
    C --> G[显式失败线索分数]
    D --> H[上游图传播与步骤特异性]
    F --> I[分数归一化与融合]
    G --> I
    H --> I
    I --> J[执行步可疑度排名]
```

### 4.2 候选步骤与失败证据

候选集合是轨迹中的全部 decision steps。失败证据只从输入轨迹内部的结构化 observation 和负向终止状态提取，包括：

- 最终测试失败或评分失败；
- traceback、异常、非零退出码；
- 工具返回的 invalid、not found、timeout、permission 等错误；
- agent 明确陈述的未完成、无法继续或达到限制；
- 若没有显式错误但输入明确标记了负向终止，则使用最后一个负向 observation 作为弱终止证据；正常的"完成""提交成功"或"任务结束"消息不能作为 failure seed。

本地 AgenTracer 的 evidence 判定按 3.1.1 的档位执行，且必须满足证据来源约束：**只有 `[command output]` 回传（A 档）和 agent 陈述中的失败声明（B 档）算数；任务正文中出现的 `assert`、`Invalid input`、`avoid timeouts` 等词汇是任务说明，不是失败观察**。实现 clue 规则时必须显式剔除命中发生在系统模板行/任务复述段的情况（实现按 3.3.1 的段落切分后只扫描 observation_text）。

如果输入没有任何负向 observation，也没有明确负向终止状态，则标记为 `no_evidence`：不生成伪造的最后一步 seed，`clue` 和 `path` 通道记为无信息，并在主实验中单独报告或排除该条轨迹。这样可以区分"算法没有找到根因"和"数据没有提供失败线索"。

实验分成两个协议：

- `evidence-aware`：输入含至少一个真实负向 observation，运行 semantic、clue 和 path 三个通道，是主方法协议。主结果限定 A 档（9 条）并附逐条案例；B 档（31 条）作为弱证据稳健性子集单独报告，不并入主结果；
- `trace-only`：输入没有负向 observation（C 档 87 条），只运行 task-only semantic 通道，clue/path 记为无信息，不能把结果解释为完整方法效果；位置等基线推迟到后续阶段，当前阶段仅作为无证据输入下的对照观察单独报告。

任务文本 $q$ 用于理解意图，但正确答案 `ground_truth` 不得进入失败证据。

### 4.3 通道一：静态文本相关性

初版使用 TF-IDF cosine，避免模型调用和额外训练。查询由任务约束和失败观察组成，文档为每个步骤的 agent、`decision_text`、`action_text`、工具名和 artifact；`raw_content` 中重复的系统 prompt、Current Plan、Response Language 和被引用任务不进入语义文档。词表与 IDF 只在单条轨迹内部构建（文档集 = 该轨迹全部步骤 + 任务文本 + 失败观察），主版本不跨轨迹共享词表，避免语义通道携带跨样本信息；跨轨迹统计只发生在 4.5 的 $S_{spec}$ 通道。

单独使用“与任务相似”会偏向复述需求的早期步骤，单独使用“与错误相似”又会偏向最后的表面症状。因此保留两个子分数：

$$
S_{sem}(t) = \lambda_q \cdot cos(v_t, v_q) + (1-\lambda_q) \cdot cos(v_t, v_f)
$$

首版固定 $\lambda_q=0.4$，不在 test 上调参。权重敏感性（$0.2,0.4,0.6$）推迟到后续阶段补充；若补充，必须展示完整网格，不能只挑最好结果作为主结果。

### 4.4 通道二：显式失败线索

`clue` 模块只对结构化 `observation_text`、工具结果和负向终止状态抽取可解释规则特征，不扫描被引用的任务说明或系统 prompt。特征包括：

- 工具/命令执行失败；
- 测试失败、异常或错误码；
- 无效动作、参数或格式；
- 文件不存在、读写目标不一致；
- agent 自己承认假设错误、结果不符或无法完成；
- 纯环境/系统错误标记。

分数定义：规则表中每条规则 $r$ 记录判定逻辑（正则/字段条件）、类别和固定权重。主版本采用以下类别权重：terminal failure 为 1.0，tool/env failure 为 0.6，negative confession 为 0.4，弱终止为 0.3。对步骤 $t$，设 $R_t$ 为命中规则集合，局部线索分为命中规则权重的最大值：

AgenTracer 适配时把 3.3.1 解析出的 observation_text 作为规则扫描对象（不接受裸全文本扫描），并预注册以下本地规则族：

- terminal failure（1.0）：回传含 `Traceback` / `AssertionError` / `NameError` / `IndentationError` / `unittest` 的 `FAILED (errors=` 或 `AssertionError: Lists differ`；
- tool/env failure（0.6）：回传含 `command not found` / `No such file` / `Permission denied` / `ModuleNotFoundError`；
- negative confession（0.4）：`decision_text` 中 `previous attempt ... failed` / `failed due to (a )?(syntax|indentation|compile|runtime) error` / `test cases? ... failed`，且该句在 observation_text 之外（陈述句）；
- 显式排除（不计分）：上述词出现在任务正文复述段、系统模板行、或 agent 复述测试代码/文件中（如 `assert result == expected, f"Test failed..."` 是代码内容不是失败观察）。

B 档模板句（"Since the previous attempt to create the file with content failed, ..."）会命中 confession 规则。预注册时接受这一事实，但：a) 不得把它提升为 terminal failure；b) 在 3.1.1 的 B 档单独报告其贡献；c) 若它导致 clue 分散在非根因步，在 5.4 的错误分析中记录；当前阶段不做"去掉 confession 规则"消融。规则表应放在配置文件并为每次运行记录版本，不能在看到 test 标签后逐条补规则。

$$
S_{clue}(t) = \max_{r \in R_t} w_r
$$

未命中任何规则为 0；命中多条取最大值而非求和，避免大段报错文本因重复命中被放大。上述数值写入 `clue_rules.yaml` 并为每次运行记录版本。clue 既保留显式症状的局部证据，也通过 symptom seed 驱动上游传播；不能再用“绝对权重较小”解释其影响，因为 4.6 不对 clue 做轨迹内 min-max。

symptom seed 判定：命中 terminal、tool/env 或 negative confession 的步骤都成为 symptom seed，seed 强度就是该步骤的 $S_{clue}(t)$；若输入明确为失败但没有显式规则命中，才把最后一个负向 observation 作为 0.3 的弱 seed。若某步只命中 tool/env 类规则，它既保留局部 clue 分，也会通过反向传播把分数传给上游 action，便于处理“错误动作立即失败”和“工具错误由更早错误触发”两种情况。

规则表应放在配置文件并为每次运行记录版本，不能在看到 test 标签后逐条补规则。

### 4.5 通道三：轨迹图与上游传播

构建有向图 $G=(V,E)$，一个节点对应一个候选步骤。边从影响源指向被影响步骤，初版包括：

- 相邻时间边 $t \rightarrow t+1$；
- 同一 agent 的相邻决策边；
- 消息发送步骤到接收者后续步骤；
- 计划任务到完成该任务的步骤；
- artifact 写入步骤到后续读取/修改步骤；
- 工具调用到对应工具结果；如果结果已附着在同一个 decision step 内，则作为该节点的 observation，不额外创建候选节点。

主版本固定边权为 artifact/tool 1.0、显式消息/任务 0.8、同 agent 0.5、纯时间相邻 0.2。如果同一对节点命中多种关系，边权取这些关系权重的最大值，避免重复关系人为放大；如果某类边无法可靠抽取，则缺省，不能凭文本猜出不存在的文件依赖。

以失败 symptom 节点为个性化种子，在反向图 $G^{-1}$ 上运行 Personalized PageRank（PPR）。反向图由正向边翻转得到：$u \to v$（$u$ 影响 $v$）翻转为 $v \to u$。令 $M$ 为反向图的行随机化转移矩阵（每个节点的出边按权重归一化；无出边的悬挂节点记自环，避免分数泄漏）。种子向量 $s \in [0,1]^T$ 按 4.4 的 symptom seed 取值，即每个命中 terminal、tool/env 或 negative confession 的步骤使用其 $S_{clue}(t)$，再归一化使 $\sum_t s_t = 1$。没有任何 seed 时不运行 PPR，直接将 $S_{up}$ 记为 0 并标记 `no_evidence`。PPR 平稳分布由迭代求解：

**seed 去重（本地数据特有，必须实现）**：同一失败在轨迹内重复出现多次（如 `Filter_56113_I` 相同 `AssertionError` 出现在 step 10/20/28/... 共 10 次，`Codeforces_7286_I` 相同 unittest FAIL 出现在 step 26/58/86/... 共 6 次）。若不去重，∑s=1 归一化会把种子强度稀释到重复步骤上。去重规则：按（命令签名哈希, 结构化错误签名：异常类型+断言行内容模板）聚类，同一簇只保留**首次出现步骤**作为 seed，seed 强度取该簇最大 $S_{clue}$；去重计数写入 `graph_stats.failed_occurrences`。

$$
r^{(0)} = s, \qquad r^{(k+1)} = (1-\alpha)\,s + \alpha\,M^{\top} r^{(k)}
$$

迭代至 $\lVert r^{(k+1)} - r^{(k)} \rVert_1 < 10^{-6}$ 收敛，主版本固定 $\alpha = 0.85$，权重敏感性（$0.7/0.85/0.95$）推迟到后续阶段补充。最终定义 $S_{up}(t) := r_t$，其绝对尺度由 4.6 的轨迹内 min-max 归一化消除。这样分数从终止症状流向可能导致它的上游步骤，而不是继续流向下游表面错误。

模板化编排步骤会在大量轨迹中重复。参考 LogHound 的 coverage-aware 思路，为步骤计算跨轨迹特异性：

$$
S_{spec}(t) = \log\frac{N+1}{df(c_t)+1}
$$

其中 $c_t$ 是步骤 $t$ 的结构化模板串，$df(c_t)$ 是包含该模板串的轨迹数，$N$ 是参与统计的轨迹总数。模板串只使用 agent role、action type、tool name 和规范化后的命令签名，不使用完整任务正文或系统 prompt。生成按固定顺序执行以保证可复现：小写化 → 将文件路径、URL、hash/ID、数字替换为统一占位符 → 去标点与停用词 → 保留结构化 token 顺序拼接。两条步骤模板串相同即属同一簇，初版只做精确匹配、不做模糊聚类。主结果不使用 test 输入上的跨样本统计，令 $S_{spec}(t)=1$；使用完整无标签 test 输入计算 $df$ 的版本属于 transductive 敏感性分析。这样主结果中 $S_{path}(t)=S_{up}(t)$，而不是令 $N=1$ 使对数项退化为 0。实现必须报告 singleton 比例和 $S_{spec}$ 方差；若模板退化为大多数 singleton，则停用 specificity 并将其作为失败分析结果。

LogHound 的路径分数内部混合了 VSM、PageRank 与 coverage；初版若照搬，语义信号会在路径通道与最终融合（4.6 的 $0.12 \cdot \widehat{S}_{sem}$ 项）中被重复计权，且“去掉 semantic”消融将无法定义。因此初版把 VSM 拆出为独立的通道一，路径通道只保留 PageRank×coverage 结构，首版定义：

$$
S_{path}(t) = S_{up}(t) \cdot S_{spec}(t)
$$

这样三个通道正交：semantic（文本相关性）、clue（显式线索）、path（结构传播），各自的贡献可在后续阶段的消融中独立移除。

### 4.6 分数归一化与最终融合

semantic 和 path 在每条轨迹内部做 min-max 归一化。若一个通道的所有候选值相同，则该通道统一记为 0。clue 不做轨迹内 min-max，直接使用 4.4 中固定的 $[0,1]$ 类别权重，避免唯一一个症状命中被归一化为 1，也保留 terminal/tool/env 的预注册强弱关系。

沿用 LogHound 原型的融合比例作为不经 test 调参的起点：

$$
S(t) = 0.12 \cdot \widehat{S}_{sem}(t) + 1.0 \cdot S_{clue}(t) + 1.0 \cdot \widehat{S}_{path}(t)
$$

最终可再除以 2.12 映射到 $[0,1]$，这不改变排名。分数相同则按原始 `ordinal` 升序稳定打破平局，因为两套参考工作的根因定义都强调最早的决定性错误。

### 4.7 输出格式

```json
{
  "trajectory_id": "Algorithm_34582_I",
  "evidence_status": "ok",
  "ranking": [
    {
      "rank": 1,
      "source_step_id": "3",
      "agent": "Product Manager",
      "score_raw": 1.657,
      "suspiciousness": 0.782,
      "components": {
        "semantic_normalized": 0.31,
        "clue": 0.70,
        "path_normalized": 0.92
      },
      "evidence": ["upstream_of_test_failure", "writes_reused_artifact"]
    }
  ],
  "symptom_seeds": [{"source_step_id": "22", "strength": 1.0}],
  "graph_stats": {"nodes": 23, "edges": 41, "ppr_converged": true, "ppr_iterations": 18},
  "config_hash": "...",
  "input_hash": "..."
}
```

组件分数和证据必须保留，否则无法做错误分析，也无法判断模型是因为语义、位置偏置还是图结构命中。

## 5. 实验方案（单一迁移实验）

### 5.1 当前阶段目标

当前阶段只做一个实验，回答一个问题：

> LogHound 的证据融合方法（语义 + 显式线索 + 反向图传播的三路静态融合，见 4.3–4.6）迁移到 agent 执行轨迹后，能否把根因步骤排到足够靠前。

实验对象为本地 AgenTracer coding test 的 127 条轨迹，使用预先固定的配置和规则运行一次完整流程。当前不做：

- 随机、位置、角色或单通道基线对比；
- 各通道贡献的消融与权重敏感性；
- 跨域验证或其他额外数据集实验。

实验不在 test 上调参：$\lambda_q=0.4$、$\alpha=0.85$、4.6 的融合权重均使用预注册固定值，规则表随运行记录版本。

### 5.2 评估口径

本实验只运行当前方法一次；A/B/C 三档是同一实验中的数据分层，不构成额外实验：

- A 档 9 条强证据：作为主要结果，在汇总指标之外逐条给出真实根因、排名与证据链；
- B 档 31 条弱陈述证据：作为同一实验的补充分层结果单独报告，不与 A 档混算；
- C 档 87 条无证据轨迹：按照 4.2 的 `no_evidence` 规则运行并单独报告，用于说明无失败证据时的适用边界，不将其结果解释为完整方法效果。

所有规则表（clue_rules.yaml）与配置 hash 必须在看到 test 标签前固定（4.4、4.6）。报告只给出当前方法的绝对效果，不出现“优于 X”或通道贡献的表述。

### 5.3 指标

主指标保持不变（评估单元与 4.2 一致）：

- Hit@1、Hit@3、Hit@5；
- MRR；
- Mean Rank 与 Median Rank；
- 归一化根因排名 NR（EXAM-style normalized rank）。

对单根因轨迹，归一化排名定义为：

$$
NR = \frac{r-1}{T-1}
$$

其中 $r$ 是真实根因首次出现的 1-based 排名。$T=1$ 时记为 0。该指标越低越好，可抵消轨迹长度从 2 到 151 的差异。

多根因数据额外报告 MAP，并以第一个命中根因计算 MRR。当前排序器只预测 step，因此仅报告 step-level 指标；module/type 标签保留在 schema 中，待未来增加分类头后再报告 Step+Module 和 All-Correct，不能把仅命中 step 等价成完整根因诊断。

统计报告口径：

- A 档（9 条）：样本过小不做 bootstrap，逐条报告真实根因的排名、NR、组件分数与 evidence；
- B 档（31 条）：汇总指标并注明样本量局限；
- C 档（87 条）：单独列一节或排除，不与主结果混算；
- 按短轨迹 $T \le 10$、中轨迹 $10<T\le30$、长轨迹 $T>30$ 分层；
- 按根因 agent 和根因位置分桶，观察命中分布（只记录现象，不做偏置归因结论）；
- 对有明确 symptom seed 的轨迹，报告根因到最近 seed 的图距离；没有 seed 的轨迹不计算该项。

注意：本实验没有基线或消融对照，绝对指标只用于判断迁移后的方法能否定位根因。

### 5.4 报告内容与错误分析

输出格式与 4.7 一致（逐轨迹 ranking JSONL + 组件分数 + evidence + symptom_seeds + graph_stats + 汇总 CSV），本实验报告重点：

1. A 档 9 条逐条案例卡：真实根因、方法给出的 Top-1/Top-3、各组件归一化分数、evidence 列表、根因到最近 seed 的图距离；
2. B 档：核对 confession 模板句（3.4、4.4）是否把分数散布到非根因步，并记录为错误分析；
3. C 档：记录无证据输入下的表现，验证 4.2 的 no_evidence 门控与 4.5 的“无 seed 不运行 PPR”是否按设计生效；
4. 错误分类统计：标签争议、根因无显式证据、图边缺失、症状误判、模板偏置，作为判断迁移方案是否需要后续改进的依据。

### 5.5 完成标准与后续阶段

本实验完成标准：

1. 一条命令可从 raw 数据复现 prepare → run → evaluate 全流程，重复运行排名与指标一致；
2. A 档 9 条逐条案例表完整产出，能说明每条的真实根因命中情况与证据链；
3. 汇总指标在 A 档给出、B 档单独给出、C 档单列；
4. 人工抽查至少 30 条（A 档全查 + B/C 抽样），确认高排名证据与真正的上游错误有关，而不只是末尾错误词匹配；B 档特别核对 confession 模板句是否在错误步骤散布分数；
5. 明确记录：实验结论只描述迁移方法的绝对效果，不含基线、消融或跨域结论。

通过后再决定是否进入 SWE-bench Pro 标注。若未通过，先按错误分析判断是数据域不匹配、步骤图抽取失败，还是“静态证据不足以识别反事实根因”。

## 6. 实施结构

目录只服务于第 5 节的单一迁移实验：当前实现 AgenTracer adapter、三路融合排序和统一评估；不设置基线/消融目录，也不加入 AgentDebug 数据或阶段。SWE-bench Pro 仅作为未来可能的数据扩展预留。

```text
loghound-agent-debug/
├── README.md
├── docs/
│   ├── DESIGN.md
│   └── experiment_log.md
├── data/
│   ├── README.md
│   ├── rawdata-intro.md
│   ├── raw/{agentracer,swebench_pro}/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── metadata/
├── configs/
│   ├── default.yaml
│   └── clue_rules.yaml
├── src/loghound_agent_debug/
│   ├── schema.py
│   ├── adapters/agentracer.py
│   ├── evidence.py
│   ├── graph.py
│   ├── features/{semantic,clue,specificity}.py
│   ├── rankers/loghound_static.py  # 三路融合主方法
│   ├── evaluate.py                 # Hit@K/MRR/NR 与 A/B/C 分层结果
│   ├── reporting.py                # 逐条案例卡与错误分类（5.4）
│   └── cli.py
├── scripts/
│   ├── view_parquet.py
│   ├── prepare_data.py
│   ├── run_experiment.py
│   ├── case_report.py              # 生成 5.4 的逐条案例卡与错误统计
│   └── audit_data.py
├── tests/
│   ├── test_adapters.py
│   ├── test_no_label_leakage.py
│   ├── test_graph_direction.py
│   ├── test_scoring.py
│   └── test_metrics.py
├── outputs/                 # 运行生成，不提交大文件
│   ├── rankings/            # 逐轨迹 ranking JSONL（4.7）
│   ├── metrics/             # 汇总 CSV 与 A/B/C 分层结果
│   └── cases/               # 案例卡与错误分类（5.4）
└── pyproject.toml
```

所有 Python 命令统一使用用户指定的 `mybook` 环境：

```bash
conda run -n mybook python ...
```

## 7. 分阶段执行计划

### P0：数据与防泄漏

- 实现统一 schema 和 AgenTracer adapter；
- 生成 `inputs.jsonl` 与 `labels.jsonl`；
- 自动验证 step 唯一性、agent 对齐、缺失值和长度；
- 写 `test_no_label_leakage.py`，确保 ranker API 无法访问 label 对象；
- 审计失败观察存在率、负向 seed 覆盖率、规则误报率、各类图边覆盖率和 specificity singleton 比例；
- 固化数据校验和和运行配置。

完成标准：127 条全部通过映射检查，转换前后 step 数和内容 hash 不变；审计脚本输出 3.1.1 的 9/31/87 三档划分并写入 `data/splits/evidence_split.json`（含每条轨迹的 evidence 类型、seed step 列表、去重后的失败簇数）；验证 3.3.1 的段落切分与任务正文误报排除规则在抽查样本上无误报。若审计后 A 档仍不足 9 条，主方法验证以逐条案例报告为主，不得宣称已通过 bootstrap 级验证。

### P1：单一迁移实验实现

- 实现当前方法主流程（semantic、clue、graph、specificity 与三路融合）；
- 实现 failure observation 解析、decision_text/action_text/observation_text 抽取与 clue 规则；
- 实现轨迹图、反向 PPR、specificity 和三路融合；
- 输出逐轨迹排名 JSONL 和汇总 CSV。

完成标准：一条命令可从 raw data 生成可复现实验结果，重复运行排名一致。

### P2：单一实验评估与错误分析

- 计算 Hit@K、MRR、NR 和 A/B/C 分层结果；
- 生成逐条案例与错误分析；
- 人工抽查至少 30 条：命中、近失、完全失败各 10 条；
- 记录失败类型：标签争议、根因无显式证据、图边缺失、症状误判、模板偏置。

完成标准：能根据单一迁移实验结果对是否进入 SWE-bench Pro 标注给出明确 go/no-go 结论。

### P3：SWE-bench Pro

通过第 5.5 节完成标准后再启动。建议标注规范：

- 标注“最早的、若被正确替换可显著改变失败结局的 agent 决策步”；
- 允许多个互相独立的根因组成最小集合；
- 区分 agent 错误与纯工具/环境错误；
- 保存 root step、直接证据、因果链、可行修复和置信度；
- 两名标注者独立标注，冲突由第三人裁决；
- 先做 20 条 pilot，计算一致性并修订指南，再扩大规模。

## 8. 推荐的论文主张边界

初版若成功，最稳妥的结论是：

> LogHound 的“文本证据 + 运行线索 + 结构路径 + 覆盖感知抑噪 + 排名融合”范式，可以在不重放 agent 环境的条件下迁移为执行步级根因排序，并在 AgenTracer coding test 上展现出执行步级定位效果；本实验只报告该迁移方法的绝对结果。

初版不能声称：

- 已经解决真实 SWE-bench Pro coding-agent 调试；
- 静态排名等价于严格的反事实因果识别；
- 在 127 条 test 上调出的最佳权重能够泛化。

这个边界既保留了 LogHound 迁移的研究贡献，也避免把“相关性排序”过度解释为已经验证的因果根因。
