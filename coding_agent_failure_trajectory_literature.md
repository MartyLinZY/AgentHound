# Coding Agent 失败轨迹根因分析：论文推荐与研究设计

> 文献状态核验日期：2026-09-10。本文把“已发表”限定为已进入正式会议或期刊论文集；预印本单独列出。

## 结论先行

如果研究目标是从 coding-agent 的完整执行轨迹中定位“从哪里开始错、为什么错、错误如何传播、为何没有恢复”，最值得优先精读的是下面五篇：

1. **Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories**（ASE 2025，Distinguished Paper）——最适合学习怎样统一表示异构 agent 日志、怎样做成功/失败轨迹的混合方法分析。[1]
2. **Beyond Final Code: A Process-Oriented Error Analysis of Software Development Agents in Real-World GitHub Scenarios**（ICSE 2026 Research Track）——目前与 repository-level coding-agent 失败过程最直接的大规模实证研究。[2]
3. **Process-Centric Analysis of Agentic Software Systems**（PACMPL/OOPSLA 2026）——最适合把线性日志转为图，自动分析阶段转换、回退、重复、混乱和验证缺失。[3]
4. **Which Agent Causes Task Failures and When?**（ICML 2025 Spotlight）——虽然研究对象是多 agent，但它把根因分析明确形式化为“谁、在哪一步造成决定性错误”，非常适合迁移到单 coding agent 的“组件—时间步”归因。[4]
5. **SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents**（NeurIPS 2024）——最适合构造可执行证据，将“是否复现 issue”和“覆盖率变化”分开，避免只看最终 patch 是否通过。[5]

如果只读三篇，建议顺序是 **[1] → [2] → [3]**。它们分别提供统一轨迹表示、coding-agent 错误实证和可扩展过程分析，基本构成一套完整研究骨架。

## 核心论文

| 优先级 | 论文与正式出处 | 数据/对象 | 对根因分析最有价值的部分 | 主要局限 |
|---|---|---|---|---|
| S | **Understanding Software Engineering Agents**，ASE 2025 | RepairAgent、AutoCodeRover、OpenHands；120 条轨迹、2,822 次交互 | 将每一步统一为 `Thought–Action–Result`；结合统计分析、动作分类、序列模式挖掘与开放编码；指出失败轨迹常出现重复且不自适应的循环，以及 thought/action/result 语义不一致 | 样本不大；显式 thought 未必忠实反映模型内部因果过程 |
| S | **Beyond Final Code**，ICSE 2026 | 8 个高排名 agent；3,977 条求解轨迹、3,931 份测试日志；SWE-bench Verified 500 题 | 联合分析求解阶段和最终测试阶段；区分高频错误、难恢复错误、跨阶段残留错误、环境与 benchmark 故障 | 主要依赖 Python 异常，覆盖不到“命令成功但方向错误”等语义失败 |
| S | **Process-Centric Analysis of Agentic Software Systems**，OOPSLA 2026 | SWE-agent、OpenHands × 4 个 LLM；4,000 条轨迹 | Graphectory 用时间边和代码层级结构边表示动作；Langutory 将轨迹压缩为 Localization–Patching–Validation 阶段语言；可检测回边、重复、策略震荡和验证缺失 | 阶段映射含规则与人工设计成分；“低效模式”不总等于根因 |
| A | **Which Agent Causes Task Failures and When?**，ICML 2025 Spotlight | 127 个多 agent 系统配置的失败日志；Who&When 数据集 | 给出“责任 agent + 决定性错误步骤 + 原因说明”的标注任务和自动归因基线；最佳方法对责任 agent 的准确率为 53.5%，但精确步骤仅 14.2%，说明 step localization 仍很难 | 不是 coding 专用；多 agent 的“谁”要改造成模型、harness、tool、environment、evaluator 等组件归责 |
| A | **Why Do Multi-Agent LLM Systems Fail?**，NeurIPS 2025 Datasets & Benchmarks | 7 个框架、1,600+ 标注轨迹 | MAST 通过 Grounded Theory 得到 14 类失败，归入系统设计、agent 间失配、任务验证三组；提供人类标注与 LLM judge 扩展范式 | 多 agent 特有的通信错误并不都适用于单 agent；taxonomy 不宜原样照搬 |
| A | **SWT-Bench**，NeurIPS 2024 | 真实 GitHub issue、gold fix、gold tests | 把测试生成拆成 issue reproduction 与 coverage 两类可执行指标；生成测试还可作为候选 patch 过滤器，论文报告可将 SWE-agent 的 patch precision 提高约一倍 | 测试仍是不完备 oracle；生成测试可能与 agent patch 共享同一误解 |
| A | **AgentTelemetry**，AIware 2026 Benchmark & Dataset Track | 14 类故障、9 类 agent span、7 个框架；另含 SWE-bench Lite 案例 | 从可观测性基础设施角度定义应该记录哪些 span；展示普通 OpenTelemetry/GenAI trace 看不到规划、推理、guardrail、delegation、memory 等故障 | benchmark 中大量结果来自控制性故障注入；SWE-bench 干预实验样本较小且未达统计显著 |
| B | **AgentDiagnose**，EMNLP 2025 System Demonstrations | 通用 agent 轨迹，当前主要验证于 Web agent | 将探索/回退、任务分解、观察读取、自验证、目标质量量化，并与人工判断比较；适合借鉴诊断面板和训练数据筛选 | 不是 coding 场景；人工验证集仅 30 条轨迹 |
| B | **Watson: A Cognitive Observability Framework**，ASE 2025 | MMLU 与 AutoCodeRover/SWE-bench-lite 案例 | 提出 cognitive observability，并从 prompt attribution 反推出快速模型未显式给出的决策依据；适合研究“为什么选这个工具/文件” | 反推的 reasoning 是解释模型，不应当直接当作真实内部因果链 |
| B | **Coding Agents in the Wild**，IEEE Access 2026 | 12,433 个 agent PR、1,495 个热门仓库、5 类 coding agent | 提供 benchmark 外部效度：被拒 PR 的可见原因以 specification mismatch、logic defect 等功能问题为主；提醒把 CI、review、项目规范纳入根因 | 84.2% 被拒 PR 没有 inline feedback；只能分析有可见证据的子集，也没有完整生成轨迹 |

## 基础设施与场景论文

- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**（ICLR 2024 Oral）：repository-level issue resolution 的任务、gold patch 和 Fail-to-Pass/Pass-to-Pass 测试基础。[6]
- **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering**（NeurIPS 2024）：理解 ACI、代码浏览/编辑/执行工具如何塑造轨迹；它提示失败并非只有 base model 原因，接口设计本身也是归责对象。[7]
- **AgentBoard**（NeurIPS 2024 Datasets & Benchmarks）：提供 progress rate，而非只有终局 success；适合定义“轨迹是否持续取得进展”。[8]
- **Agent-as-a-Judge: Evaluate Agents with Agents**（ICML 2025）：DevAI 含 365 条层级需求，可借鉴“按需求树逐项检查整个过程”，比无 rubric 的单次 LLM judge 更适合可审计归因。[9]
- **SyncMind**（ICML 2025）：若研究多人/多 agent 协作开发，重点看 agent 对代码库状态变化的 out-of-sync 恢复能力。[10]
- **GSO**（NeurIPS 2025 Datasets & Benchmarks）：若研究性能优化型 coding agent，论文给出低层语言、惰性优化和瓶颈定位失败，并公开轨迹。[11]
- **Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents**（ICLR 2026）：EnConda-Bench 将环境配置拆为规划、错误感知/诊断、反馈驱动修复、最终执行，适合专门研究 setup/dependency 根因。[12]

## 建议采用的根因模型

不要把一条失败轨迹压成单一标签。建议至少同时标五层：

| 层次 | 核心问题 | 示例 |
|---|---|---|
| 任务结果 | 最终哪里没满足 oracle？ | Fail-to-Pass 未通过、回归、patch 未生成/未应用 |
| 故障表现 | 用户或 evaluator 最后看见什么？ | AssertionError、timeout、错误文件被修改、PR 被拒 |
| 首次偏离 | 最早哪个行为偏离了合理路径？ | 误读 issue、忽略 stack trace、错误定位模块 |
| 关键失败步骤 | 从哪一步开始，若不纠正就无法在剩余预算内恢复？ | 基于错误假设实施大范围 patch，随后验证不足而提交 |
| 根因组件 | 为什么会发生？责任落在哪个边界？ | task/spec、model、prompt/workflow、context/memory、tool/ACI、environment、test oracle/evaluator |

“首次偏离”“关键失败步骤”和“最后异常”经常不是同一步。例如，agent 先错误理解 API contract，随后做出表面合理的修改，最后在测试中出现 `TypeError`；`TypeError` 是 manifestation，错误 contract 假设才是 candidate root cause。只有在替换早期步骤、补充信息或进行 replay 后能改变后续结果，因果说法才更可信。

### 推荐 taxonomy

一级标签可以采用：

1. **任务/规格理解**：遗漏约束、歧义未澄清、目标漂移。
2. **规划与分解**：计划缺失、顺序错误、无效反复、预算分配错误。
3. **代码库理解与定位**：错误文件/符号、跨文件依赖遗漏、架构或领域知识不足。
4. **补丁推理与实现**：局部逻辑错误、过拟合样例、破坏隐式 API contract、改动范围不当。
5. **工具与观察处理**：工具选择或参数错误、忽略/误读输出、推理与动作不一致。
6. **环境与依赖**：安装、版本、路径、权限、数据库/服务、非确定性故障。
7. **验证与 oracle**：未复现原问题、测试不足、只跑回归不跑 reproduction、误判结果。
8. **终止与恢复**：过早终止、成功后不终止、循环、回退/换策略失败。
9. **状态与协调**：上下文丢失、记忆污染、代码状态不同步、子 agent 交接失真。
10. **基准/evaluator**：测试收集、环境构建、结果解析、错误格式或 false negative。

每类再加三个正交属性：`source_component`、`phase (L/P/V)`、`recoverability`。这样不会把“在哪里发生”“为什么发生”“是否恢复”混成一个分类维度。

## 可落地的研究方案

### 1. 数据与实验单元

- 以“task × agent framework × model × seed/run”为基本单元；同一任务至少保留多次 rollout，成功与失败配对分析。
- 优先从 SWE-bench Verified 开始，但增加更新更晚、语言更多或专项任务的数据，降低只对 12 个 Python repo 过拟合的风险。
- 保存完整环境身份：仓库 commit、镜像、依赖锁、测试命令、framework 与 prompt 版本、模型版本和解码参数。

### 2. 轨迹 schema

最小事件建议包含：`run_id, step_id, parent_step, timestamp, actor/component, phase, thought_summary, action/tool, arguments, cwd, observation, exit_code, file_diff, test_delta, token/cost, context/memory delta, termination_reason`。

内部 chain-of-thought 不可用或不可靠时，用**可观察的决策摘要、工具调用和状态变化**代替，不要让研究依赖私有 reasoning trace。

### 3. 人工根因标注

- 先用 50–100 条失败轨迹做开放编码，再形成 codebook；不要先把 MAST 或某篇论文的 taxonomy 生硬套入。
- 两名标注者独立标注，至少报告类别一致性、关键步骤距离和分歧仲裁过程。
- 每个标签必须绑定 evidence span；允许多标签和因果链，而非强制单一根因。
- 标注三个时间点：`first_deviation`、`critical_failure_step`、`manifestation_step`。
- 给根因置信度与“信息不足/不可判定”选项，避免 LLM judge 被迫编造原因。

### 4. 自动化诊断

可以组合三条路线：

1. **规则/程序分析**：重复同参工具调用、未测试便提交、测试前后错误相同、diff 与定位结果不一致。
2. **图与序列特征**：借鉴 Graphectory/Langutory，检测回边、阶段震荡、缺失验证和异常结构复杂度。
3. **evidence-grounded LLM judge**：输入 task、工具 schema、轨迹和 evaluator log，输出步骤、类别、责任组件、引用证据与置信度；只把 judge 当预测器，用人工 held-out 集校准。

评价指标至少包括：根因类别 macro-F1、责任组件准确率、关键步骤 exact/±1/±3 accuracy、证据 span precision/recall、校准误差，以及对同一任务不同 agent/run 的一致性。

### 5. 从相关性走向因果证据

轨迹长度、异常数量和失败高度相关，但都可能只是任务难度的结果。建议：

- 同题匹配成功/失败 rollout，控制 task difficulty、模型和 framework。
- 在候选关键步骤实施最小干预：补充遗漏观察、纠正工具参数、替换错误假设、强制运行 reproduction test，然后 replay 后续过程。
- 比较干预前后的解决率、剩余步骤、成本和是否出现同类错误。
- 将“干预能稳定改变结果”作为强根因证据；仅有时间先后或相关性时，称为 failure correlate，而不要称 root cause。

## 值得做的研究空白

最有潜力的题目是：**面向 repository-level 单 coding agent 的、可反事实验证的关键失败步骤定位**。

现有工作要么偏行为描述与相关性（ASE/ICSE/OOPSLA 三篇），要么在多 agent 场景做责任与步骤归因（Who&When），要么依赖预设故障或通用 agent。可以构造一个数据集，为每条 coding-agent 失败轨迹标注：

`首次偏离 → 错误传播 → 恢复尝试 → 关键不可恢复步骤 → 最终表现 → 责任组件`，

再通过 checkpoint replay 或最小轨迹干预验证标签。Who&When 的最佳方法对精确失败步骤只有 14.2% 准确率，说明“定位到哪一步”本身仍有很大研究空间；coding 任务又有测试、编译、diff 和版本状态等可执行证据，比开放式 agent 更适合做因果验证。[4]

## 前沿但尚不宜当作“已发表主证据”的工作

- **AgentRx: Diagnosing AI Agent Failures from Execution Trajectories**（arXiv 2026）：定义 critical failure step，使用约束生成、逐步验证日志和 LLM judge；方法与拟议课题非常接近，但截至核验日仍应按预印本引用。[13]
- **When Agents Go Astray: Course-Correcting SWE Agents with PRMs / SWE-PRM**（2025 workshop/arXiv）：taxonomy-guided process reward model 在运行中发现重复探索、循环、未终止等问题并干预，适合研究在线恢复，但不是主会正式论文。[14]
- **Wink: Recovering from Misbehaviors in Coding Agents**（AIware 2026）：来自生产流量，taxonomy 包括 specification drift、reasoning problems、tool-call failures，并报告大规模在线自干预结果。它已进入会议论文集，但属于较短的应用型会议论文；可作为产业证据，不宜替代主线三篇。[15]
- **Aegis: Taxonomy and Optimizations for Overcoming Agent–Environment Failures**（arXiv 2025）：提醒将环境视为责任组件，而非所有失败都归咎于模型。[16]
- **Understanding Code Agent Behaviour** 与 **Beyond Resolution Rates**（arXiv 2025/2026）：前者对 OpenHands、SWE-agent、Prometheus 的成功/失败轨迹做比较；后者特别提醒 trajectory length 与 failure 的关联会受任务难度混杂。适合跟踪，但应清楚标注为预印本。[17][18]

## Sources

1. Islem Bouzenia and Michael Pradel. “[Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories](https://conf.researchr.org/details/ase-2025/ase-2025-papers/40/Understanding-Software-Engineering-Agents-A-Study-of-Thought-Action-Result-Trajector).” ASE 2025 Research Papers; ACM SIGSOFT Distinguished Paper Award.
2. Zhi Chen, Wei Ma, and Lingxiao Jiang. “[Beyond Final Code: A Process-Oriented Error Analysis of Software Development Agents in Real-World GitHub Scenarios](https://arxiv.org/abs/2503.12374).” ICSE 2026 Research Track. DOI: 10.1145/3744916.3773140.
3. Shuyang Liu et al. “[Process-Centric Analysis of Agentic Software Systems](https://doi.org/10.1145/3798271).” PACMPL 10 (OOPSLA1), 2026.
4. Shaokun Zhang et al. “[Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems](https://proceedings.mlr.press/v267/zhang25cq.html).” ICML 2025.
5. Niels Mündler et al. “[SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/94f093b41fc2666376fb1f667fe282f3-Abstract-Conference.html).” NeurIPS 2024.
6. Carlos E. Jimenez et al. “[SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html)” ICLR 2024 Oral.
7. John Yang et al. “[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html).” NeurIPS 2024.
8. Chang Ma et al. “[AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/877b40688e330a0e2a3fc24084208dfa-Abstract-Datasets_and_Benchmarks_Track.html).” NeurIPS 2024 Datasets & Benchmarks.
9. Mingchen Zhuge et al. “[Agent-as-a-Judge: Evaluate Agents with Agents](https://proceedings.mlr.press/v267/zhuge25a.html).” ICML 2025.
10. Xuehang Guo et al. “[SyncMind: Measuring Agent Out-of-Sync Recovery in Collaborative Software Engineering](https://proceedings.mlr.press/v267/guo25l.html).” ICML 2025.
11. Manish Shetty et al. “[GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ddd98389004161943e646322a16348c3-Abstract-Datasets_and_Benchmarks_Track.html).” NeurIPS 2025 Datasets & Benchmarks.
12. Jiayi Kuang et al. “[Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents](https://openreview.net/pdf?id=Q8qgloDKUO).” ICLR 2026.
13. Shraddha Barke et al. “[AgentRx: Diagnosing AI Agent Failures from Execution Trajectories](https://arxiv.org/abs/2602.02475).” arXiv, 2026.
14. Shubham Gandhi et al. “[When Agents Go Astray: Course-Correcting SWE Agents with PRMs](https://arxiv.org/abs/2509.02360).” arXiv / NeurIPS 2025 workshop.
15. Rahul Nanda et al. “[Wink: Recovering from Misbehaviors in Coding Agents](https://arxiv.org/abs/2602.17037).” AIware 2026.
16. Kevin Song et al. “[Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures in LLM Agents](https://arxiv.org/abs/2508.19504).” arXiv, 2025.
17. Oorja Majgaonkar et al. “[Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories](https://arxiv.org/abs/2511.00197).” arXiv, 2025.
18. Tural Mehtiyev and Wesley Assunção. “[Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure](https://arxiv.org/abs/2604.02547).” arXiv, 2026.
19. Tianyue Ou et al. “[AgentDiagnose: An Open Toolkit for Diagnosing LLM Agent Trajectories](https://aclanthology.org/2025.emnlp-demos.15/).” EMNLP 2025 System Demonstrations.
20. Krishna Chaitanya Balusu. “[AgentTelemetry: A Fault Detection Benchmark and Toolkit for LLM Agent Observability](https://doi.org/10.1145/3805760.3814931).” AIware 2026.
21. Benjamin Rombaut et al. “[Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents](https://conf.researchr.org/details/ase-2025/ase-2025-papers/148/Watson-A-Cognitive-Observability-Framework-for-the-Reasoning-of-LLM-Powered-Agents).” ASE 2025.
22. Mahd Hindi et al. “[Coding Agents in the Wild: Failure Modes and Rejection Patterns of AI-Generated Pull Requests](https://doi.org/10.1109/ACCESS.2026.3696573).” IEEE Access, 2026.
