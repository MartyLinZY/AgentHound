> 当前开发版本的完整对照见 [PPT_CODE_AUDIT.md](PPT_CODE_AUDIT.md)。以下记录保留方法结构与输入边界；历史结果不代表清理版成绩。

# RootSE 锚点回溯首版

这一版本将 LogHound 的“线索关联实体、沿结构寻找候选”用于 Coding Agent 的步骤定位。它是可验证的实验实现，当前规则图实验没有证明准确率提升，因此保留现有三通道默认行为。

## 实现范围

- `adapters/rootse.py`：`trace-only` 保留原轨迹输入；`postmortem` 额外提供首条 `failed_tests` 的代码、报错与测试 ID，遵循 TrajAudit 的首条测试选择策略。测试保存在独立 `DiagnosticTest` 中，无历史步骤 ID。`gold_patch`、`failure_id`、`failure_reason` 不进入排序输入。
- `anchors.py`：识别测试失败、工具/环境错误、运行错误和非零退出码，记录原字段、精确字符范围、原文、实体、执行状态和来源。规则扫描完整 observation，不受 LLM 依赖抽取的前缀截断影响。匹配仍是候选证据，打印出的代码或文档可能造成误报。
- `graph.py`：原 `pagerank()` 默认仍为全图均匀重启。传入 personalization 后，重启和悬空质量均返回锚点分布，避免将失败分数传播到无关连通分量。
- `rankers/anchor_path.py`：复用语义通道、融合权重和依赖图，只改变实验变量；输出 Top-5 的依赖路径、证据片段和锚点关联。没有非空依赖路径的锚点得分明确标为 `anchor_seed_without_dependency_path`。
- `rootse_evaluation.py`：要求完整、唯一的轨迹与步骤集合，核对标签 ID、收敛状态和输入协议；报告 Top-1/3/5、MRR、±1/±2 步容忍指标、按 Agent/仓库/长度分组结果。距离按原始顺序计算，不假定字符串 ID 可以转为整数。
- `scripts/run_rootse_offline.py`：只用规则图或已有验证缓存，无 API 调用。产出源码/配置/数据哈希、依赖图、各版本完整排名、指标、成对得失样本和耗时。

## 两类证据边界

历史决策依赖抽取仍只接收步骤字段。Postmortem 的测试信息不传入 `_build_request()`，因此不会让早期决策“提前知道”最终测试。

外部测试可通过精确路径或 test 名称匹配关联历史步骤。这类关联标为 `entity_match`，用于检索和初始化，不称为因果、版本或覆盖关系；关联的文本区间一并保存。没有匹配时锚点保持 unlinked，不附会到最后一步。

同一命令后续零退出且无识别到的失败时，仅记录 `later_same_command_success_unverified`，保留失败分数。当前实现无法保证测试范围和代码版本完全相同，不宣称已确认恢复。工具调用成功也不意味着最终任务成功。

## 四个对照版本

| 参数 | 线索 | 结构分数 |
|---|---|---|
| `baseline` | 原固定规则 | 原全图 PageRank |
| `anchors-global` | 新结构化锚点 | 原全图 PageRank |
| `legacy-backtrace` | 原固定规则锚点 | 失败锚点 personalized PageRank |
| `anchors-backtrace` | 新结构化锚点 | 失败锚点 personalized PageRank |

所有版本使用相同依赖集合、TF-IDF 和 `0.50/0.05/0.45` 权重。在 Postmortem 下所有版本的语义查询均增加相同测试代码和错误描述，回溯版本另允许外部锚点实体关联。未使用标签搜索权重。

没有任何可关联锚点时，回溯版本明确进入 `semantic_only_no_linked_anchor`，图分和局部线索分为零，保留语义排序。图不连通但存在锚点时，PPR 仍可能停在症状步骤；当前把这个现象作为实验结果报告，未根据全量标签再加入调参补丁。

默认离线实验：

```bash
.venv/bin/python scripts/run_rootse_offline.py \
  --root /path/to/TrajAudit/RootSE \
  --out-dir outputs/rootse-anchor-run1
```

只跑一种协议与指定版本：

```bash
.venv/bin/python scripts/run_rootse_offline.py \
  --root data/raw/rootse \
  --protocols trace-only \
  --variants baseline anchors-backtrace \
  --out-dir outputs/rootse-anchor-run2
```

使用已验证 LLM 依赖缓存：

```bash
.venv/bin/python scripts/run_rootse_offline.py \
  --root data/raw/rootse \
  --graph-source cache \
  --cache-dir outputs/rootse/llm_cache \
  --out-dir outputs/rootse-anchor-cached
```

缓存复用还需要原来的 `MODEL_NAME`、`API_ENDPOINT`（若曾配置）和请求参数，它们参与缓存键计算。密钥不写入 manifest。该入口始终 cache-only，缺缓存直接报未完成；需要新构图时先使用原 `run_rootse_experiment.py`，再运行本入口。不要把缓存读取费用当作首次构图费用。

## 验证和当前限制

测试覆盖：原始 ID 不变、HF/GitHub 布局同 ID、输入序列化回读、改变 gold 字段不影响排名、最终测试不进入历史依赖请求、原文区间准确、Python/Go/JS 错误格式、零失败不建锚点、无关成功不当作恢复、图传播方向、无关连通分量零失败分数、路径证据、无锚点回退、完整集评估、离线不调用 API、缺缓存不出完整指标、旧结果不被覆盖。

当前只实现规则锚点与候选路径输出，尚未加入新的 LLM 锚点分类/候选判定、自动恢复验证或反事实回放。Postmortem 不执行测试，也不下载仓库源码。现有文件状态解析器原样保留，规则图稀疏是已知限制，冻结全字段 LLM 图实验已完成，未证明可靠增益；结果见相邻冻结仓库。当前清理版规则追踪有行为修复，尚未运行其新图定位实验。

RootSE 已用于方法开发和错误分析，当前结果是探索性实验。102 条对应 98 个任务，后续交叉验证按任务分组，最终泛化结论需要未参与开发的数据。分数不是校准过的根因概率，证据路径也不等于因果证明。
