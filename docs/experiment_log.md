# 实验日志

## 2026-08-23：P0/P1 首次可复现实验

- 数据：本地 AgenTracer coding test，127 条；不读取 `ground_truth`、`mistake_step`、`mistake_agent` 或 `mistake_reason` 参与排序。
- 证据审计：A 硬失败 9 条，B 弱陈述 31 条，C 无证据 87 条，符合设计基线。
- 固定配置：`configs/default.yaml`；clue 规则版本 `0.1.0`；TF-IDF 任务权重 0.4；全图 PageRank alpha 0.85；融合权重 0.12/1.0/1.0。
- specificity：主排名使用 singleton 模式（每步为 1），同时报告跨轨迹模板统计；本次模板 singleton ratio 为 0.959742。
- 产物：`outputs/rankings/`、`outputs/metrics/`、`outputs/cases/`。

这次运行的完整指标保存在 `outputs/metrics/summary.json`，逐条排名保存在
`outputs/rankings/rankings.jsonl`。输出目录不提交到版本库，使用 README 中的命令即可重建。

## 2026-09-02：全图 PageRank 方案复现实验

- 目的：验证第三通道改为不依赖失败节点/`clue_hits` 的全图非个性化 PageRank 后，能否完成 step-level 根因排序。
- 运行命令：`conda run -n mybook python scripts/run_experiment.py`。
- 数据：本地 AgenTracer coding test，127 条；A/B/C 分层为 9/31/87。排序阶段不读取标签字段，根因标签仅用于评估。
- 配置：`configs/default.yaml`；clue 规则版本 `0.1.0`；TF-IDF 任务权重 0.4；PageRank `alpha=0.85`、`tolerance=1e-6`；融合权重 semantic/clue/path=`0.12/1.0/1.0`；specificity 仅作诊断，不乘入 PageRank。
- PageRank 审计：127/127 条收敛；PageRank 概率和最大误差为 0；每条轨迹均运行结构通道，C 档仅关闭 clue。

### 分层指标

| 分层 | n | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|---:|
| A 硬失败 | 9 | 0.000 | 0.333 | 0.333 | 0.213 |
| B 弱证据 | 31 | 0.032 | 0.258 | 0.452 | 0.219 |
| C 无证据 | 87 | 0.241 | 0.540 | 0.644 | 0.408 |
| 全部 | 127 | 0.173 | 0.457 | 0.575 | 0.348 |

### 结果解读

- A 档 9 条硬失败中，根因进入 Top-3 的有 3 条，Top-1 为 0 条；当前结构通道和固定融合仍不足以稳定定位硬失败根因。
- B 档结果受模板化 confession 影响，根因 Top-5 命中 14/31；该结果只作为弱证据稳健性观察。
- C 档没有可见失败线索，但 PageRank 仍正常运行；其结果不能解释为完整 evidence-aware 方法效果。
- 当前结论只描述新方案的绝对效果，不与旧版 PPR 结果作未经控制的优劣比较。后续应重点检查同 agent 链、早期编排节点和图边缺失造成的结构偏置。

### 2026-09-08：新增 agent-level 评测

- agent-level 排名按每个 agent 的最佳 step 在 step 排名中的首次出现顺序去重；step-level 继续按根因 step 计算 Hit@K。两种粒度都输出 Hit@1/3/5 和 MRR。
- 127 条全部数据：agent-level Hit@1=`0.496`（63/127），Hit@3=`0.969`，Hit@5=`1.000`；step-level Hit@1=`0.173`（22/127），Hit@3=`0.457`，Hit@5=`0.575`。
- 新结果写入 `outputs/metrics/summary.json`；逐轨迹的预测 agent、agent 排名和两套命中字段写入 `outputs/metrics/per_trajectory.csv`。

### 产物

- 汇总指标：`outputs/metrics/summary.json`
- 逐轨迹排名：`outputs/rankings/rankings.jsonl`
- A 档案例：`outputs/cases/a_tier_cases.json`、`outputs/cases/case_report.md`
- 证据分层：`data/splits/evidence_split.json`
