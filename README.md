# LogHound-Agent-Debug

对 RootSE 的 coding-agent 失败轨迹做步骤级根因排序：给定任务和一条已保存的轨迹，为每个步骤计算可疑度并输出排名。排序阶段只读轨迹本身，**不读** `failure_id`、`failure_reason`、`gold_patch` 等标注字段。

本仓库只处理 RootSE 一个数据集（102 条轨迹 / 5,268 步），不再包含其他数据集的适配器、脚本或产物。

## 方法

三路无标签通道，加权 Reciprocal Rank Fusion（k=60）：

| 通道 | 度量 | 权重 |
|---|---|---:|
| `semantic` | 步文本（decision + action，不含 observation）与任务/失败文本的 TF-IDF 余弦 | 1.0 |
| `path` | 依赖图反向图上的普通 PageRank（均匀 teleport） | 0.5 |
| `anchor` | 步文本与失败测试报错符号的重合 | 0.5 |

常量通道会被 RRF 跳过（不制造伪信号）；同分按 ordinal 升序，这个并列规则隐含"越早越可疑"的先验，不要改成平均名次。

## 目录

```
configs/default.yaml                 # 三通道/RRF/LLM 请求参数
data/raw/rootse/                     # RootSE 原始轨迹与标签（本地，不提交）
outputs/baseline/                    # 唯一基准产物
outputs/cache/dependency_cache/      # 依赖抽取缓存（离线复跑用）
scripts/run_rootse_experiment.py     # 跑全量：依赖 → 排序 → 指标
scripts/evaluate_rootse.py           # 单独复算指标
scripts/probe_channel_gap.py         # 只读探测：通道消融/池内误差归因
src/loghound_agent_debug/
  adapters/rootse.py                 # 无标签适配器（anchor 只从 failed_tests 抽取）
  features/{semantic,anchor}.py      # 两路静态特征
  graph.py                           # 依赖图 + PageRank
  rootse_dependencies.py             # LLM 依赖抽取（带缓存/预算/引文校验）
  rootse_operations.py               # 从 action/observation 抽状态依赖
  rankers/loghound_static.py         # 三通道 RRF 排序器
```

## 运行

数据放在 `data/raw/rootse/`。完整跑一遍需要 `API_KEY`、`MODEL_NAME`（可选 `API_ENDPOINT`），放在本地 `.env`，不要提交。

```bash
# 全量（首次需要依赖抽取请求）
python scripts/run_rootse_experiment.py --workers 1 --out-dir outputs/baseline

# 只复用缓存、零网络请求（复现基线）
python scripts/run_rootse_experiment.py --workers 1 --cache-only --out-dir outputs/baseline
```

中断后用 `--resume` 只补缺失轨迹。

## 基准结果

`outputs/baseline/metrics.json`（102 条，零 HTTP 请求复现）：

| Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---:|---:|---:|---:|---:|
| 28 (27.5%) | 50 (49.0%) | 60 (58.8%) | 78 (76.5%) | 0.435 |

分 agent：swe-agent 30.2 / 49.1 / 56.6 / 73.6；openhands 16.7 / 33.3 / 53.3 / 83.3；auto-code-rover 50 / 100 / 100 / 100；swe-agent-live 22.2 / 44.4 / 44.4 / 44.4。

**这是唯一基准**。任何改动都要在同一批 102 条上对比，并同时报 Top-K 与 MRR；池内精确率必须和 Recall@K 同表（分母是"根因仍在池内的轨迹数"，单独看会虚高）。

## 无标签保证

`TrajectoryInput` 不包含标注字段；`failure_anchors` 只从 `failed_tests.error_description` 抽取，且有测试断言 `failure_reason`/`gold_patch` 不进入 anchor 或排序输入。评估阶段才读 `labels.jsonl`。

## 说明

`docs/` 被 `.gitignore` 排除，不受版本控制，改写前请先备份。
