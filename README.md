# LogHound-Agent-Debug

对 RootSE 的 coding-agent 失败轨迹做步骤级根因排序：给定任务和一条已保存的轨迹，为每个步骤计算可疑度并输出排名。排序阶段只读轨迹本身，**不读** `failure_id`、`failure_reason`、`gold_patch` 等标注字段。

本仓库只处理 RootSE 一个数据集（102 条轨迹 / 5,268 步），不再包含其他数据集的适配器、脚本或产物。

## 方法

四路无标签通道，加权 Reciprocal Rank Fusion（k=60）：

| 通道 | 度量 | 权重 |
|---|---|---:|
| `semantic` | 步文本（decision + action，不含 observation）与任务/失败文本的 TF-IDF 余弦 | 1.0 |
| `path` | 依赖图反向图上的普通 PageRank（均匀 teleport） | 0.5 |
| `anchor` | 步文本与失败测试报错符号的重合 | 0.5 |
| `action` | 动作类型二值先验：决策步（edit / 纯推理）= 1，取证据步（view / shell）= 0 | 1.0 |

`action` 只有两档、不分 agent、不标定 lift；RRF 只用名次，单调就够了。压缩轨迹（≤5 步，如 auto-code-rover 把整条轨迹压成 3–4 段）里动作语义不可比，该通道跳过。

常量通道会被 RRF 跳过（不制造伪信号）；同分按 ordinal 升序，这个并列规则隐含"越早越可疑"的先验，不要改成平均名次。

## 目录

```
configs/default.yaml                 # 四通道/RRF/LLM 请求参数
data/raw/rootse/                     # RootSE 原始轨迹与标签（本地，不提交）
outputs/baseline/                    # 唯一基准产物
outputs/cache/dependency_cache/      # 依赖抽取缓存（离线复跑用）
scripts/run_rootse_experiment.py     # 跑全量：依赖 → 排序 → 指标
scripts/evaluate_rootse.py           # 单独复算指标
scripts/probe_channel_gap.py         # 只读探测：通道消融/池内误差归因
src/loghound_agent_debug/
  adapters/rootse.py                 # 无标签适配器（anchor 只从 failed_tests 抽取）
  features/{semantic,anchor,action_type}.py  # 静态特征与动作分类
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
| 36 (35.3%) | 59 (57.8%) | 71 (69.6%) | 81 (79.4%) | 0.509 |

分 agent Top-1 / Top-5 / Top-10：swe-agent 39.6 / 64.2 / 77.4；openhands 23.3 / 76.7 / 86.7；auto-code-rover 50 / 100 / 100；swe-agent-live 33.3 / 44.4 / 44.4。

相对三通道（28 / 50 / 60 / 78，MRR 0.435）：Top-1 +8、Top-3 +9、Top-5 +11、MRR +0.073。配对检验 Top-1 p=0.057、Top-3/Top-5 p<0.01。

**这是唯一基准**。任何改动都要在同一批 102 条上对比，并同时报 Top-K 与 MRR；池内精确率必须和 Recall@K 同表（分母是"根因仍在池内的轨迹数"，单独看会虚高）。

## 无标签保证

`TrajectoryInput` 不包含标注字段；`failure_anchors` 只从 `failed_tests.error_description` 抽取，且有测试断言 `failure_reason`/`gold_patch` 不进入 anchor 或排序输入。评估阶段才读 `labels.jsonl`。

## 说明

`docs/` 被 `.gitignore` 排除，不受版本控制，改写前请先备份。
