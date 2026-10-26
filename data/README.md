# 数据目录说明

本目录只保存 **RootSE** 一个数据集的说明与本地原始数据。原始数据、标签和运行产物不纳入 Git。

## 目录结构

```text
data/
└── raw/
    └── rootse/          # RootSE 发布文件：data/、README.md、manifest.json
```

`outputs/` 下的产物（`baseline/`、`cache/`）同样被 `.gitignore` 排除。

## 数据集

RootSE 是公开发布的 repository-level coding-agent 轨迹集：102 条人工标注轨迹，合计 5,268 步。原始文件按 `agent/model/instance_*.json` 组织，每条轨迹包含 `original_traj`（步骤序列）、`failure_id`（最早决定性错误步骤）、`failure_reason`（诊断说明）、`failed_tests` 和 `gold_patch`。

| 项 | 值 |
|---|---|
| 轨迹数 | 102 |
| 步骤数 | 5,268 |
| 轨迹长度 | 中位 52 步（p25=37, p75=69, max=155） |
| 根因绝对步序号 | 中位 19（p25=6, p75=28） |
| agent 族 | swe-agent 53、openhands 30、auto-code-rover 10、swe-agent-live 9 |

## 标签隔离

- 排序输入 `TrajectoryInput` 不含任何标注字段。
- `failure_anchors` 只从 `failed_tests.error_description` 抽取，不读 `failure_id` / `failure_reason` / `gold_patch`；`tests/test_semantic_anchor_rrf.py` 与 `tests/test_no_label_leakage.py` 对此有断言。
- 评估阶段才读取 `failure_id` 作为根因步骤。

## 依赖图数据

依赖图由两部分合成：

1. `rootse_operations.track_state_dependencies`：从 action/observation 机械抽取文件状态依赖，确定性、无网络；
2. `rootse_dependencies.DependencyExtractor`：LLM 提议的跨步依赖，必须通过引文校验才成为图边，结果缓存在 `outputs/cache/dependency_cache/`。

完整重跑需要 `API_KEY` 与 `MODEL_NAME`；只复现基线时用 `--cache-only`，零 HTTP 请求。

## 许可

RootSE 上游数据集 README 声明 MIT；引用或分发前请核对上游版本及说明。共享代码仓库不作为数据集分发渠道。
