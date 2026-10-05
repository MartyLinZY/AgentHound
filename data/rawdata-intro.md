# 当前原始数据格式

本说明只覆盖当前使用的 AgenTracer coding test 与 RootSE。原始数据保存在本机 `data/raw/`，不随代码仓库分发。

## AgenTracer coding test

**文件**：`data/raw/agentracer/tracertraj-code-test.parquet`，127 条轨迹。

Parquet 主要字段：

| 字段 | 含义 |
|---|---|
| `question_ID` | 任务/轨迹 ID |
| `question` | 编码任务描述 |
| `ground_truth` | 参考答案；排序阶段禁止读取 |
| `history` | 有序执行历史，通常为 NumPy 数组或列表 |
| `mistake_agent` | 根因 agent 标签；排序阶段禁止读取 |
| `mistake_step` | 根因步骤标签；排序阶段禁止读取 |
| `mistake_reason` | 人工根因说明；排序阶段禁止读取 |

`history` 中每个元素对应一个轨迹步骤，主要包含 `name`、`role`、`step` 和 `content`。`content` 可能混合思考、命令、消息及显式终端回传；适配器保留步骤边界，并只从回传和明确失败陈述提取证据。

审计分层是 A 硬失败 9 条、B 弱失败陈述 31 条、C 无失败观察 87 条。该文件名标明 test 子集，只用于固定方案的分层评估，不用于拟合参数或规则。

## RootSE

**文件组织**：`data/raw/rootse/data/<agent>/<model>/<instance>.json`，102 条轨迹、5,268 个步骤。来源版本和文件校验和记录在 `data/raw/rootse/manifest.json`。

每个 JSON 实例包含任务、轨迹和标签字段：

| 字段 | 含义 |
|---|---|
| `original_mission_prompt` | agent 接收的任务文本 |
| `original_traj` | 按顺序保存的轨迹步骤 |
| `failure_id` | 人工标注的最早决定性错误步骤；仅评估阶段读取 |
| `failure_reason` | 根因说明；仅评估阶段读取 |
| `agent`、`model`、`repo`、`instance_id` | 轨迹来源与任务元信息 |

`original_traj` 的步骤包含 `thought`、`response`、`action`、`observation`、`state` 等字段。适配器将任务、决策文本、动作和观察构造成排序输入，并将 `failure_id`、`failure_reason` 单独构造成标签。

## 输入与标签隔离

两种数据经 adapter 转换后都遵循相同约束：排序器只能接收 `TrajectoryInput`；标签对象仅交给评估器。不得从 `ground_truth`、`mistake_*`、`failure_id` 或 `failure_reason` 构造排序特征或失败观察。
