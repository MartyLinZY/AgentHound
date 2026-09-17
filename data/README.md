# 数据管理约定

所有实验数据必须位于本目录，不能让运行代码直接依赖仓库外的绝对路径。

## 目录约定

```text
data/
├── raw/                     # 原始数据，只读保存
│   ├── agentracer/
│   └── swebench_pro/         # ScaleAI 版任务基准与官方 mini-SWE-agent 轨迹
├── interim/                 # 适配器解析后的中间数据
├── processed/               # 统一 schema 的 JSONL
├── splits/                  # train/dev/test 清单，只存 trajectory_id
└── metadata/                # 来源、版本、校验和、许可和数据统计
```

## 当前已有数据

`raw/agentracer/tracertraj-code-test.parquet` 来自工作区中的 AgenTracer 本地仓库：

```text
project/AgenTracer/data/tracertraj-code-test.parquet
```

2026-08-22 的本地审计结果：

| 项目 | 结果 |
|---|---:|
| 轨迹数 | 127 |
| 字段数 | 7 |
| 字段 | `question_ID`, `question`, `ground_truth`, `history`, `mistake_agent`, `mistake_step`, `mistake_reason` |
| 缺失值 | 0 |
| `history` 物理类型 | NumPy `ndarray` |
| 根因 step 可唯一映射到 history | 127/127 |
| 根因 step 对应 agent 与标签一致 | 127/127 |
| 轨迹长度最小值/中位数/均值/最大值 | 2 / 13 / 20.80 / 151 |
| 根因 agent 为 Engineer | 92/127，72.4% |
| 根因位于 step 4 | 42/127，33.1% |
| 原始字段中的测试输出/退出码/失败 observation | 无 |

该文件名明确包含 `test`。在未取得官方训练集前，禁止使用它拟合权重、规则阈值或角色先验。第一版只使用预先固定的无监督参数，并把任何基于该文件进行的选择性调参标为探索性分析。由于它没有 step-level 失败 observation，不能直接满足 evidence-aware 主协议；若不补充外部测试输出，只能作为 trace-only 对照数据。

## SWE-bench Pro 数据状态（ScaleAI 企业版）

`data/raw/swebench_pro/SWE-bench_Pro-test.parquet` 已于 2026-08-22 从 HuggingFace 下载：

```text
https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/resolve/main/data/test-00000-of-00001.parquet
```

SHA-256：`c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8`（7.8 MB）

审计结果：

| 项目 | 结果 |
|---|---:|
| 条数 | 731 |
| 字段数 | 16 |
| 仓库数 | 11（ansible、openlibrary、flipt、qutebrowser、teleport 等） |
| 语言分布 | go 280 / python 266 / js 165 / ts 20 |
| `base_commit` 唯一性 | 731/731 |
| `fail_to_pass` 为空 | 0/731 |
| split | 仅 test |

**必须注意的事项：**

- 这是 **ScaleAI 企业版** SWE-bench Pro（731 条），**不是 Princeton 原版**（50 条、取自 SWE-bench Verified、Python 仓库）。原版 `princeton-nlp/SWE-bench_Pro` 在本环境 HF 上已不可访问（401）。论文/报告中描述数据来源时必须写清版本。
- `instance_id` 已匿名化（哈希形式），无法直接反查到原始 GitHub issue 和 PR，但 `patch`、`base_commit` 仍在。
- `fail_to_pass` / `pass_to_pass` / `issue_categories` / `issue_specificity` 是**字符串化的 JSON 列表**（`"[...]"` 形式），解析必须用 `ast.literal_eval`，直接 `json.loads` 会报错。
- HF 卡片未声明 license（与 SWE-bench Verified 一致）。数据内容来自各 GitHub 仓库 issue/补丁，各自许可不同；仅限研究用途，发布衍生数据前需自行确认。
- 它是**静态任务基准**（issue→patch→测试），原始数据**没有 agent 轨迹，也没有根因标签**。轨迹生成统一使用官方 mini-SWE-agent scaffold；生成后再按 DESIGN.md P3 的标注规范人工标注根因步。
- 轨迹目录必须随运行记录保存 mini-SWE-agent scaffold 的 commit、模型配置、turn limit、`instance_id`、输出 patch 和官方评测结果；不得混入其他 agent 框架生成的轨迹。
- 未经可行性门槛（DESIGN.md §5.5）通过前，不投入大范围标注；该文件只作为最终外部测试素材的原始任务源。

## 防泄漏规则

模型或排序器可读取：

- `question`；
- `history`；
- 从失败轨迹本身提取的终止状态、测试输出、错误消息和工具反馈。

模型或排序器不可读取：

- `ground_truth`；
- `mistake_step`；
- `mistake_agent`；
- `mistake_reason`；
- 由上述标签直接或间接派生的统计量。

此外，`failure_observations` 只有在原始轨迹中真实存在时才能进入输入；不能由 `mistake_reason`、`mistake_step` 或 `ground_truth` 反推生成。

统一后的数据分成两个文件，物理隔离输入与标签：

```text
processed/<dataset>/inputs.jsonl
processed/<dataset>/labels.jsonl
```

评估器是唯一允许同时读取两者的模块。
