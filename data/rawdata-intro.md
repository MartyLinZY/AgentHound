# 一、AgentTracer 数据集

**位置**：`data/raw/agentracer/tracertraj-code-test.parquet`（127 行 × 7 列，每行 = **一条被注入故障的 coding-agent 执行轨迹**，来自 MetaGPT 式多 agent 团队模拟，并有根因标注）。

## 7 个字段

| 字段               | 类型      | 含义                                                                                                    |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `question_ID`    | str     | 任务唯一 ID，如 `Algorithm_34582_I`（任务来源+题号+失败注入标记 I）                                                       |
| `question`       | str     | 任务描述正文。含题目标题、Problem Statement、函数签名、输入输出约束、示例、Guidelines                                              |
| `ground_truth`   | str     | 该任务的**标准正确答案代码**（如完整的 `is_interleave` 实现）。⚠️ 推理时不可用                                                   |
| `history`        | ndarray | **执行轨迹本身**：一个有序的 agent 动作序列（长度 2~151，均值 ~20.8），见下方详解                                                |
| `mistake_agent`  | str     | 根因 agent 角色：Engineer (92) / Product Manager (19) / Team Leader (11) / Architect (5)                   |
| `mistake_step`   | str     | 根因步骤的下标（0-based；转换为 int 后能直接索引 `history[step]`），127 条全部对齐、无越界                                                  |
| `mistake_reason` | str     | 人工写的根因解释，如 "This step is critical because it directly affects the logic that determines..."。⚠️ 推理时不可用 |

## `history` 的内部结构（重点）

是一个 numpy `ndarray`，每个元素是一个 dict，**4 个固定键**：

| 键         | 示例值                            | 含义                                                                                        |
| --------- | ------------------------------ | ----------------------------------------------------------------------------------------- |
| `name`    | `"Team Leader"` / `"Engineer"` | 执行该步的 agent 角色（Team Leader 893 / Engineer 1233 / PM 310 / Architect 199 / Data Analyst 7） |
| `role`    | `"assistant"`                  | 对话角色，全部是 assistant；该字段本身没有消息收发语义，显式的发送关系需要从 `content` 中的消息文本解析 |
| `step`    | `3`                            | 步骤序号，0-based，与数组下标一致                                                                      |
| `content` | 见下                             | **核心内容**，一段自由文本，混合了 thinking / 消息 / 工具命令                                                  |

`content` 里的实际内容有 4 种形态（**全部是文本，没有独立的结构化字段**）：

1. **thinking 帧**（最多，2642 条里 2450 条）——固定模板开头 `"Team Leader thinking: \n# Past Experience\n<experience>\n# Tool State\n# Current Plan\n[GOAL]..."`，后面跟任务复述和 agent 的思考；
2. **输出命令**——思考后用 ```json 代码块给出一组命令，如：

   ```json
   [{"command_name": "Plan.append_task", "args": {...}},
    {"command_name": "Editor.create_file", "args": {"filename": "interleaving_strings_prd.md"}},
    {"command_name": "TeamLeader.publish_message", "args": {"send_to": "Alice"}}]
   ```

   工具命令包括：`Editor.*`（文件读写）、`Plan.*`（任务管理）、`TeamLeader.publish_message`（角色间消息）、`RoleZero.reply_to_human`、`end` 等；
3. **消息**——`"Team Leader sends to Alex: Please implement..."`（188 条）；
4. **工具结果回传**——理论上有（如 open_file 后的文件内容），但我抽查时**没有独立的 observation 字段**，都嵌在后续 thinking 的文本里。

一句话：**一条记录 = 一个 agent 的一次"思考+动作"回合，整条轨迹 = 团队按流水线串行接力（Team Leader→PM→Architect→Engineer），每时刻只有一个 agent 在工作**。

## 一条示例（Algorithm_34582_I，Interleaving Strings 任务）

```
history[0]  Team Leader   thinking + 拆解任务、分配 PRD 给 Alice
history[2]  Team Leader   "sends to Alice: 创建 PRD..."
history[3]  Product Mgr   thinking + Editor.create_file 创建 prd.md   ← mistake_step=3 根因在这
history[10] Product Mgr   Plan.finish_current_task
history[12] Team Leader   转发给 Bob 设计架构
history[15] Architect     Editor.open_file 读 PRD
history[17] Architect     Editor.write 写 system_design.md
history[21] Team Leader   转发给 Alex 实现
history[22] Team Leader   "sends to Alex: 请实现..."                ← 最后一步，无后续（这就是失败轨迹）
```

***

# 二、SWE-bench Pro（ScaleAI 企业版）

**位置**：`data/raw/swebench_pro/SWE-bench_Pro-test.parquet`（731 行 × 16 列，每行 = **一个"GitHub issue → 金标准补丁 → 测试"的任务实例**，注意：**没有执行轨迹、没有根因标签**）

## 16 个字段

| 字段                           | 示例                                                                                                       | 含义                                                                                               | |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | :---------------------------------------------------------- |
| `repo`                       | `NodeBB/NodeBB`                                                                                          | 仓库名（共 11 个企业级仓库：ansible、openlibrary、flipt、qutebrowser、teleport、navidrome 等）                      | |
| `instance_id`                | `instance_NodeBB__NodeBB-0499890...-vnan`                                                                | 匿名化 ID（`instance_<repo><base_commit 哈希>-v<随机串>`），无法反查原始 issue                                    | |
| `base_commit`                | `1e137b0705...`                                                                                          | 任务起始 git commit，731 条全部唯一                                                                        | |
| `patch`                      | `diff --git ...` (1.4k~180k 字符)                                                                         | **金标准补丁**：维护者合并的完整 diff，`+` 行即正确答案                                                               | |
| `test_patch`                 | `diff --git a/test/database/keys.js ...`                                                                 | 伴随的测试改动（325~322k 字符），用于验证修复                                                                     | |
| `problem_statement`          | `"**Title: Email Validation Status Not Handled...**"`                                                    | issue 正文：标题、描述、复现步骤、期望行为（419~8k 字符）                                                             | |
| `requirements`               | `"- The loadUserInfo(...) should include logic to..."`                                                   | 任务需要动到的函数/组件及期望行为（124~6.7k，可空）                                                                  | |
| `interface`                  | `"Type: Method\nName: db.mget\nPath: src/..."`                                                           | 涉及接口/API 的签名、路径、输入输出说明（可空）                                                                       | |
| `repo_language`              | `js`                                                                                                     | 仓库语言：go 280 / python 266 / js 165 / ts 20                                                        | |
| `fail_to_pass`               | `["test/database.js                                                                                    | Test database test/database/keys.js::Key methods...", ...]`                                     | 应用金标准补丁**前失败、后通过**的关键测试（字符串化的 JSON 列表，需 `ast.literal_eval`） |
| `pass_to_pass`               | `["test/database.js                                                                                    | Test database should work", ...]`                                                               | 前后都必须**保持通过**的回归测试（防止修复破坏其它功能）                              |
| `issue_specificity`          | `["major_bug","data_bug","ui_ux_bug"]`                                                                   | 问题类型标签（major_bug/code_quality_enh/refactoring_enh/edge_case_bug/data_bug/api_feat 等 8 类） | |
| `issue_categories`           | `["back_end_knowledge","database_knowledge","authentication_authorization_knowledge","ui_ux_knowledge"]` | 涉及的知识领域（共 8 类，可以多标签）                                                                             | |
| `before_repo_set_cmd`        | `git reset --hard <commit>\ngit clean -fd\n...`                                                          | 评测环境初始化命令（把仓库重置到 base_commit）                                                                   | |
| `selected_test_files_to_run` | `["test/database.js","test/database/keys.js","test/user/emails.js"]`                                     | 需要运行的测试文件清单                                                                                      | |
| `dockerhub_tag`              | `nodebb.nodebb-NodeBB__NodeBB-0499890...`                                                                | 官方评测 Docker 镜像 tag（`swebench` harness 用它建环境）                                                     | |

## 几个要点

- `patch` + `test_patch` 是"答案"，`FAIL_TO_PASS` 命中 + `PASS_TO_PASS` 不破 = 任务解决；
- **判定标准是跑测试，不是文本相似度**——评测需要在 Docker 里应用智能体生成的 patch 后执行测试；
- 附带字段（`requirements`、`interface`、`issue_specificity`、`issue_categories`）是 ScaleAI 企业版比 Princeton 原版多的，适合做任务难度分层、给 agent 的提示工程、或者按类别做子分析。

***

# 三、两者对比

| | AgentTracer                        | SWE-bench Pro (ScaleAI)       |
| ------- | ---------------------------------- | ----------------------------- |
| 一个单位是   | **失败的执行轨迹**                        | 静态任务实例                        |
| 是否有执行步骤 | ✅ 有（history，串行多 agent）             | ❌ 没有，需自己跑 agent 生成            |
| 是否有根因标签 | ✅ `mistake_step` + `mistake_agent` | ❌ 无                           |
| 答案形式    | `ground_truth`（参考代码）               | `patch` + `test_patch` + 测试用例 |
| 失败如何定义  | 数据生成流程中的故障注入；当前文件不含 step-level 失败 observation | 任务本身待解决，成败由测试判定               |
| 规模/域    | 127 / 多 agent coding               | 731 / 11 个企业级仓库，4 种语言         |
| 在你的流程中  | **P0-P2 trace-only 对照集**；补充失败 observation 后才可作为 evidence-aware 主验证集 | **P4 最终外部测试素材**（先跑 agent 再标注） |
