# Qwen baseline 试跑

入口：`scripts/run_qwen_baseline.py`。使用 TrajAudit 官方 All-at-Once 的 system/user 提示词与标注准则，不导入或执行其 Claude 默认配置。仅接受模型名中包含 Qwen 且不包含 Claude 的模型，禁止自动回退。

按用户要求，本次使用移动云 MoMA 的 Qwen3.5-27B。用户提供端点 `https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions`；2026-10-05 使用该账户查询同源 `/v1/models` 成功，确认准确模型 ID 为 `Qwen3.5-27B`，已更新默认值。

用 `--api-key-file` 读取本地纯文本密钥文件；也可用 `--api-key-env MOMA_API_KEY` 读取环境变量。命令行仅传文件路径或变量名，密钥值不写入 manifest。

```bash
.venv/bin/python scripts/run_qwen_baseline.py \
  --root ../TrajAudit/RootSE \
  --trajaudit-source ../TrajAudit \
  --out-dir outputs/moma-qwen35-27b-short8-new \
  --api-key-file /path/to/moma_ai.key \
  --base-url https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions \
  --selection shortest-input --recover-reason-quotes
```

默认按 agent/model 分组轮流取样，共 8 条。上方命令使用 `--selection shortest-input`，按实际发送的输入字符数选择最短8条，不读取标签作选择。由于这8条均来自 AutoCodeRover，该试跑不代表完整基准。完整 102 条加 `--limit 0`，无网络预检加 `--dry-run`。输出目录必须为空；每条样本只发一次请求，首次失败后停止检查配置，不对错误请求连续收费。

默认使用普通文本模式；本次 JSON object 约束请求在180秒后超时，因此不设为默认。可用 `--response-format json_object` 单独实验。`--recover-reason-quotes` 仅兼容严格两字段结构中解释文字的引号问题；步骤 ID 必须能独立解析且属于原始轨迹，恢复记录显式标注。返回正文保存在各运行目录的 `responses/`；HTTP 状态、结束原因和脱敏错误信息一并记录。

输入为任务、完整轨迹和首条最终失败测试，不提供 gold patch 或根因标注。请求不截断原始轨迹；服务拒绝过长上下文时记录失败，不悄悄删去证据。模型输出必须包含有效的原始步骤 ID，截断响应不得当成有效结果。

产物包括请求/原数据哈希、响应报告的模型、逐条预测、token usage、运行状态与完整集合准确率。只在所有选定样本成功返回时生成 metrics；dry-run 不生成模型指标。

此前 `outputs/qwen-pilot-preflight/` 和 `outputs/qwen-full-preflight/` 使用旧的 Qwen3-235B 模型名，仅为无请求预检。Qwen3.5-27B 的独立预检输出为 `outputs/moma-qwen35-27b-pilot-preflight/` 和 `outputs/moma-qwen35-27b-full-preflight/`，API 调用数为 0。这不是新的 Qwen 推理结果。

接口已返回明确的32768 token上下文限制，原定分组pilot中的长轨迹被拒绝。不能直接以当前部署和不截断的All-at-Once协议跑完整102条；需要另行验证长上下文部署，或明确引入分块方法并作为不同baseline报告。实测记录见 `MOMA_QWEN_RESULTS.md`。
