# 我们方法的 DeepSeek 官方补跑

此前 `run_rootse_deepseek.sh` 的六方法清单仅包含外部基线；本入口补齐当前 AgentHound-clean 的 LLM 依赖构图及三通道排序/锚点回溯。结果单独保存，不覆盖已完成基线或历史移动云图。

```bash
cd /Users/linzheyuan/Documents/ChatGPT/CodingAgentDebug/output/rootse_loghound_2026-10-05/AgentHound-clean
caffeinate -i bash scripts/run_rootse_deepseek_ours.sh \
  /Users/linzheyuan/code/selfbase/deepseek_official_flash.key 16 \
  2>&1 | tee -a outputs/rootse-deepseek-ours-console.log
```

第二参数为并发数，默认16；第三参数可指定新输出目录。已准备默认16并发的目录，若改变并发参数需指定新目录，避免协议指纹冲突。

入口依次执行：校验已完成基线评估 → 冻结构图输入 → 9目标技术pilot → pilot门槛通过后执行5166目标构图（102轨迹）→ 同图两种输入协议、四种排序变体 → 离线报告，与已完成的六个DS基线放入同表。不会重跑六基线。两种输入协议为postmortem / trace-only；四种变体为baseline / anchors-global / legacy-backtrace / anchors-backtrace。

官方设置与六基线一致：deepseek-flash，https://api.deepseek.com/chat/completions，thinking=enabled，reasoning_effort=high，temperature=0，每调用输出预算32768。保留输出截断、格式失败和部分图记录；本次没有根据已见准确率调整提示词、验证器、排序权重或预算。每调用上限一致不代表总成本一致。

默认结果目录：`outputs/rootse-migration-state-v2-deepseek-official-flash-thinking-32k-v1/`。准备检查已通过：102轨迹、5166目标、最大请求503600字符。补跑入口交付时只进行了离线准备与模拟响应测试，没有发送新API请求，真实pilot会在用户运行上述命令时执行。

技术pilot失败会停止全量，不会跳过门槛。已有终态请求可恢复复用；pilot请求单独计费、单独记录，不计入正式5166目标请求用量。最终 `EXPERIMENT_SUMMARY.md` 包含8组我们方法结果及6基线；`metrics.json` / `rankings/` 等路径以报告产物为准。
