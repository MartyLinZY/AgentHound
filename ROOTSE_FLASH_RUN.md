# 自行运行 RootSE 六基线（flash）

已配置模型 `deepseek-v4-flash`；其余实验参数沿用上一轮：102条×6方法，temperature=0，输出32768，16案例并发。只换模型，未修改TrajAudit格式处理或失败策略。本入口不运行我们方法的LLM构图。

```bash
cd /Users/linzheyuan/Documents/ChatGPT/CodingAgentDebug/output/rootse_loghound_2026-10-05/AgentHound-clean
caffeinate -i bash scripts/run_rootse_flash.sh /Users/linzheyuan/code/selfbase/moma_ai.key
```

保持终端运行。脚本自动安装依赖、检查数据、测试、运行并生成结果。不要同时启动第二份。

运行中：终端显示进度 `[n/612]`，完整日志 `outputs/rootse-flash-console.log`。另一终端查看：

```bash
tail -f /Users/linzheyuan/Documents/ChatGPT/CodingAgentDebug/output/rootse_loghound_2026-10-05/AgentHound-clean/outputs/rootse-flash-console.log
```

结果目录：`outputs/rootse-suite-deepseek-v4-flash-32k-v1/`

- `EXPERIMENT_SUMMARY.md`：全部完成后自动生成的结果表与论文对照。
- `metrics.json`：各方法Tol@0..3、有效输出、失败计数、tokens。
- `predictions.csv`：逐案例预测、标签与容差命中。
- `<method>/<case>/execution.log`：具体案例执行日志。
- `<method>/<case>/calls/`：逐次请求及响应；不包含认证密钥。

意外中断后，确认旧进程已退出，再执行同一命令，复用已有终态结果；截断和格式失败等终态不会自动重跑。新输出目录与旧v4.1结果分开。

交付检查：103测试通过；612任务无API预检通过；尚未发出新模型推理请求。配置文件为 `experiments/rootse_suite/config_deepseek_v4_flash.json`。
