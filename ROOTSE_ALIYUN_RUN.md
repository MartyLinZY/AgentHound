# 阿里云 Qwen3.8-Flash 六基线

模型：`qwen3.8-flash`。使用用户指定的阿里云北京MAAS接口；开启 `enable_thinking=true`、流式返回并汇总usage。正式实验保持temperature=0、max_tokens=32768、16案例并发，共6方法×102案例。仅更换供应商/模型及接收传输方式，未修改TrajAudit动作格式策略。

密钥保存于仓库外 `/Users/linzheyuan/code/selfbase/aliyun_maas_qwen38.key`，权限0600；不在代码、日志或配置中。

```bash
cd /Users/linzheyuan/Documents/ChatGPT/CodingAgentDebug/output/rootse_loghound_2026-10-05/AgentHound-clean
caffeinate -i bash scripts/run_rootse_aliyun.sh /Users/linzheyuan/code/selfbase/aliyun_maas_qwen38.key
```

运行进度在终端 `[n/612]`。完整日志 `outputs/rootse-aliyun-console.log`。

完成后查看 `outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1/EXPERIMENT_SUMMARY.md`；同目录 `predictions.csv` 为逐条结果，`metrics.json` 为指标和费用用量。每次调用保存原始`.sse`流和合并后的response JSON；思考内容不会作为最终答案解析。

中断后确认旧进程停止，再执行相同命令续跑。已有终态不重跑。不同模型的输出目录相互独立；本入口不运行我们方法的依赖构图。

已完成：短连接请求成功（最终正文“连接测试成功”，93 total tokens）、106项离线测试、612任务无API预检。未运行正式基线；短测试不能确认最长轨迹的上下文容量。短测试原始记录在 `outputs/aliyun-qwen38-connection-probe/`。
