# DeepSeek 官方并发运行

独立复制入口 `scripts/run_rootse_deepseek.py`、工作进程 `scripts/run_rootse_deepseek_worker.py`、请求层 `scripts/deepseek_suite_transport.py`。原 Qwen 脚本及配置没有修改，复用已安装依赖，不安装或升级共享虚拟环境。

接口为 https://api.deepseek.com/chat/completions，模型 `deepseek-flash`。配置文件为 `experiments/rootse_suite/config_deepseek_official_flash.json`：默认 16 个并发工作进程，每个进程顺序发送该方法所需请求；显式开启 `thinking.type=enabled`、`reasoning_effort=high`，输出预算 32768 tokens。密钥从指定文件读取，不进入配置、请求快照或源码。

在此项目目录执行：

```bash
# 两个最大轨迹：完整输入，All-at-Once 方法，并发 2
bash scripts/run_rootse_deepseek.sh /Users/linzheyuan/code/selfbase/deepseek_official_flash.key --limit 2 --methods all_at_once --workers 2

# 全量：102 条轨迹 × 原有六种方法，默认并发 16
bash scripts/run_rootse_deepseek.sh /Users/linzheyuan/code/selfbase/deepseek_official_flash.key --workers 16
```

试跑和全量分别使用 `outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1-smoke-2/` 与 `outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1/`。重复相同命令可以恢复中断工作，并回放已保存请求；已记录的终止失败不会被静默重试。变更配置或代码后应通过 `--out` 使用新目录，协议指纹会阻止混合不同设置。

每次运行生成 `deepseek_summary.json`，记录实际样本数、每条请求的 token 用量、`finish_reason`、输出字符数、推理字符数和重试详情。流式原文及合并响应保存在 `calls/`。`finish_reason=length`、上下文超限、缺失流结束标记、空最终答案均不会被算作有效结果；没有静默裁剪 All-at-Once 输入，也没有自动增加输出预算。其余方法保留原算法行为（包括 agent_locate 原有上下文处理）。

试跑只覆盖两个样本的 All-at-Once，不代表六种方法均已通过全量运行。完成全量后可用原只读报告器生成准确率报告：

```bash
.venv/bin/python scripts/report_rootse_suite.py --out outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1 --source ../TrajAudit
```

接口参数参考：[DeepSeek 官方文档](https://api-docs.deepseek.com/)。

实测（2026-10-07，北京时间）：两个最大文件的 All-at-Once 请求均为 HTTP 200、`finish_reason=stop`、预测状态 `valid`；输入 tokens 为 193317 / 173353，输出 tokens（含推理）为 1142 / 2227，耗时 8.22 / 13.46 秒。请求时间区间确实重叠。两个样本未出现上下文超限或输出截断。完整结果见试跑目录 `deepseek_summary.json`。离线请求层及原 Qwen 回归测试 15 项通过。
