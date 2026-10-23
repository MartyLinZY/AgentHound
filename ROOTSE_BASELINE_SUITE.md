# RootSE 全部方法实验

当前主实验使用移动云deepseek-v4.1-flash，对All-at-Once、Step-by-Step、Binary Search、TRAIL、FAMAS与TrajAudit各执行完整102条。模型服务已实测接收999030输入tokens。全量结束后才离线读取根因标签计分。

在仓库执行：

```bash
bash scripts/reproduce_rootse_suite.sh /path/to/moma_ai.key
```

默认产物：`outputs/rootse-suite-deepseek-v41-flash-32k-v1/EXPERIMENT_SUMMARY.md`。需要独立新一轮时传第二个输出目录参数；默认重跑复用已有终态案例及全部中间响应。旧8192预算结果不覆盖。

固定配置在experiments/rootse_suite/config.json，依赖锁在同目录requirements.lock.txt。源仓库提交、102条数据、714份参考轨迹、方法代码、模板及适配器代码均记录哈希；任何推理协议变化需要新输出目录。模型统一默认思考，temperature=0，max_tokens=32768，每次连接读取超时900秒，最多3次传输重试，16个案例进程并发。该输出预算覆盖官方方法各自原设置，旧All-at-Once也在此预算下独立重跑。

算法调用官方函数，保留各自的解析/回退、动作摘要、参考轨迹选择和调查步数。统一传输层对未完成回答（finish_reason非stop）中止该案例，保留失败分母；不用残缺JSON/思考里的猜测凑预测。TRAIL的原enable_thinking=False不发送，以统一默认思考模式。原始调用参数和实际请求都留档。

每次调用先保存request，再记录attempt，后存response。恢复时从方法起点重放已保存响应，哈希不同立即拒绝。中断时没有response的在途调用计unknown，重试次数仍受上限约束。不要手动修改calls文件或直接删除失败样本来重跑。

进度可查看各方法案例state.json与outputs/rootse-suite-console.log；不读取中途准确率来调参。最终报告包括Tol@0..3、五项对All-at-Once的配对检验和Holm校正、资源/失败、轨迹来源分组、公开CSV复核及复现限制。当前无需LLM裁判，不调用Claude，也不报告Justification Accuracy。

只准备协议不请求API：

```bash
.venv/bin/python scripts/run_rootse_suite.py --prepare-only
```

全部612个任务完成后，可离线重建报告：

```bash
.venv/bin/python scripts/report_rootse_suite.py --source ../TrajAudit --out outputs/rootse-suite-deepseek-v41-flash-32k-v1
```

FAMAS参考轨迹审计及选中轨迹哈希见experiments/rootse_suite/reference_audit.json；所有案例都有参考，44例无成功参考。TrajAudit使用generic版本，开启诊断与观察折叠，调查最多5次。这些输入和实现差异会在最终结论中保留。

## 已完成的 v1 复核与图表

六方法各102条已完成。完整人工审阅总结：outputs/rootse-suite-deepseek-v41-flash-32k-v1/2026-10-06--rootse-baselines--r1--round-review.md。

一条命令复用冻结实验并重建本轮报告、图表：

```bash
bash scripts/reproduce_rootse_review.sh /path/to/moma_ai.key
```

该入口单独安装锁定绘图依赖，不改变推理环境；限定本轮v1目录。analyze_rootse_suite.py只做离线审计与绘图，发现结果不符合本轮叙述会拒绝生成，避免把本轮结论套用到新实验。独立新实验仍用原suite入口和新输出目录。

新增图表及审计在analysis/，原始预测从未修改。缓存复核612个案例和2777个调用state哈希不变，新增请求0；全部75项测试通过。论文CSV差异、输入差异、失败用量和单轮统计限制均保留在报告。
