# Qwen 关闭 thinking：自有方法的故障语义裁决

一句话：在依赖图排序后，结合失败测试和完整轨迹证据，选出最早引入且未被修正的决定性错误。

仅新增我们的方法。外部六基线的代码、配置、预测均保持冻结。

模型沿用阿里云部署 qwen3.8-flash，全部请求显式 enable_thinking=false，stream=true，temperature=0，32768输出预算，16并发。重新构建5166目标的图，并对102轨迹各做一次最终裁决。历史图不用于本次推理。

裁决读取完整的任务、首条失败测试和每一步的决定/动作/观察，以及同图三种排序的候选和依赖关系；可选任意原步骤，不限制Top-5。要求引用所选步骤与失败测试原文。引用只能证明文本存在，不构成已验证的因果关系；没有反事实重放。无效裁决回退到原锚点排序并公开失败数。

主对照为同一批Qwen非思考图上的旧锚点排序；旧全局作为第二对照。全部102分母，Tol@0主指标，Tol@1..3补充。事前固定提示词、校验和回退规则，不按本轮命中调参。与历史Qwen外部基线的比较只是描述性，因为其thinking开启。本轮用于验证工程改动，不是独立未见测试集上的泛化证明。

运行（复用已完成终态，不重新请求）：

```bash
caffeinate -i .venv/bin/python scripts/run_rootse_qwen_ours.py --stage all --api-key-file /Users/linzheyuan/code/selfbase/aliyun_maas_qwen38.key
.venv/bin/python scripts/report_rootse_qwen_ours.py
```

技术pilot只检查协议和引用，不使用金标命中选择协议。先9个构图pilot（沿用原组别/长度选择），再全量构图，最后最长裁决输入的单例技术pilot与102条正式裁决。所有新增请求均关闭thinking。

默认结果目录：outputs/rootse-ours-qwen38-nothinking-causal-v1。调用日志、完整响应、usage、输入及源码快照都保留。输出成本分别报告正式图、图pilot、正式裁决、裁决pilot。

2026-10-12全量完成：131项测试通过；5166构图目标中5097完整、66格式失败、3服务错误；102裁决中52有效、50按预定规则回退。全部5278次请求（含pilot）thinking=false，响应reasoning_content总长度0。同图旧锚点7/102，新裁决19/102；新增16、丢失4。35条有效但错的裁决中32条偏晚，预选六个共同错例仍全错，因此不能宣称已解决最早错误定位。

缓存离线复查：102裁决复用终态，无新增请求，26486个原始记录及排名文件哈希不变。外部基线7468文件哈希不变。未放宽引用校验、未根据本轮标签调提示词。详细结论见结果目录的 analysis/检查结论.md；统计、成本、原始响应、配对CSV和两张图均已保存。

离线诊断与绘图：

```bash
.venv/bin/python scripts/audit_rootse_qwen_judge.py
../AgentHound/.venv-rootse-analysis/bin/python scripts/plot_rootse_qwen_ours.py
```
