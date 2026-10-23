# 决策影响度实验，2026-10-13

一句话：对固定依赖图估计直接决策影响强度，并按权重传播失败信号，检查边权是否改善根因定位。

只新增自有代码，复用上一轮Qwen非思考图：102轨迹、4467合并边。外部baseline、原始图、上一轮裁决和结果不修改。当前调用模型qwen3.8-flash，temperature=0，stream=true，enable_thinking=false，32768输出预算，16并发。

推理拆成两个独立阶段：weights仅看任务与局部批次（至多8个目标及它们的全部既有父节点，字段全文），对固定边估计0–1连续直接影响，不能读取最终失败测试；共525批。nodes每轨迹一次，看任务、轨迹和已提供的失败测试，对每步判定新引入/继承/无/不确定错误及明确恢复状态，不读取图或旧预测。两者都不读取根因标签。返回引用字段ID，由程序验证ID存在，不复制长引文；引用ID存在不能证明理由正确。

正式运行前，按请求字符数选择最短、中位、最长三条技术pilot，不计算金标命中。v1六次pilot因五档值/ID协议失败；v2改连续值与局部批次，14次pilot仍有少量漏项。v3在查看金标命中前采用逐条验证，保留通过的行，缺失/重复/非法条目显式中性回退；v2请求与响应原样导入，由传输层验证请求哈希完全相同后离线复用。技术门槛为全部请求JSON可解析，两个阶段的条目覆盖各>=95%，不按命中选择。正式627逻辑调用，传输重试最多3，格式失败不重试。原始技术历史、模型响应和usage全部保留，导入记录只计一次费用。

主对照weighted vs equal。equal对每条轨迹全部边赋该轨迹已估计权重的平均值，与weighted保持边权总量相同。另保留固定0.1诊断组，不能把新equal组称为旧PageRank。传播采用Pij=wij/max(1,sum_k wkj)、gamma=1，在DAG上逆拓扑计算R=f+PR。f沿用上一轮结构化锚点，不根据本轮标签重选；不再混合TF-IDF，不加最终模型裁决。无锚点/全零时回退旧锚点排名并单独统计。

节点错误评分q只保留introduced且未明确recovered的score。q-only、q-equal、q-weighted、q-shuffled共享同一q；次对照为q-weighted vs q-equal和q-only。若只有q-only有效不能把收益算到传播上。缺失/无效的权重条目用0.5，缺失/无效节点q=1（仅乘法中性，不是确信其错误），不丢弃同批合法条目；无效JSON整批使用同规则。失败均保留102分母；所有回退与条目覆盖显式记录。

打乱对照在每个案例内随机交换全部边权，保留拓扑和边权多重集合；100个预定seed（20261013..20261112），正式固定展示seed20261013，同时报告完整分布。它们不是100次独立模型运行，不能用于伪造跨模型运行mean±std。

主指标为全量精确Top1。三组事前配对对比用exact McNemar和Holm，因轨迹按仓库相关，p值仅探索性；按仓库bootstrap5000次，seed20261013。数据已经参与开发，单次运行，不声称独立泛化。该实验只检验日志推测的权重，未执行真实决策干预，不能声称测得因果效应。

运行入口：

```bash
caffeinate -i .venv/bin/python scripts/run_rootse_influence.py --stage all --api-key-file /Users/linzheyuan/code/selfbase/aliyun_maas_qwen38.key
.venv/bin/python scripts/report_rootse_influence.py
../AgentHound/.venv-rootse-analysis/bin/python scripts/plot_rootse_influence.py
```

正式输出outputs/rootse-influence-qwen38-nothinking-v3，源码/配置/图/数据哈希冻结；历史文件保护清单保存在preservation_before.json。运行前146测试通过。指纹a6326dd5178cd994b884b33c711b6d697083755a1626f83e93377a210e6a973b。
