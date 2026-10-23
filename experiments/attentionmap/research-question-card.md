# Research Question Card

Question：失败条件化的可补边关系学习，能否提高Agent轨迹根因排序，尤其是旧图回溯不到的根因？
Type：exploratory。
Hypothesis：保留单步证据，同时让失败相关的后续步骤向所有更早步骤分配诊断权重，能够修复固定图缺边导致的排序损失。
Current evidence：10月13日固定Qwen图的102条实验中，q-only命中21，q×weighted命中12；54条根因传播分数为0。对应原始features与rankings位于outputs/rootse-influence-qwen38-nothinking-v3，实施时逐例复算。
Missing evidence：同输入可学习排序对照、仓库分组评估、关系扰动和去图实验。
What would support it：attention优于MLP，零传播根因子集恢复更多，关系扰动带来一致下降。
What would falsify it：MLP效果相同或更好，关系扰动没有影响，收益只来自局部评分。
Minimal next action：按PROTOCOL.md固定特征、分组、超参数与种子，完成102条交叉验证并保存逐例OOF预测。
Decision：run experiment。
Evidence：ER-20261023-influence-cache-01（既有实验原始features/rankings）；ER-20261023-graphormer-02（全文方法§3）；ER-20261023-graphtracer-03（全文方法§3–4）。
