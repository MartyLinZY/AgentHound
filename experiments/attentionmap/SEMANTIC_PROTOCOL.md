# 冻结语义表示与 Attention Map 对照

2026-10-27，semantic_v2；基于 attentionmap_v1，首轮代码提交 4a82c4e。

问题：把每一步的词法 hash 换成冻结语义表示，是否改善跨仓库根因排序？使用同一表示的线性模型、MLP 和 attention 的差异，区分表示收益与关系收益。

表示：使用 sentence-transformers/all-MiniLM-L6-v2，固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。编码器参数冻结、eval 模式，在本机执行。按 tokenizer token 切为连续不重叠的 254-token 块，加特殊标记后最多 256 token；每块 mask-aware mean pooling、L2 归一化，再按有效内容 token 数加权合并并归一化。覆盖所有块；空文本为零向量。

将 384 维向量乘固定的 384×128 随机正交投影（NumPy PCG64 seed=0、QR、固定对角符号），再 L2 归一化。投影与标签、数据分布无关。128 维使新旧排序器具有完全相同的参数形状和初始化；本轮检验的是这一具体编码与压缩配置。

输入字段不变：decision/action、observation、任务与首条失败测试；四组文本/交互特征替换为语义表示，数值特征、位置、长度、图先验、q 及缺失标记逐元素保持。标注只用于训练折损失和评价。模型架构保持首版，包括失败条件 Q/K、可补边、单步残差与六种变体。

固定原实验的 102 条、5,268 步、35 组、5 折、种子 17/29/43、40 epochs、hidden=32、AdamW lr=0.001、weight_decay=0.001、batch=8、clip=1、CPU 1 thread。运行两组表示各 90 次训练；词法组重新训练用于核对旧结果一致性。无验证集调参、无早停、无按本轮成绩改配置。

主比较：semantic dense 与 lexical dense；semantic dense 与 semantic linear/MLP。补充固定图、无失败条件、无图先验、测试时 attention 打乱、删除 50%/100% 先验边。RRF 沿用原实验相同输入的固定分数。报告各种子 Top-1/3/5、MRR、配对新增/丢失命中、35 组 bootstrap 95% 区间（2,000 次、seed=20261027），并报告旧传播为零的 54 例。

记录编码器权重/源码/原始输入/特征/划分哈希、实际文本与块数量、编码耗时、排序器训练耗时、参数量；保存每折模型、逐例预测、attention 矩阵和重放结果。完整保留 outputs/attentionmap-v1，输出写入 outputs/attentionmap-semantic-v2。

这是既有 RootSE 的分组开发实验；当前对照衡量冻结表示变化。原 RQ 中的新任务外部评估、受控修改验证关系以及重新生成 LLM 特征的等总预算实验仍为后续工作。

模型与 pooling 来源：[官方模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)。分块、投影和对照设计为本轮实现。
