# Who&When 迁移实验补充协议（模型结果产生前）

使用 Kevin355/Who_and_When revision 59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c 中的原始 JSON，126 + 58 = 184 条。该 revision 的 Parquet 和 JSON 有 3 处标签差异：Parquet 的 mistake_step 超过 history 长度，原始 JSON 的标签有效。以发布方原始 JSON 标签为准，不自行修标签，保存两种发布文件及逐条审计。代码核对官方枚举从 0 开始，所有 history 条目均计步。该 HF 发布 split 叫 train；本项目没有用它训练或调参，本轮作为冻结方法的外部迁移评测。

模型输入只取 question、history.content/name/role。工具终端内容映射 observation，thought 角色映射 thought，其余映射 response；角色元数据存 state。所有字段保留原文。mistake_step、mistake_agent、mistake_reason、ground_truth/groundtruth 不进入 prompt 或 attention。只提供已知“最终失败”状态；没有测试错误时不构造虚假失败信息，图的语义 query 用任务向量（failure fallback=task）。这不同于原作者提供任务标准答案的部分 baseline，只比较本轮同输入的三个方法，不直接宣称复现其论文分数。

同一次准备冻结 full / units_aligned / balanced 的全部请求。图沿用冻结 MiniLM 编码与 v1 规则，原始 dependency 列表为空；attention 仍包含相邻、路径、符号、语义边。full 使用完整按时间排序的轨迹；两个有预算方法使用 16384 token 和相同原生诊断提示。方法实现不因该数据结果调整。关联算法无训练参数，编码 CPU，无额外 LLM 建图成本。

先审计原始标签范围与字段完整性，再冻结请求。按 question_ID 或完全相同规范化问题的连通组计算区间；检查与 RootSE 的任务/轨迹精确重复，以及外部内部重复。单独报告 Algorithm-Generated、Hand-Crafted、步数 ≤5 / 6–20 / >20；Top-5 的均匀随机参照是 min(5,n)/n，避免短轨迹使指标虚高。仅能据此论证跨任务诊断迁移，不能据此替代独立 coding 数据评测。

数据预审发现两对完全相同 history 对应不同 question，共四条记录。保持发布的 184 条全集主表；额外报告剔除这四条的 180 条质量敏感性结果。分组再并入相同 history 的记录，避免把复用轨迹当成独立样本。此规则在任何外部模型请求前确定。

主要比较 balanced−units_aligned 与 balanced−full 的 Top-1/Top-5；与 RootSE 的四项主要检验合并做 Holm 校正。其余差异是描述性分析。方法全部运行后再读取标签计算效果。
