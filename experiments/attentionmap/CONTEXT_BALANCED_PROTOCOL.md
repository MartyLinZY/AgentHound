# v4：决策覆盖、证据深度与跨数据集迁移

冻结前定义，2026-10-29。RootSE 是已经反复分析过的开发集，不能作为未见测试集。

RootSE 102 条，每条同时运行三条件：units_repeat（v3 units 原请求精确重跑）、units_aligned（原事实，统一原生标注目标并引导错误链比较）、balanced（相同 aligned 提示，改变上下文预算分配）。三条件均只调用一次诊断模型，返回严格五候选。102×3 请求按固定随机顺序调度。

balanced 保持 v1 图、attention、priority、overview、节点 2736 token 原文原子单元上限不变。先在总上下文 65% 内按原 priority 放入各节点的完整 thought/response 单元；工具节点以首个完整动作/输出单元作为 core。随后按原 priority 补充完整动作/输出，剩余空间补入能容纳的其他 core。总预算 16384 cl100k token，计入 overview、字段包装和链接；共同任务/标注指南不计入。65% 是本轮预先指定的设计值，不做测试集调参。

aligned 去掉额外添加的 earliest uncorrected 约束，以数据集原始标注指南为目标；要求区分最初决定、沿用、执行、纠正与重引入，并比较竞争解释。Top-5 不扩大，不按标签调整排序或加分。

外部主要数据采用公开 Who&When，完整 126 Algorithm-Generated + 58 Hand-Crafted = 184 条。所有转换只读取 question 和 history 的白名单字段；mistake_step/agent/reason、任务 ground truth 严格隔离。原生目标、step 编号先核对发布代码。外部比较 full、units_aligned、balanced；使用同一模型配置及原生任务说明。图用冻结 MiniLM + 原 v1 的相邻/共享路径/符号/语义边，没有 RootSE 专用依赖缓存；因此报告跨任务、跨结构迁移，单独说明图源变化。外部结果不用于回改本轮方法。

所有条件复用 qwen3.8-flash，temperature=0，thinking=true，max_tokens=32768，16 workers；仅网络/429/5xx 最多三次重试，格式错误记失败，不按答案重试。原始请求、SSE、响应、usage、配置和代码哈希保存。无搜索工具；只能解释历史轨迹中的搜索，不会执行其中的工具。

主指标 Top-1、Top-5、MRR@5；RootSE 按原 35 任务/仓库组，外部按 question_ID 联合重复问题分组，5,000 次整组 bootstrap，20,000 次整组配对 sign-swap；本轮主要 contrasts 的 Top-1/Top-5 统一 Holm 校正。报告全集（无效请求记零）和共同有效交集。额外报告根因展开率、完整决策覆盖、原文 token、provider usage、延迟和失败数。外部原生标签不与 RootSE 合并平均。

仅在标签隔离、请求/原文跨度、精确 step 映射和技术 pilot 通过后执行正式请求。保留所有旧结果与历史哈希。无本地训练。
