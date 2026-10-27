# Attention map Top-1 原因与 baseline 对照（2026-10-27）

本轮对冻结模型做离线回放和分支干预，没有新增训练或 API 请求。

当前语义 dense Top-1 28.43%，Top-5 74.84%。平均每种子 47.33/102 条真根因位于第 2–5 名，占 Top-1 错误的 64.84%。68 条在三个种子中都错，38 条始终在第 2–5 名。

训练 Top-1 99.92%，分组测试 28.43%；独立 MLP 为 95.92% 对 24.51%。93.15% 的最终错误在联合训练模型的单步分支就已出现。关系分支平均纠正 7.67 条、改错 5 条，净增加 2.67 条；3 条在所有种子中都被关系分支改错。错误预测平均早于根因 41 条、晚于根因 32 条。

`failure_key` 向各 key 加同一个失败向量，其贡献是一行内的常数，被 softmax 抵消。对 306 份折外预测置零后 Top-1 改变 0 次。这个问题应重写条件化方式；删除冗余项不会直接提升现有结果。单步分支、seed 和 query 分支仍包含失败信息。

| 方法 | Top-1 |
| --- | ---: |
| 原影响度加权传播（含回退） | 9.80% |
| 原 q-only（含回退） | 20.59% |
| 固定 RRF | 26.47% |
| 原词法 attention | 19.93% |
| 语义 MLP | 24.51% |
| 语义固定图 | 27.45% |
| 语义 attention | 28.43% |
| 关系不注入失败的语义 attention | 29.74% |

| 完整诊断方法 | Qwen3.8-Flash thinking | 官方 DeepSeek Flash thinking |
| --- | ---: | ---: |
| All-at-Once | 47.06% | 55.88% |
| Step-by-Step | 33.33% | 37.25% |
| Binary Search | 24.51% | 54.90% |
| TRAIL | 43.14% | 50.98% |
| FAMAS（本地适配） | 28.43% | 35.29% |
| TrajAudit（generic） | 50.98% | 34.31% |

均以同一批 102 条为分母，失败输出计错。学习方法是三种子分组折外均值。历史 LLM 采用完整诊断协议、开启 thinking、每次 32K 输出上限；attention 采用冻结 MiniLM 和既有非思考图/q 缓存，两者不是同模型同总预算结构消融。

Qwen All-at-Once 正确的 48 条中，每个种子都有 40 条的真根因已经在 attention Top-5 里。其根因被候选覆盖但未排第一的有 23–27 条。原始轨迹核对发现正确分析/错误补丁混淆、错误需求范围判断与后续实施混淆、关系分数覆盖局部证据、精确字符串契约被忽略等案例。

下一轮优先固定候选器，用需求、测试断言、原始补丁和修复轨迹做候选裁决；同一 Qwen 配置比较完整轨迹、候选重排、候选加关系证据。同步用训练折内验证控制过拟合，再测试失败与步骤的交互及关系门控。新任务用于独立验证。

复算核验：1,836 份保存模型预测、1,836 份 LLM baseline 预测、204 份历史 q/传播完整排名；2,328 份诊断来源文件与新增 205 份历史文件哈希未变。历史完整方法保留回退，不能用仅原始分数的 20/102、7/102 替代其 21/102、10/102。

复现：

```sh
.venv-attention-semantic/bin/python scripts/diagnose_attentionmap_top1.py
.venv-attention-semantic/bin/python scripts/report_attentionmap_top1.py
```

[完整报告](../../outputs/attentionmap-top1-diagnosis-20261027/analysis-report.md) · [统计附录](../../outputs/attentionmap-top1-diagnosis-20261027/stats-appendix.md) · [案例核对](../../outputs/attentionmap-top1-diagnosis-20261027/case-review.md)
