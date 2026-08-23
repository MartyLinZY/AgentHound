# 实验日志

## 2026-08-23：P0/P1 首次可复现实验

- 数据：本地 AgenTracer coding test，127 条；不读取 `ground_truth`、`mistake_step`、`mistake_agent` 或 `mistake_reason` 参与排序。
- 证据审计：A 硬失败 9 条，B 弱陈述 31 条，C 无证据 87 条，符合设计基线。
- 固定配置：`configs/default.yaml`；clue 规则版本 `0.1.0`；TF-IDF 任务权重 0.4；PPR alpha 0.85；融合权重 0.12/1.0/1.0。
- specificity：主排名使用 singleton 模式（每步为 1），同时报告跨轨迹模板统计；本次模板 singleton ratio 为 0.959742。
- 产物：`outputs/rankings/`、`outputs/metrics/`、`outputs/cases/`。

这次运行的完整指标保存在 `outputs/metrics/summary.json`，逐条排名保存在
`outputs/rankings/rankings.jsonl`。输出目录不提交到版本库，使用 README 中的命令即可重建。
