> 本文为清理前冻结实验的历史记录。结果和验证文件位于相邻 `AgentHound` 仓库；本 clean 分支尚未执行新推理，默认输出改为 state-v2。代码对照见 [PPT_CODE_AUDIT.md](PPT_CODE_AUDIT.md)。

# RootSE 完整迁移实验

用户已授权按顺序完成：构图协议与审计 → 102条LLM构图 → 同图四消融 → 强基线对照 → 机制和复现检查。P0–P4均已完成：5166目标终态，102轨迹×8组排名，48项助手语义盲审及最终复现验证通过；未进行中途定位准确率调参。

## 一键入口

```bash
bash scripts/reproduce_rootse_migration.sh /path/to/moma_ai.key
```

默认输出outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1。第二参数可指定独立新输出目录。复用终态目标，不重新生成失败回答；目录锁阻止重复启动。已有参数、代码或数据指纹不同则拒绝混入本轮。

入口执行锁定环境、83项测试、准备/校验协议、9目标技术pilot、5166目标全量构图、102×4×2离线消融、指标/配对报告和PNG/PDF图。首次构图与pilot成本分开记录。末尾自动生成报告不代表人工审阅和最终复现审计已经完成，应以verification.json和最终过程记录为准。

## 冻结方法与可比性

移动云deepseek-v4.1-flash、temperature0、32768输出/调用、默认thinking、16目标并发，与此前六方法套件相同服务配置；总算力不相同。构图继承原最多8个历史候选选择，给选中字段全文，最大请求503600字符。目标仅含thought/response/action，不读目标observation、根因标签、最终失败测试。完整选中字段不等于完整历史全部候选。旧3000字符前缀协议另存不改。

失败目标保留终态，只造成缺失边；四方法共享成功验证依赖+规则边。无LLM边时明确标记rule_only_fallback，不删除案例。完整图/部分图分开统计，严格完整构图敏感性将部分图整例记0且分母仍102。格式不重复请求，网络/429/5xx最多3次传输尝试；每次attempt和响应usage留档。

主表postmortem输入首条失败测试；trace-only独立输入消融。四版本为baseline、anchors-global、legacy-backtrace、anchors-backtrace，融合权重保持0.50/0.05/0.45。评价统一Tol@0..3，排序另报Top1/3/5与MRR。相对同图三个版本、All-at-Once和Binary的五项预定比较，报告配对计数与Holm，同时仓库配对bootstrap5000次（seed20261006）。这些是开发数据上的探索性结果，不是未见泛化或跨seed稳定性证明。

## 运行与最终检查

- 全量日志：outputs/rootse-migration-console.log。
- 进度：默认输出progress.json及targets/*/*/state.json；不要读取中途命中率来改配置。
- 技术pilot：9/9完整，13验证依赖，9请求，288633 tokens，无重试；原文校验不保证语义正确。
- 冻结配置：experiments/rootse_migration/config.json。
- 冻结指纹：c9654534ce42c555528dab67ea8768f10fb86c2fea329bd600afffcd8c775fca。
- 启动PID与保护：outputs/rootse-migration-launch.json；caffeinate仅持续到该主进程退出。
- 本聊天rootse跟进用于运行期间15分钟静默检查；最终交付完成后暂停，避免重复通知。

全量后使用不含gold/预测命中的analysis/blind_edge_review.json审阅关系语义，记录支持/不支持/不确定及证据。这是助手审阅，不冒充独立人工标签。原始图与预测不因审阅改写。

最终执行：

```bash
.venv/bin/python scripts/verify_rootse_migration.py --api-key-file /path/to/moma_ai.key
```

该命令再执行完整入口，核对正式+pilot目标/调用state和8组rankings哈希不变，请求数不增加，并扫描输出中的实际密钥值（不打印）。检查结果写verification.json，日志verification_reproduction.log。完整总结EXPERIMENT_SUMMARY.md；图与分析在analysis/。

## 2026-10-06 最终验收

主协议新锚点回溯10/102，旧同图全局9/102，All-at-Once56/102。新方案未显示稳健定位收益；Top-3/Top-5/MRR低于旧全局排序。相对旧回溯5/102的变化，Holm p=0.1875，不能宣称可靠提升。

5061目标完整、105输出截断；37条轨迹所有目标抽取完整，65条部分图，全102条保留。正式5205次请求，已知68,130,043 tokens，39次502用量未知；pilot另计9次/288,633 tokens。48项助手盲审已保存，不是独立人工金标准。

完整入口重跑83测试通过，10358个目标/调用state和排名哈希不变，新增请求0；26243个产物密钥扫描无匹配。独立算术复核通过，且八组依赖哈希一致。两张PNG及PDF已检查，未发现遮挡或数字标注错误。

- [完整总结](outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1/EXPERIMENT_SUMMARY.md)
- [边质量审阅](outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1/QUALITY_REVIEW.md)
- [复现验证](outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1/verification.json)
- [独立离线核算](outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1/analysis/independent_audit.json)

推理完成后的离线核算：`.venv/bin/python scripts/audit_rootse_migration_results.py`。盲审判断是已保存的助手审阅记录，可复核但不是脚本自动生成的人类金标准。
