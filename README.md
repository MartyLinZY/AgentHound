# LogHound-Agent-Debug

本目录用于把 LogHound 的静态故障定位思想迁移到 coding-agent 的失败轨迹根因定位。

当前目标是：输入一条已经失败的 agent 执行轨迹，输出所有执行步的可疑度排名；第一阶段先在已有根因步标注的数据上验证可行性，再决定是否为 SWE-bench Pro 轨迹开展人工标注。

## 当前内容

- `docs/DESIGN.md`：问题定义、LogHound 迁移映射、算法设计、实验设计和实施路线。
- `data/README.md`：数据目录规范、已核验的数据事实和防泄漏规则。
- `data/raw/agentracer/tracertraj-code-test.parquet`：本地已有的 AgenTracer coding 测试集副本。
- `src/loghound_agent_debug/`：统一 schema、AgenTracer adapter、证据抽取、TF-IDF、反向 PPR 图和评估器。
- `configs/`：预注册的固定参数与 clue 规则。

## 设计结论

在具有 step-level 失败观察的轨迹上，初版采用完全离线的三路静态证据融合：

1. 失败证据与执行步的文本相关性；
2. 显式错误、工具失败和终止症状等轨迹线索
3. 从失败症状沿轨迹依赖图反向传播得到的上游因果分数。



## 运行实验

所有 Python 命令使用项目约定的 `mybook` 环境。一条命令完成数据转换、A/B/C 审计、排名、指标和 A 档案例报告：

```bash
conda run -n mybook python scripts/run_experiment.py
```

如需单独重跑后处理：

```bash
conda run -n mybook python scripts/evaluate.py
conda run -n mybook python scripts/case_report.py
```

输出位于 `outputs/`（已被 git 忽略），统一输入和标签分别位于
`data/processed/agentracer/inputs.jsonl` 与 `labels.jsonl`。排序器只接受前者；评估器才会同时读取标签。

本地数据审计固定得到 A/B/C=`9/31/87`。A 档是有硬失败回传的主验证集，B 档是 agent 陈述式弱证据，C 档无负向观察，只运行 task-only semantic 对照，不解释为完整方法效果。

详细方案见 [docs/DESIGN.md](docs/DESIGN.md)。
