# Attention map v1

把固定步骤依赖改为失败条件下的回看权重，保留单步评分分支。完整定义见[PROTOCOL.md](PROTOCOL.md)。

环境：Python 3.12；安装本项目的attention可选依赖即可训练。完整本机依赖版本见requirements.lock.txt。现有`.venv-attention`已经可用，原`.venv`保留。

```bash
python3.12 -m venv .venv-attention
.venv-attention/bin/python -m pip install -e '.[attention]'
.venv-attention/bin/python scripts/run_attentionmap.py --stage prepare
bash scripts/reproduce_attentionmap.sh
```

实验复用相邻`TrajAudit/RootSE`、`outputs/rootse-ours-qwen38-nothinking-causal-v1`和`outputs/rootse-influence-qwen38-nothinking-v3`。默认输出`outputs/attentionmap-v1`；已有完成的折会校验后复用。输入/源码变化会拒绝混用旧实验，另指定新输出目录运行。

主要文件：
- `features/`：不含标签的NPZ；`labels.json`：独立训练/评估标签。
- `runs/方法/seed-*/fold-*/`：模型权重、训练/测试名单、损失、时间、OOF分数。
- `maps/`：每个attention变体、种子、轨迹的A/seed/mass。
- `summary.json`：原始汇总；`analysis/report.md`：核验报告与历史baseline回退口径。
- `analysis/verification.json`：独立指标复算、分组隔离、来源哈希和矩阵检查。
- `analysis/comparison.png`、`analysis/attention_example.png`及对应PDF：可用于汇报。

本轮102条、35组、5折、3种子、6模型，90次训练已经完成。Attention Top-1=19.93%、Top-5=55.23%；同输入MLP=18.30%/49.67%，本轮RRF=26.47%/49.02%。首版采用固定词法hash表示；下一轮固定语义编码与相同对照，检验表示质量和关系学习的各自贡献。

普通`q_only`和`old_weighted`汇总是直接按缓存原始分数重排，历史实现则保留全零回退排名。正式历史对比使用analysis/report.md中的21/102和10/102。
