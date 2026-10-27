# 冻结语义表示对照

分支 `codex/attentionmap-semantic-20261027`，基于首版 attention map。协议见 [SEMANTIC_PROTOCOL.md](SEMANTIC_PROTOCOL.md)。

实验以冻结 MiniLM 替换文本 hash，继续比较同输入线性排序、MLP、固定图及三种稠密 attention。通过固定投影保留 128 维，模型可训练参数、初始化、旧数值输入、先验图、5 折及 3 种子相同。编码在本机完成，语义模型不参与反向传播。分块覆盖全文。

```bash
uv venv --python 3.12 .venv-attention-semantic
uv pip install --python .venv-attention-semantic/bin/python -r experiments/attentionmap/semantic-requirements.lock.txt
uv pip install --python .venv-attention-semantic/bin/python -e '.[attention-semantic]'
.venv-attention-semantic/bin/python -m pytest -q
bash scripts/reproduce_attentionmap_semantic.sh
```

需要已有 `outputs/attentionmap-v1` 和其中记录的原始 RootSE/构图/q 缓存。公开编码器下载到 `outputs/models/`，revision 固定。默认 MPS 编码；没有 MPS 时用新输出目录并传 `--encoding-device cpu`。排序器固定 CPU 单线程。

```bash
.venv-attention-semantic/bin/python scripts/run_attentionmap_semantic.py --encoding-device cpu --out outputs/attentionmap-semantic-cpu --stage all
.venv-attention-semantic/bin/python scripts/analyze_attentionmap_semantic.py --out outputs/attentionmap-semantic-cpu
```

主输出 `outputs/attentionmap-semantic-v2/`：

- `experiment.json`、`snapshot/`：训练前固定的协议、源码、模型身份及配置。
- `embeddings/`、`encoding.json`：独立文本缓存、投影向量、完整 384 维向量、token/分块数和耗时。
- `lexical/`、`semantic/`：两组各 90 次训练的特征、独立标签、分组、模型、预测、扰动和 attention。
- `analysis/report.md`、`comparison.json`、`per_case_metrics.json`：统一指标、组 bootstrap、逐例及重放核验。
- `analysis/comparison.png/pdf`：两种表示的成对图表。

重复入口校验身份与哈希后复用已完成编码和训练。旧实验保留不动，词法重跑与旧实验逐例验证。结果解释以统一报告为准。
