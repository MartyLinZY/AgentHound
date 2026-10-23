# RootSE 全量 baseline 复现

当前按用户要求使用移动云 `deepseek-v4.1-flash`：官方支持1M，部署实测接收999030输入tokens。执行 `bash scripts/reproduce_rootse_million.sh /path/to/moma_ai.key`，结果保存至`outputs/rootse-full-deepseek-v41-flash-v1/`；该命令完成推理、论文报告、已有14B结果的配对对照及百万容量汇总。

## 首轮14B记录与通用协议

首轮固定实验为 All-at-Once，数据覆盖完整102条。模型使用移动云 MoMA 的 `Qwen2.5-14B-Instruct-1M`，不是此前8条试跑所用的Qwen3.5-27B。模型选择仅依据上下文探测：最长输入840125字符、185575输入tokens成功返回。探测记录在 `outputs/moma-context-survey/`，没有依照定位准确率筛模型。

在仓库根目录执行一条命令：

```bash
bash scripts/reproduce_rootse_full.sh /path/to/moma_ai.key
```

本机实际密钥文件路径由用户提供，脚本仅在进程内读取其内容，不复制进仓库或命令行。密钥文件为单行纯文本。需要Python 3.12或uv、Git、网络及该模型的访问额度。脚本建立/复用`.venv`，安装锁定依赖、运行测试，在缺少TrajAudit目录时自动克隆并切到固定提交，然后执行推理、评估、生成报告。

默认输出：`outputs/rootse-full-qwen25-14b-1m-v1/`。同一目录重跑是断点续跑，已完成案例不发送新请求。需要新的独立运行时传第二个参数：

```bash
bash scripts/reproduce_rootse_full.sh /path/to/moma_ai.key outputs/rootse-full-independent-run2
```

脚本不会读取准确率来决定重试、改提示词或换模型。推理部分只保存预测和请求状态，结束后独立脚本读取标签评估。完整输入不截断，无gold patch；上下文错误、格式错误等保留在102条分母。温度0不保证远端权重/运行环境永远不变，也不能保证再次请求逐字一致。

配置在 `experiments/rootse_full/config.json`；数据逐文件哈希在`dataset_manifest.json`；论文表格的结构化引用在`paper_reference.json`。运行前冻结配置、代码、数据与提示词指纹。不要直接修改已经开始的实验配置；协议变化应建立新的配置和输出目录。

推理采用官方纯JSON提取函数，不沿用pilot的引号恢复。每条首个有效响应即结束；JSON无效不重试。网络超时、429/5xx有限重试，共不超过3次，原作者脚本超时/限流上限为5次。失败和未知用量都保留；旧pilot结果和模型探测不计入正式运行指标。

只重算报告、不调用模型：

```bash
.venv/bin/python scripts/report_rootse_baseline.py \
  --source ../TrajAudit \
  --out outputs/rootse-full-qwen25-14b-1m-v1
```

只构造全部输入、验证数据完整性：

```bash
.venv/bin/python scripts/run_rootse_full_baseline.py --prepare-only
```

产物包括：

- `REPORT.md`：结果、论文对照、分组结果、用量与限制。
- `metrics.json`、`predictions.csv`、`evaluation.json`：全量评估、逐条预测、标签对照。
- `paper_audit.json`：重新计算作者公开CSV指标，显式报告重复行。
- `manifest.json`、`snapshot/`：冻结配置、源文件与哈希。
- `requests/`、`responses/`、`states/`：每条实际输入、原始返回与重试/故障过程。
- `evaluation_provenance.json`：评估脚本哈希。

论文对照使用[TrajAudit v3](https://arxiv.org/abs/2605.26563v3)的without-reference设置。论文主骨干为Claude Sonnet4.5，本轮不调用Claude，只读取作者公开结果。模型不同，差异不能作为同模型方法优劣的证据；论文Qwen3-235B的40.2%是TrajAudit结果，不是All-at-Once。

过程记录：`experiments/rootse_full/PROCESS.md`。

## 更强模型对照

使用Qwen3.5-397B-A17B并与14B逐条比较，执行：

```bash
bash scripts/reproduce_rootse_strong.sh /path/to/moma_ai.key
```

强模型结果位于outputs/rootse-full-qwen35-397b-a17b-v1，配置为experiments/rootse_full/config_qwen35_397b.json。该入口在14B结果缺失时也会先跑14B；已有结果会复用。比较脚本验证全部请求除model外相等，并输出MODEL_COMPARISON.md/json与逐案例CSV。Qwen3.5默认思考，因此实际算力开销不同，即使输出预算相同也不能视为只改变参数规模的实验。


## 百万上下文 DeepSeek-V4.1-Flash

官方模型标称1M，MoMA部署探测见outputs/probes/deepseek-v41-flash-*.json。

```bash
bash scripts/reproduce_rootse_million.sh /path/to/moma_ai.key
.venv/bin/python scripts/compare_rootse_million.py --baseline outputs/rootse-full-qwen25-14b-1m-v1 --candidate outputs/rootse-full-deepseek-v41-flash-v1
```

使用固定102条及官方All-at-Once提示词，temperature=0、max_tokens=8192、3并发。runner v2仅修改允许调用的模型，旧runner不变。容量测试是接收长度验证，不代表长文本检索/诊断能力。

移动云容量实测：第二次合成输入请求HTTP200，输入999030 tokens，输出1 token；首次502，usage未知。该结果确认接近百万输入可接收，不证明其能有效利用所有上下文。
