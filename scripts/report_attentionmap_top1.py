#!/usr/bin/env python3
"""Render the offline Top-1 diagnosis from verified prediction records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_attentionmap_semantic import measure, require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/attentionmap-top1-diagnosis-20261027"
SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
METHODS = {
    "all_at_once": "All-at-Once", "step_by_step": "Step-by-Step",
    "binary_search": "Binary Search", "TRAIL": "TRAIL",
    "FAMAS": "FAMAS（本地适配）", "agent_locate": "TrajAudit（generic）",
}
VARIANTS = {
    "linear": "语义线性排序", "mlp": "语义 MLP", "fixed_graph": "语义固定图",
    "dense_unconditioned": "语义 attention：关系不注入失败",
    "dense": "语义 attention", "dense_no_prior": "语义 attention：无图先验",
}


def read(path: Path):
    return json.loads(path.read_text())


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main() -> None:
    d = read(OUT / "diagnosis.json")
    comparisons = read(OUT / "comparisons.json")
    baseline = read(OUT / "baseline_summary.json")
    predictions = read(OUT / "case_seed_diagnostics.json")
    packets = read(OUT / "selected_cases.json")["cases"]
    original = read(ROOT / "experiments/attentionmap/semantic-results.json")["methods"]
    labels = read(SOURCE / "labels.json")
    cases = {c["case_id"]: c for c in read(SOURCE / "manifest.json")["cases"]}
    fixed_scores = read(SOURCE / "baselines.json")
    fixed = {}
    for name in ("q_only", "old_weighted", "rrf"):
        metrics = [measure(scores[name], labels[cid]) for cid, scores in fixed_scores.items()]
        fixed[name] = {key: float(np.mean([r[key] for r in metrics])) for key in ("top1", "top5", "mrr")}
    require(fixed["rrf"]["top1"] == original["fixed/rrf"]["top1"], "RRF summary differs")
    score_only = dict(fixed)
    historical_sources = {}
    influence_dir = ROOT / "outputs/rootse-influence-qwen38-nothinking-v3"
    influence_metrics = read(influence_dir / "metrics.json")
    historical_sources[str(influence_dir / "metrics.json")] = hashlib.sha256((influence_dir / "metrics.json").read_bytes()).hexdigest()
    historical_details = {}
    for key, method in (("q_only", "q-only"), ("old_weighted", "weighted")):
        records = {}
        for cid, gold in labels.items():
            path = influence_dir / "rankings" / method / f"{cid}.json"
            historical_sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = read(path)
            require(saved["trajectory_id"] == cases[cid]["trajectory_id"], "Historical trajectory mismatch")
            ids = [int(r["source_step_id"]) for r in saved["ranking"]]
            require(sorted(ids) == list(range(len(ids))), "Historical ranking ID mismatch")
            rank = min(ids.index(i)+1 for i in gold)
            records[cid] = {"rank": rank, "top1": int(rank == 1), "top5": int(rank <= 5), "mrr": 1/rank,
                            "fallback": saved.get("influence_fallback")}
        fixed[key] = {m: float(np.mean([r[m] for r in records.values()])) for m in ("top1", "top5", "mrr")}
        for metric, reference in (("top1", "step_level_top1"), ("top5", "step_level_top5"), ("mrr", "mrr")):
            require(abs(fixed[key][metric] - influence_metrics["metrics"][method][reference]) < 1e-12,
                    "Historical complete-method summary differs")
        historical_details[key] = records
    (OUT / "historical-ranking-check.json").write_text(json.dumps(historical_details, indent=2) + "\n")
    (OUT / "report-source-hashes.json").write_text(json.dumps(historical_sources, indent=2) + "\n")
    n, seeds = d["n"], d["seeds"]
    require(n == 102 and len(predictions) == n*len(seeds), "Unexpected sample count")
    for s in seeds:
        require(sum(d["rank_and_branch"][str(s)]["rank_bins"].values()) == n, "Rank totals differ")
    rank_bins = {k: float(np.mean([v["rank_bins"][k] for v in d["rank_and_branch"].values()]))
                 for k in ("1", "2", "3-5", "6-10", ">10")}
    dense = original["semantic/dense"]["top1"]
    near = rank_bins["2"] + rank_bins["3-5"]
    missed = n - rank_bins["1"]
    mean_branch = {k: float(np.mean([v[k] for v in d["rank_and_branch"].values()])) for k in
                   ("local_hits", "relation_gained", "relation_lost", "wrong_local_already_wrong", "wrong_earlier", "wrong_later")}
    short = [r for r in predictions if r["n_steps"] <= 5]
    longer = [r for r in predictions if r["n_steps"] > 5]
    derived = {"rank_bins_mean_counts": rank_bins, "mean_branch_counts": mean_branch,
               "near_miss_share": near/missed, "n_short_cases": len(short)//3,
               "top5_excluding_short": float(np.mean([r["rank"] <= 5 for r in longer])),
               "fixed_baselines": fixed, "score_only_baselines_without_historical_fallback": score_only}
    (OUT / "derived_summary.json").write_text(json.dumps(derived, indent=2) + "\n")

    external_rows, exact_rows = [], []
    for method, title in METHODS.items():
        row = [title]
        for suite in ("qwen38_thinking", "deepseek_official_thinking", "moma_deepseek41"):
            b = baseline[f"{suite}/{method}"]
            row.append(f"{pct(b['top1'])}（{b['hits']}/102；有效 {b['valid']}）")
            exact_rows.append({"protocol": suite, "method": method, "n": n, "hits": b["hits"],
                               "valid": b["valid"], "top1": b["top1"], "top1_seed_sd": "", "top5": "", "mrr": ""})
        external_rows.append(row)
    external_table = table(["方法", "Qwen3.8-Flash thinking", "官方 DeepSeek Flash thinking", "MoMA DeepSeek-V4.1（历史）"], external_rows)
    internal_rows = []
    ordered_keys = ["fixed/q_only", "fixed/old_weighted", "fixed/rrf", "lexical/dense"] + ["semantic/"+v for v in VARIANTS]
    for key in ordered_keys:
        protocol, name = key.split("/")
        if protocol == "fixed":
            r = fixed[name]
            title = {"q_only": "已有节点错误评分 q-only", "old_weighted": "已有影响度加权传播", "rrf": "固定 RRF"}[name]
            sd = 0.0
        else:
            r = original[key]
            title = "原词法 attention" if protocol == "lexical" else VARIANTS[name]
            sd = float(np.std([x["top1"] for x in r["per_seed"]], ddof=1))
        internal_rows.append([title, pct(r["top1"]), pct(r["top5"]), f"{r['mrr']:.4f}"])
        exact_rows.append({"protocol": protocol, "method": name, "n": n, "hits": r["top1"]*n,
                           "valid": n, "top1": r["top1"], "top1_seed_sd": sd,
                           "top5": r["top5"], "mrr": r["mrr"]})
    internal_table = table(["方法", "Top-1", "Top-5", "MRR"], internal_rows)
    with (OUT / "exact-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exact_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(exact_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    y = np.arange(6)
    for suite, offset, color, title in (("qwen38_thinking", -.16, "#2878A8", "Qwen3.8-Flash / thinking"),
                                       ("deepseek_official_thinking", .16, "#D47936", "Official DeepSeek Flash / thinking")):
        values = [100*baseline[f"{suite}/{m}"]["top1"] for m in METHODS]
        ax.scatter(values, y+offset, color=color, label=title, s=60, zorder=3)
        for value, pos in zip(values, y+offset):
            ax.text(value+.8, pos, f"{value:.2f}", va="center", fontsize=9, color=color)
    ax.axvline(100*dense, color="#252C34", linestyle="--", label=f"Semantic attention: {100*dense:.2f}% (3-seed mean)")
    ax.set_yticks(y, ["All-at-Once", "Step-by-Step", "Binary Search", "TRAIL", "FAMAS (adapted)", "TrajAudit (generic)"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 65), xlabel="Exact Top-1 (%) / all 102 cases", title="Historical full-diagnosis baselines and attention ranking")
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.36), frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"baseline-comparison.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
    names = list(VARIANTS)
    for key, offset, color, title in (("train_top1", -.15, "#97A7B3", "Training (checkpoint replay)"),
                                     ("test_top1", .15, "#2878A8", "Held-out repository/task groups")):
        matrix = np.array([[r[key]*100 for r in d["training"][v]] for v in names])
        axes[0].barh(np.arange(6)+offset, matrix.mean(1), height=.28, color=color, label=title)
        for i, values in enumerate(matrix):
            axes[0].scatter(values, np.full(3, i+offset), s=15, color="#333333", zorder=3)
    axes[0].set_yticks(range(6), ["Linear", "MLP", "Fixed graph", "Unconditioned A", "Attention", "No graph prior"])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 105), xlabel="Top-1 (%)", title="A. Training-to-test gap")
    axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.27), frameon=False)
    axes[0].grid(axis="x", alpha=.15)
    colors = ["#2878A8", "#D47936", "#E6AD70", "#A6B1BA", "#D1D7DC"]
    bars = axes[1].bar(range(5), list(rank_bins.values()), color=colors)
    for bar, value in zip(bars, rank_bins.values()):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.7, f"{value:.2f}", ha="center")
    axes[1].set_xticks(range(5), list(rank_bins))
    axes[1].set(ylim=(0, 38), xlabel="Rank of the true root", ylabel="Mean cases per seed (N = 102)", title="B. Where Top-1 misses go")
    axes[1].text(.5, -.25, f"Ranks 2-5: {near:.2f} cases\n{100*near/missed:.2f}% of Top-1 misses", transform=axes[1].transAxes, ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ranking-diagnosis.{ext}", dpi=180)
    plt.close(fig)

    case_notes = {
        "5fcfbc3071430840": ("正确分析与错误实现被混在一起", "第 1 步分析提到了用 `get_email_field_name()` 取得自定义邮箱字段；第 2 步补丁却写成 `getattr(user, 'email', '')`。失败测试检查 CustomEmailField，改变真实邮箱后 token 仍有效。模型一直选择分析方向正确的第 1 步，Qwen All-at-Once 选择真正写错补丁的第 2 步。此例只有 3 步，Top-5 自动覆盖全轨迹；它只用于解释首位判断。"),
        "2488543283c20334": ("关系分数覆盖了正确的单步判断", "Ansible 的失败测试断言锁的 `__enter__` 被调用。第 5 步使用 `acquire()/release()`，没有 `with self._lock:`，正是契约不匹配；第 2 步是更早的编辑工具参数错误，未完成那次插入。单步分支三个种子都选对第 5 步，关系分支却将第 2 步推到第一。Qwen All-at-Once 选择第 5 步。"),
        "28c72892a2950b21": ("错误承诺和后续实施没有分清", "OpenLibrary 只要求 Booknotes 在冲突时保留记录。第 3 步就计划取消共享函数的删除逻辑；第 9 步实施了这个计划，使 Bookshelves 的冲突处理也改变。标注根因是最早形成错误范围判断的第 3 步。单步分支三个种子都选对，关系分支把第 9 步推到第一；Qwen All-at-Once 选择第 3 步。"),
        "db0d9102f516fa52": ("精确契约差异没有转化为排序证据", "Qutebrowser 要求提示以 `executable)` 结尾，第 13 步写成 `executable.)`，多一个句点；第 11 步只是建立复现脚本。模型三个种子都选第 11 步，真根因均排第 10；Qwen All-at-Once 选第 13 步。第 13 步编辑操作本身失败，核对原始轨迹可见错误字符串随后在第 20/21 步沿用，所以需要追踪错误承诺是否持续。"),
    }
    case_sections = []
    for cid, (title, explanation) in case_notes.items():
        c = packets[cid]
        rows = c["rows"]
        data_rows = [[r["seed"], r["root"], r["pred"], r["rank"], r["local_rank"],
                      f"{r['local_margin']:+.3f}", f"{r['relation_margin']:+.3f}"] for r in rows]
        case_sections.append(f"## {title}\n\n轨迹 `{c['trajectory_id']}`，共 {c['case']['n_steps']} 步。\n\n{explanation}\n\n" +
                             table(["种子", "根因 step ID", "预测 step ID", "最终根因排名", "单步根因排名", "单步分差", "关系分差"], data_rows) +
                             f"\n\n[原始轨迹]({c['case']['sources']['raw']['path']}) · [Qwen 原始判断]({c['baseline']['qwen38_thinking/all_at_once']['state_path']})\n")
    (OUT / "case-review.md").write_text("# 代表性错误案例核对\n\n所有 step ID 沿用数据的零起始编号。分差 = 错误预测步骤得分 − 真根因得分，正值支持错误预测。案例从预定义错误组按较短轨迹、ID 选择；除关系覆盖组外，要求 Qwen All-at-Once 正确。此样本用于核对机制，不用于估计机制在全部 102 条中的发生率。标注原因仅用于评估后的核对。\n\n" + "\n".join(case_sections))

    stats_rows = []
    for name, c in comparisons.items():
        lo, hi = c["group_bootstrap_95ci"]
        stats_rows.append([name, f"{100*c['delta']:+.2f}", f"[{100*lo:+.2f}, {100*hi:+.2f}]", f"{c['group_label_swap_p']:.5f}", f"{c['holm_adjusted_p']:.5f}"])
    training_rows = []
    for v, title in VARIANTS.items():
        tr = np.array([r["train_top1"] for r in d["training"][v]])*100
        te = np.array([r["test_top1"] for r in d["training"][v]])*100
        m = original["semantic/"+v]
        training_rows.append([title, f"{tr.mean():.2f} ± {tr.std(ddof=1):.2f}", f"{te.mean():.2f} ± {te.std(ddof=1):.2f}", "/".join(str(r["top1_count"]) for r in m["per_seed"])])
    stats = f"""# 统计附录

比较单位为同一批 102 条轨迹。按仓库或任务连接成 35 组；5 折测试组不进入相应训练。三个训练种子为 17、29、43。每个案例先对三个种子取命中均值，再与 baseline 配对，未将 306 条案例×种子记录当作独立样本。

下表为训练集和折外测试集 Top-1 的三种子均值 ± 样本标准差，单位为百分点。训练指标来自保存 checkpoint 的训练样本回放，每个案例会出现在多个训练折中，仅描述拟合程度，不作为独立检验样本。每折训练 81 或 82 条，dense 有 43,653 个参与反向传播的参数；其中 failure_key 数学上冗余。

{table(["方法", "训练 Top-1", "测试 Top-1", "测试命中数（17/29/43）"], training_rows)}

效应为“语义 dense − 对照”的 Top-1 差，单位百分点。95% 区间按 35 组有放回抽样 5,000 次，组内保留所有案例，统计量为案例加权平均；每次抽样因组大小不同会改变总案例数。双侧检验在组内整体交换两方法标签 20,000 次，采用加一校正。全部 16 项探索性比较作为同一族做 Holm 校正，随机种子 20261027。区间是逐项区间，不是同时覆盖区间；组间可交换假设和同一数据反复开发的背景保留在解释中。

{table(["对照", "差值 pp", "95% 组 bootstrap 区间", "组交换 p", "Holm p"], stats_rows)}

这些历史 LLM baseline 各有一次完整协议执行，无法估计各自解码运行间方差。Qwen/官方 DeepSeek 的 thinking、原文读取和方法调用次数与小模型排序器不同，配对检验描述这两套现有系统在本批数据上的差异，不能单独归因于 attention 结构。MoMA 旧套件只作历史描述，未加入上述检验族。

rank 2–5 占错误比例按 142/219 = {100*near/missed:.2f}% 计算，这是三种子描述性合计，不把 219 当作独立错误样本。{len(short)//3} 条轨迹长度不超过 5，Top-5 必然命中；排除后其余 93 条的 Top-5 为 {pct(derived['top5_excluding_short'])}。训练拟合、局部/关系分数拆分及置零干预均使用冻结模型，不构成重新训练后的消融结果。

全部数值见 exact-results.csv、comparisons.json、case_seed_diagnostics.json。baseline 无合法预测、截断和 API 失败均保留在 102 分母里，按错误计；不使用仅有效样本的准确率。baseline 的 tolerance@k 是时间步容差，未当作候选 Top-k。

历史 q-only 与影响度传播必须读取完整保存排名：原方法在信号全零等条件下保留基础排序，因此分别命中 21、10 条。attention 输入缓存仅保存对应原始分数；若直接按分数和早先步骤排序，会得到 20、7 条，省略了原方法回退。本报告主表采用历史完整方法的 21、10，原分数复算另存 derived_summary.json，逐例原排名核验见 historical-ranking-check.json。
"""
    (OUT / "stats-appendix.md").write_text(stats)

    code = ROOT / "src/loghound_agent_debug/rankers/attention_map.py"
    report = f"""# Attention map 为什么排不到 Top-1，以及与 baseline 的差距

当前主要问题是：候选里经常已经有真根因，但模型还不能稳定判断“哪一步真正违反任务或测试要求”，并存在明显的训练到跨仓库测试的泛化落差。语义 attention 的 Top-1 为 **28.43%（平均 29/102）**，Top-5 为 **74.84%（平均 76.33/102）**。

## 首位排序卡在哪里

每个种子平均有 **47.33 条**真根因排在第 2–5 位，占全部 Top-1 错误的 **64.84%**。三个种子都错的轨迹有 68 条，其中 38 条始终排第 2–5，21 条始终在 Top-5 外。这个分布说明需要同时改善候选内判别和候选外召回。9 条短轨迹的 Top-5 自动覆盖全部步骤，排除后其余 93 条仍为 72.40%。

**训练拟合远好于跨仓库测试。** dense 训练 Top-1 为 99.92%，测试只有 28.43%；MLP 为 95.92% 对 24.51%。每折约 81–82 条训练轨迹，dense 学习约 4.4 万个参数。已有记录支持明显过拟合/泛化不足的判断；区分模型容量、表示和仓库分布各自贡献，需要在训练折内做下一轮验证。

**大部分错误在单步判断时已经出现。** 在完整模型每种子平均 73 个错误中，68 个在单步分支就没排对，占 93.15%。关系分支平均纠正 7.67 个、改错 5 个，净增加 2.67 个命中。这里的“单步分支”是联合训练模型中的 local 分数，独立训练 MLP 的对照见下表。

回读原始轨迹发现两种不同的判断要求：Django 案例的早期分析是对的，之后补丁硬编码邮箱字段才错；OpenLibrary 案例则是早期计划已经扩大了需求作用范围，后来代码只是执行这个错误计划。模型需要识别错误首次引入的位置及其后续是否被修复，统一偏爱早或晚都不够。全量错误中平均 41 个选早了、32 个选晚了，并不支持把本轮共因归为“总选后续症状”。

Ansible 案例进一步说明关系可能放大无关错误：真正问题是用 `acquire()/release()` 代替测试要求的锁上下文管理器。单步评分三个种子都选对第 5 步，关系分数却把更早一次编辑工具参数错误的第 2 步推到第一。全量有 7 条至少一个种子出现这种覆盖，3 条在三个种子中都出现。[逐例证据与分差](case-review.md)

![训练测试差距与根因排名分布]({OUT}/ranking-diagnosis.png)

## 失败信息注入的一处实现问题

[attention_map.py 第 63–66 行]({code}:63) 把同一个失败向量分别加到每个 query 和 key。对第 j 行、第 i 列：

`logit[j,i] = (q[j] + a)ᵀ(k[i] + b) / √d`

其中 `(q[j] + a)ᵀb` 对一行的所有候选 i 都相同，会被逐行 softmax 完全抵消。因此 **failure_key 这条分支没有实际区分作用**。全部 306 份折外预测中，把它置零后 Top-1 改变 **0 次**；attention 最大差仅 1.34e−6，符合浮点误差。剩余 query 注入产生的是所有行共享的失败相关 key 偏置，表达形式也较有限。单步分数和 seed 仍使用失败信息。

这个问题解释了为何当前设计没有实现预期的双侧失败条件化；因为删除无效分支不改变预测，单删它不会直接修复 Top-1。另一个独立测试是置零 failure_query，合计命中从 87/306 到 89/306；重新训练的无失败关系版本则为 29.74%，两者是不同实验。

## 与已有 baseline 的对照

同一批 102 条轨迹与逐条真值已经核对。下表先列现有信号和相同输入的学习方法，所有学习指标为三种子折外均值。q-only 和影响度传播读取历史完整排名（保留全零信号回退），RRF 由冻结分数复算；前三项使用既有缓存，未重新调用模型。

{internal_table}

语义 dense 比独立 MLP 高 3.92 个百分点，比固定图高 0.98，比关系不注入失败的版本低 1.31。按仓库/任务组的配对区间均包含零，当前结果支持“语义表示改善候选质量”，尚未确定复杂关系模块的首位收益。

下面是此前已跑完的六种完整诊断方法。每格给出 Top-1、精确命中数、有效输出数；所有比例统一除以 102，失败输出计错。

{external_table}

attention 比 Qwen All-at-Once 低 **18.63 个百分点**，比 Qwen TrajAudit 低 **22.55 个百分点**，比官方 DeepSeek All-at-Once 低 **27.45 个百分点**。三项差距的组 bootstrap 区间均在零以下，Holm 校正后 p 均小于 0.05。[完整配对统计](stats-appendix.md)

![完整诊断 baseline 对照]({OUT}/baseline-comparison.png)

一个直接指向改进方向的结果是：**Qwen All-at-Once 正确的 48 条里，每个种子都有 40 条的真根因已在 attention Top-5 中**。其中 attention 尚未排第一、但该 baseline 能选对的有 23–27 条。三个种子都在第 2–5 位的 38 条里，Qwen All-at-Once 能选对 20 条。这说明已有大模型判断能解决相当一部分候选内歧义；候选重排的实际收益需用新协议测量，不能把该交集直接当作重排准确率。

比较协议：LLM baseline 读取任务、原始轨迹和首个失败测试，隐藏根因与参考补丁；Qwen 和官方 DeepSeek 开启 thinking，每次输出上限 32,768。attention 使用冻结 MiniLM 表示与既有 Qwen 非思考图/q 缓存，进行监督分组训练。历史六方法来自同一 TrajAudit 适配实现，FAMAS 为本地适配、TrajAudit 为 generic 配置。因模型、输入呈现和调用预算不同，上表用于比较现有系统的定位结果；结构贡献由前一张同输入表判断。

## 下一轮改动的优先级

1. **把候选判别做实。** 固定当前候选器，对候选步骤联合读取需求、失败断言、原始 action/patch、相关前后文，判断违反哪条约束、错误何时首次出现、之后是否修复。先用同一 Qwen 配置比较 All-at-Once、候选重排、候选加关系证据，记录实际 token 与调用量。保留覆盖完整轨迹的对照：Qwen 命中的 48 条还有 8 条真根因不在当前 Top-5 内。
2. **控制训练泛化。** 用训练折内的分组验证选择早停与正则，比较更小评分头；测试折只作最终评估。现有 102 条已经反复开发，新增任务保留为外部验证。
3. **重写失败条件化并设置关系门控。** 可把失败与步骤做非线性交互或逐维门控，再生成 Q/K；关系分数经可学习且受约束的门控加到单步分数。先检验改变失败输入确实改变有效 attention，再比较重训练的 Top-1。仅删除抵消项不会带来现有预测变化。

周三汇报可以用这一句话概括：**“当前已经能把根因放进前五，但还不能稳定区分相关步骤与真正错误；接下来用需求和测试约束做候选裁决，并验证关系证据能否进一步提高首位命中。”**

本轮新增训练 0、API 调用 0；重放 1,836 份 attention 预测，逐例核对 1,836 份历史 LLM baseline 预测及 204 份 q/影响度排名；诊断的 2,328 份来源文件及新增核验的 205 份历史排名/指标文件哈希未变。[统计附录](stats-appendix.md) · [图表说明](figure-catalog.md) · [精确数据](exact-results.csv)

复现：在仓库使用 `.venv-attention-semantic/bin/python scripts/diagnose_attentionmap_top1.py`，然后运行 `.venv-attention-semantic/bin/python scripts/report_attentionmap_top1.py`。报告和诊断独立保存，现有模型参数与预测保持冻结。
"""
    (OUT / "analysis-report.md").write_text(report)
    (OUT / "figure-catalog.md").write_text("""# 图表说明

| 文件 | 用途 | 图注与解释 |
| --- | --- | --- |
| baseline-comparison.png / .pdf | 主对照 | 同批 102 轨迹，点为一次完整 LLM 协议的精确 Top-1，虚线为 attention 三种子均值。全部无效输出计错。不同协议未对齐总预算；配对区间见统计附录。 |
| ranking-diagnosis.png / .pdf | 原因分析 | 左图为保存模型的训练/折外 Top-1，条为三种子均值，黑点是各个种子；右图为各根因排名区间的三种子平均案例数，总和 102。训练拟合差距和 2–5 名的集中支持把泛化与候选判别列为下一步重点。 |
| exact-results.csv | 精确数表 | 记录分母、命中数或种子平均命中数、有效输出、Top-1、种子标准差、Top-5、MRR。LLM 只返回一个 ID，因此 Top-5/MRR 留空。 |

可支撑的主张：① 当前 dense 训练近乎全对而分组测试为 28.43%，有明显泛化落差；② 64.84% 的 Top-1 错误真根因在 2–5 名，适合检验候选重排；③ failure_key 的行常数被 softmax 抵消，离线干预未改变 Top-1。候选重排是否提升、门控能否改善泛化，应由冻结新协议和新任务验证，不从示例直接外推。
""")
    sources = read(OUT / "source_hashes.json")
    for name, expected in sources.items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, "Source changed during reporting")
    for name, expected in historical_sources.items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, "Historical ranking changed during reporting")
    (OUT / "report-verification.json").write_text(json.dumps({"source_files_unchanged": len(sources),
        "additional_historical_files_unchanged": len(historical_sources),
        "exact_result_rows": len(exact_rows), "case_reviews": len(case_notes),
        "rank_total": sum(rank_bins.values()), "new_training_fits": 0, "new_api_calls": 0,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(derived, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
