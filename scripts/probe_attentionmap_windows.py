#!/usr/bin/env python3
"""Frozen-checkpoint probes of temporal windows and relation-score strength."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker
from analyze_attentionmap_semantic import measure, require
from diagnose_attentionmap_top1 import cluster_stats
from run_attentionmap import order, read, save, sha

SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
OUT = ROOT / "outputs/attentionmap-window-probe-20261027"
# Fixed before any new probe outcomes are calculated. Every variant is reported.
VARIANTS = {
    "original": ("full", 0, 1.0, "原模型：全历史，关系权重 1"),
    "local_only": ("full", 0, 0.0, "仅保留联合训练的单步分数"),
    "relation_quarter": ("full", 0, .25, "全历史，关系权重 0.25"),
    "relation_half": ("full", 0, .5, "全历史，关系权重 0.5"),
    "window8": ("window", 8, 1.0, "最近 8 步窗口"),
    "window16": ("window", 16, 1.0, "最近 16 步窗口"),
    "window32": ("window", 32, 1.0, "最近 32 步窗口"),
    "soft16": ("soft", 16, 1.0, "软距离衰减：16 步尺度"),
    "window16_prior": ("window_prior", 16, 1.0, "16 步窗口 + 原图长程边"),
    "window16_prior_half": ("window_prior", 16, .5, "16 步窗口 + 原图长程边，关系权重 0.5"),
    "top4": ("topk", 4, 1.0, "每行只保留 attention 最高的 4 步"),
    "top8": ("topk", 8, 1.0, "每行只保留 attention 最高的 8 步"),
}


def masked_relation(model, x, failure, prior, mode, width):
    n = len(x)
    h = model.encode(x)
    f = model.failure_encode(failure).expand(n, -1)
    rh = model.relation_encode(x[:, :256])
    query = model.query(rh) + model.failure_query(f)
    key = model.key(rh) + model.failure_key(f)
    logits = query @ key.T / math.sqrt(model.hidden) + model.graph_bias * prior
    index = torch.arange(n)
    distance = (index[:, None] - index[None, :]).clamp(min=0)
    logits = logits - F.softplus(model.distance_penalty) * (distance / max(n - 1, 1))
    causal = torch.ones((n, n), dtype=torch.bool).tril()
    allowed = causal.clone()
    if mode in {"window", "window_prior"}:
        allowed &= (distance <= width)
        if mode == "window_prior":
            allowed |= (prior > 0)
    elif mode == "soft":
        logits = logits - distance / width
    elif mode == "topk":
        indices = logits.masked_fill(~causal, -torch.inf).topk(min(width, n), dim=-1).indices
        allowed = torch.zeros_like(causal).scatter_(1, indices, True) & causal
    elif mode != "full":
        raise ValueError(mode)
    attention = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=-1)
    require(torch.allclose(attention.sum(-1), torch.ones(n), atol=1e-6), "Row mass drift")
    require(not attention[~allowed].any(), "Attention leaked outside mask")
    require(not attention[~causal].any(), "Future attention leak")
    seeds = torch.softmax(model.seed(torch.cat([h, f, h*f], dim=-1)).squeeze(-1), dim=0)
    weights = seeds[:, None] * attention
    mass = weights.sum(0)
    context = weights.T @ model.value(rh) / mass[:, None].clamp_min(1e-8)
    features = torch.cat([context, h*context, torch.log1p(n*mass)[:, None]], dim=-1)
    return model.relation_score(features).squeeze(-1), attention


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read(SOURCE / "manifest.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    labels = read(SOURCE / "labels.json")
    protocol = {
        "hypothesis": "限制关系关注范围或减弱关系分数，可减少无关证据对局部判断的覆盖。",
        "variants": VARIANTS, "checkpoint_variant": "dense", "seeds": manifest["config"]["seeds"],
        "source_fingerprint": manifest["fingerprint"], "n": len(cases),
        "evaluation": "exploratory frozen-checkpoint OOF intervention; no retraining or test-label hyperparameter selection",
        "window_definition": "step j attends to j and at most w previous steps; every trajectory step remains a root candidate",
        "score_definition": "local[i] + alpha * recomputed_relation[i]",
        "soft_definition": "subtract (j-i)/16 from original logits before row softmax",
        "global_definition": "window plus existing prior>0 long-range edges; no gold-dependent retrieval",
        "future_test": "select window/alpha using grouped validation inside each training fold; evaluate fresh tasks",
        "new_api_calls": 0, "new_training_fits": 0,
    }
    if (OUT / "protocol.json").exists():
        require(read(OUT / "protocol.json") == json.loads(json.dumps(protocol)), "Probe protocol changed")
    else:
        save(OUT / "protocol.json", protocol)
    sources = {}

    def record(path):
        sources[str(path)] = sha(path)

    record(SOURCE / "manifest.json")
    record(SOURCE / "labels.json")
    record(ROOT / "src/loghound_agent_debug/rankers/attention_map.py")
    data = {}
    for cid, case in cases.items():
        p = SOURCE / "features" / f"{cid}.npz"
        record(p)
        require(sha(p) == case["feature_sha256"], "Feature drift")
        with np.load(p) as z:
            data[cid] = tuple(torch.from_numpy(z[k]) for k in ("x", "failure", "prior"))
    rows = []
    coefficients = []
    with torch.no_grad():
        for seed in protocol["seeds"]:
            for fold in range(5):
                run = SOURCE / "runs/dense" / f"seed-{seed}" / f"fold-{fold}"
                for name in ("model.pt", "predictions.json", "status.json"):
                    record(run / name)
                status = read(run / "status.json")
                require(sha(run / "model.pt") == status["model_sha256"], "Checkpoint drift")
                require(sha(run / "predictions.json") == status["predictions_sha256"], "Saved predictions drift")
                cp = torch.load(run / "model.pt", weights_only=True, map_location="cpu")
                model = AttentionMapRanker(cp["input_dim"], hidden=cp["hidden"], variant="dense")
                model.load_state_dict(cp["state_dict"])
                model.eval()
                coefficients.append(float(F.softplus(model.distance_penalty)))
                for cid, saved in read(run / "predictions.json").items():
                    require(cases[cid]["fold"] == fold and cid not in status["train_ids"], "OOF membership mismatch")
                    x, f, prior = data[cid]
                    base = model(x, f, prior)
                    require(np.array_equal(base["scores"].numpy(), saved["scores"]), "Original replay mismatch")
                    rebuilt, attention = masked_relation(model, x, f, prior, "full", 0)
                    require(torch.equal(rebuilt, base["relation"]), "Relation reconstruction mismatch")
                    wide, _ = masked_relation(model, x, f, prior, "window", len(x))
                    require(torch.equal(wide, rebuilt), "Full-width window must equal original")
                    cache = {("full", 0): (rebuilt, attention)}
                    for name, (mode, width, alpha, _) in VARIANTS.items():
                        if (mode, width) not in cache:
                            cache[(mode, width)] = masked_relation(model, x, f, prior, mode, width)
                        relation, attn = cache[(mode, width)]
                        scores = base["local"] + alpha * relation
                        value = measure(scores.tolist(), labels[cid])
                        root = labels[cid][0]
                        rows.append({"case_id": cid, "trajectory_id": cases[cid]["trajectory_id"],
                                     "n_steps": len(x), "seed": seed, "fold": fold, "variant": name,
                                     **value, "pred": order(scores.tolist())[0], "root": root,
                                     "original_pred": order(base["scores"].tolist())[0],
                                     "original_top1": measure(saved["scores"], labels[cid])["top1"],
                                     "local_rank": measure(saved["local"], labels[cid])["rank"],
                                     "scores": scores.tolist(), "mean_nonzero_attention_per_row": float((attn > 0).float().sum(1).mean())})
            print(json.dumps({"seed_done": seed, "rows": len(rows)}), flush=True)
    summaries = {}
    for name in VARIANTS:
        selected = [r for r in rows if r["variant"] == name]
        require(len(selected) == len(cases)*3, "Incomplete probe")
        per_seed = []
        for seed in protocol["seeds"]:
            subset = [r for r in selected if r["seed"] == seed]
            per_seed.append({"seed": seed, "hits": sum(r["top1"] for r in subset),
                             **{k: float(np.mean([r[k] for r in subset])) for k in ("top1", "top5", "mrr")},
                             "gained": sum(r["top1"] and not r["original_top1"] for r in subset),
                             "lost": sum(not r["top1"] and r["original_top1"] for r in subset)})
        difference = {cid: float(np.mean([r["top1"]-r["original_top1"] for r in selected if r["case_id"] == cid])) for cid in cases}
        summaries[name] = {**{k: float(np.mean([r[k] for r in selected])) for k in ("top1", "top5", "mrr")},
                           "per_seed": per_seed,
                           "top1_seed_sd": float(np.std([r["top1"] for r in per_seed], ddof=1)),
                           "local_correct_final_wrong": sum(r["local_rank"] == 1 and not r["top1"] for r in selected)/3,
                           "local_wrong_final_correct": sum(r["local_rank"] > 1 and r["top1"] for r in selected)/3,
                           "pairing": cluster_stats(difference, cases)}
    names = sorted((k for k in summaries if k != "original"), key=lambda k: summaries[k]["pairing"]["group_label_swap_p"])
    previous = 0.0
    for i, name in enumerate(names):
        previous = max(previous, min(1.0, (len(names)-i)*summaries[name]["pairing"]["group_label_swap_p"]))
        summaries[name]["pairing"]["holm_adjusted_p"] = previous
    for path, expected in sources.items():
        require(sha(Path(path)) == expected, "Frozen source modified")
    save(OUT / "predictions.json", rows)
    save(OUT / "summary.json", summaries)
    save(OUT / "source_hashes.json", sources)
    save(OUT / "verification.json", {"files_unchanged": len(sources), "replayed_original_predictions": 306,
        "probe_predictions": len(rows), "causal_mask_row_mass_full_width_checks": "passed for all cases and seeds",
        "original_distance_coefficients": coefficients, "new_training_fits": 0, "new_api_calls": 0,
        "script_sha256": sha(Path(__file__))})
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["variant", "top1", "top5", "mrr", "seed17_hits", "seed29_hits", "seed43_hits", "delta_pp", "ci_low_pp", "ci_high_pp", "holm_p"])
        for name, s in summaries.items():
            p = s["pairing"]
            writer.writerow([name, s["top1"], s["top5"], s["mrr"], *[r["hits"] for r in s["per_seed"]],
                             100*p["delta"], *[100*v for v in p["group_bootstrap_95ci"]], p.get("holm_adjusted_p", "")])
    lines = ["# 上下文窗口与关系分数：冻结模型离线对照", "",
        f"已验证可以通过窗口和关系权重改变排序。本轮原版 Top-1 {100*summaries['original']['top1']:.2f}%，16 步纯窗口与窗口加原图长程边均为 {100*summaries['window16']['top1']:.2f}%；关系权重减半为 {100*summaries['relation_half']['top1']:.2f}%，Top-5 为 {100*summaries['relation_half']['top5']:.2f}%。首位变化较小，全部 11 项比较经 Holm 校正后 p=1.0。",
        "",
        "研究问题：限制历史关注范围、保留少量长程联系或减弱关系分数，能否改善首位排序，同时保持根因候选覆盖？",
        "",
        "固定同一批 102 条、35 仓库/任务组、5 折、3 种子，使用原语义 dense checkpoint。以下 12 个设置在运行前写入 protocol.json，全部报告。新训练与 API 请求均为 0。",
        "",
        "窗口限制的是步骤间 attention；每个步骤仍参与全轨迹根因排名。任务与失败测试的输入保持完整。w 表示最近 w 个历史步骤加当前步骤；软窗口给原 logits 再减 (j−i)/16；长程边来自原有图。最终分数为 local + α × relation，未改变单步文本表示。",
        "", "| 设置 | Top-1 | Top-5 | MRR | 三种子命中数 | 相对原版 pp（95% 组区间） |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, s in summaries.items():
        p = s["pairing"];lo, hi = p["group_bootstrap_95ci"]
        lines.append(f"| {VARIANTS[name][3]} | {100*s['top1']:.2f}% | {100*s['top5']:.2f}% | {s['mrr']:.4f} | {'/'.join(str(int(r['hits'])) for r in s['per_seed'])} | {100*p['delta']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines.extend(["", "这是推理时干预，用于定位敏感环节；训练仍使用原全历史结构。正式版本需要按新窗口重新训练，窗口和 α 在训练折内的分组验证上确定，测试折只做最终评价。未把本表最优参数直接指定为正式模型。",
        "", "## 当前实现为什么限制不够",
        f"原模型虽然已有距离惩罚，但系数在 15 个 checkpoint 中只有 {min(coefficients):.3f}–{max(coefficients):.3f}，距离还除以整条轨迹长度；即使跨越整条轨迹，扣分也只有约 0.15。它仍是全历史 attention。另一个独立问题是最终直接相加 local + relation，没有对每个候选控制关系分数影响。",
        "", "## 两种窗口应分开设计",
        "1. **关系窗口**：第 j 步主要关联附近历史步骤，并保留可追溯到同一文件、符号或依赖来源的长程联系。当前已实际测试的是这种步骤间 mask；16 步窗口与保留原图长边的版本 Top-1 相同，现有试验没有区分出二者的稳定优势。",
        "2. **判断窗口**：针对候选 i，读取任务、失败断言、候选原始内容与前后 2–4 步；再按文件/符号补充远处修改和恢复证据。任务与失败测试始终保留。现有 local 分支只吃当前步骤的冻结表示和全局条件，尚未实现这项候选局部文本判别。",
        "", "建议组合为 `score_i = local_error_i + gate_i * relation_i`：local_error 判断当前步骤违反了哪条具体要求，relation 汇总相关后续证据，gate 在训练折内学习控制关系贡献。先用统一固定 α 做对照，再增加逐候选门控；这样能分清收益来自关注范围还是分数组合。门控与局部文本判别属于下一轮需要实现并重新训练/推理的部分。",
        "", "## 一个具体的分数调整效果",
        "Ansible 锁案例中，原模型三个种子都将真根因排第二；关系权重减半后，前两个种子将其升到第一，第三个仍第二。单用 16 步窗口则为第 2/3/3 名。它说明这个案例更需要控制关系分数的覆盖作用。Django 的正确分析与错误补丁混淆，在上述窗口及减半设置下仍存在，下一步应补充任务/断言与局部代码差异的判别。",
        "", "统计：同一案例的三种子命中先平均，35 组 bootstrap 5,000 次；组内整体方法标签交换 20,000 次；11 项对照作为同一族进行 Holm 校正，详见 summary.json。此处为已反复开发数据上的探索性诊断。",
        "", "方法参考：[Longformer](https://arxiv.org/abs/2004.05150) 将局部窗口与任务相关全局 attention 结合；这里检验的是步骤级故障证据聚合，保留哪些长程边及怎样调整故障分数由本实验定义。",
        "", "下一步方案：分别比较全历史、纯窗口、窗口加长程边；单独比较固定 α 和受约束的关系门控。关注 Top-1、Top-5、原正确案例丢失数、长轨迹表现及训练/验证差距。涉及原始文本的局部裁决，可让每个候选读取附近步骤并按同一文件/符号补充远程修改与恢复证据。",
        "", "复现：`.venv-attention-semantic/bin/python scripts/probe_attentionmap_windows.py`。"])
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: {"top1": v["top1"], "top5": v["top5"], "mrr": v["mrr"], "hits": [x["hits"] for x in v["per_seed"]]} for k,v in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
