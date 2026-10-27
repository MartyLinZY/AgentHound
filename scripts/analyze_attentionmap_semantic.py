#!/usr/bin/env python3
"""Recompute paired OOF metrics and replay every checkpoint for both encodings."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker
from run_attentionmap import read, save, sha

METRICS = ("top1", "top3", "top5", "mrr")
MODES = {"scores": "", "shuffled_scores": "_shuffled",
         "drop50_scores": "_drop50", "drop100_scores": "_drop100"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def measure(scores: list[float], gold: list[int]) -> dict:
    values = np.asarray(scores)
    require(np.isfinite(values).all(), "Nonfinite scores")
    order = np.lexsort((np.arange(len(values)), -values))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    rank = min(int(ranks[g]) for g in gold)
    return {"rank": rank, "top1": int(rank == 1), "top3": int(rank <= 3),
            "top5": int(rank <= 5), "mrr": 1 / rank}


def paired(left: dict, right: dict, cases: dict) -> dict:
    seeds = list(left)
    values = {cid: float(np.mean([left[s][cid]["top1"] - right[s][cid]["top1"]
                                 for s in seeds])) for cid in cases}
    groups = sorted({c["group"] for c in cases.values()})
    group_deltas = [[values[cid] for cid, c in cases.items() if c["group"] == g] for g in groups]
    rng = np.random.default_rng(20261027)
    samples = [np.mean([v for index in rng.integers(len(groups), size=len(groups))
                        for v in group_deltas[index]]) for _ in range(2000)]
    return {"top1_delta": float(np.mean(list(values.values()))),
            "group_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "per_seed": [{"seed": s,
                          "gained": sum(left[s][c]["top1"] > right[s][c]["top1"] for c in cases),
                          "lost": sum(left[s][c]["top1"] < right[s][c]["top1"] for c in cases)}
                         for s in seeds]}


def inspect_run(out: Path, original: Path, representation: str) -> tuple[dict, dict, dict]:
    manifest = read(out / "manifest.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    labels = read(out / "labels.json")
    summary = read(out / "summary.json")
    require(sha(out / "labels.json") == manifest["labels_sha256"], "Label drift")
    require(sha(out / "baselines.json") == manifest["baselines_sha256"], "Baseline drift")
    data = {}
    for cid, case in cases.items():
        feature = out / "features" / f"{cid}.npz"
        require(sha(feature) == case["feature_sha256"], "Feature drift")
        with np.load(feature, allow_pickle=False) as z, np.load(original / "features" / f"{cid}.npz") as old:
            for name in ("x", "failure", "prior"):
                require(z[name].shape == old[name].shape, "Feature shape changed")
                require(np.isfinite(z[name]).all(), "Nonfinite features")
            require(np.array_equal(z["prior"], old["prior"]), "Prior changed")
            require(np.array_equal(z["x"][:, 512:], old["x"][:, 512:]), "Numeric features changed")
            if representation == "lexical":
                require(all(np.array_equal(z[k], old[k]) for k in z.files), "Lexical feature changed")
            data[cid] = tuple(torch.from_numpy(z[k]) for k in ("x", "failure", "prior"))
    metrics = {}
    predictions = {}
    cost = 0.0
    fits = replayed = maps = original_equal = 0
    active_parameters = {}
    for variant in manifest["config"]["variants"]:
        predictions[variant] = {}
        for seed in manifest["config"]["seeds"]:
            combined = {}
            for fold in range(manifest["config"]["folds"]):
                relative = Path("runs") / variant / f"seed-{seed}" / f"fold-{fold}"
                run = out / relative
                status = read(run / "status.json")
                prediction = read(run / "predictions.json")
                require(status["fingerprint"] == manifest["fingerprint"], "Run fingerprint changed")
                require(sha(run / "model.pt") == status["model_sha256"], "Checkpoint drift")
                require(sha(run / "predictions.json") == status["predictions_sha256"], "Prediction drift")
                require(np.isfinite(status["losses"]).all(), "Nonfinite training loss")
                train, test = set(status["train_ids"]), set(status["test_ids"])
                require(not train & test and train | test == set(cases) and test == set(prediction), "OOF IDs mismatch")
                require(test == {cid for cid, c in cases.items() if c["fold"] == fold}, "Wrong test fold")
                for key in ("group", "repository", "task_id"):
                    require(not {cases[c][key] for c in train if cases[c][key]} &
                            {cases[c][key] for c in test if cases[c][key]}, f"{key} leakage")
                require(not set(combined) & test, "Duplicate OOF cases")
                checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
                require(checkpoint["fingerprint"] == manifest["fingerprint"], "Checkpoint identity mismatch")
                model = AttentionMapRanker(checkpoint["input_dim"], hidden=checkpoint["hidden"], variant=variant)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                old_prediction = read(original / relative / "predictions.json") if representation == "lexical" else None
                for cid, row in prediction.items():
                    with torch.no_grad():
                        replay = model(*data[cid])
                    require(np.array_equal(replay["scores"].numpy(), row["scores"]), "Checkpoint replay differs")
                    require(np.allclose(np.asarray(row["local"]) + row["relation"], row["scores"], atol=1e-6), "Residual mismatch")
                    replayed += 1
                    if old_prediction is not None:
                        require(row == old_prediction[cid], "Re-trained lexical predictions differ from v1")
                        original_equal += 1
                    if "attention" in replay:
                        with np.load(out / "maps" / variant / f"seed-{seed}" / f"{cid}.npz") as z:
                            a = z["attention"]
                            require(np.array_equal(a, replay["attention"].numpy()), "Map replay differs")
                            require(np.allclose(a.sum(1), 1, atol=1e-6) and not np.triu(a, 1).any(), "Invalid temporal map")
                            require((a >= 0).all(), "Negative attention")
                            require(np.array_equal(z["seeds"], replay["seeds"].numpy()), "Seed replay differs")
                            require(np.array_equal(z["mass"], replay["mass"].numpy()), "Mass replay differs")
                            if variant == "fixed_graph":
                                allowed = (data[cid][2].numpy() > 0) | np.eye(len(a), dtype=bool)
                                require(not a[~allowed].any(), "Fixed graph has unsupported edges")
                            maps += 1
                cost += status["seconds"]
                fits += 1
                active_parameters.setdefault(variant, set()).add(status["parameters_with_gradient"])
                combined.update(prediction)
            require(set(combined) == set(cases), "Incomplete OOF predictions")
            predictions[variant][seed] = combined
            for score_key, suffix in MODES.items():
                if score_key not in next(iter(combined.values())):
                    continue
                method = variant + suffix
                measured = {cid: measure(row[score_key], labels[cid]) for cid, row in combined.items()}
                metrics.setdefault(method, {})[seed] = measured
                expected = summary["methods"][method]["per_seed"][manifest["config"]["seeds"].index(seed)]
                for key in METRICS:
                    require(abs(np.mean([r[key] for r in measured.values()]) - expected[key]) < 1e-12, "Metric mismatch")
    checks = {"fits": fits, "replayed_predictions": replayed, "verified_maps": maps,
              "old_predictions_identical": original_equal, "fit_seconds": cost,
              "active_parameters": {k: sorted(v) for k, v in active_parameters.items()}}
    return metrics, checks, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/attentionmap-semantic-v2")
    args = parser.parse_args()
    out = args.out
    experiment = read(out / "experiment.json")
    original = Path(experiment["source"])
    historical = read(out / "historical_files.json")
    for name, expected in historical.items():
        require(sha(original / name) == expected, "Historical asset changed")
    for name, expected in experiment["source_hashes"].items():
        require(sha(out / "snapshot" / name) == expected, "Snapshot drift")
    encoding = read(out / "encoding.json")
    require(sha(out / "encoding.json") == experiment["encoding_sha256"], "Encoding log drift")
    manifests = {name: read(out / name / "manifest.json") for name in ("lexical", "semantic")}
    cases = {c["case_id"]: c for c in manifests["lexical"]["cases"]}
    for left, right in zip(manifests["lexical"]["cases"], manifests["semantic"]["cases"]):
        require({k: v for k, v in left.items() if k != "feature_sha256"} ==
                {k: v for k, v in right.items() if k != "feature_sha256"}, "Paired cases differ")
    source_count = 0
    for c in cases.values():
        for source in c["sources"].values():
            require(sha(Path(source["path"])) == source["sha256"], "Raw source drift")
            source_count += 1
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    measures, checks, predictions = {}, {}, {}
    for name in ("lexical", "semantic"):
        measures[name], checks[name], predictions[name] = inspect_run(out / name, original, name)
    require(checks["lexical"]["active_parameters"] == checks["semantic"]["active_parameters"], "Parameter budget differs")
    seeds = manifests["semantic"]["config"]["seeds"]
    labels = read(out / "semantic/labels.json")
    baselines = read(out / "semantic/baselines.json")
    rrf = {cid: measure(row["rrf"], labels[cid]) for cid, row in baselines.items()}
    measures["fixed"] = {"rrf": {s: rrf for s in seeds}}
    summaries = {}
    for representation, methods in measures.items():
        for method, runs in methods.items():
            zero = [cid for cid, c in cases.items() if c["old_zero_root"]]
            per_seed = [{"seed": seed, "top1_count": sum(row["top1"] for row in run.values()),
                         **{k: float(np.mean([row[k] for row in run.values()])) for k in METRICS}}
                        for seed, run in runs.items()]
            summaries[f"{representation}/{method}"] = {
                **{k: float(np.mean([r[k] for r in per_seed])) for k in METRICS}, "per_seed": per_seed,
                "zero_root_n": len(zero),
                "zero_root_mean_hits": float(np.mean([sum(run[c]["top1"] for c in zero) for run in runs.values()]))}
    comparisons = {}
    for method in manifests["semantic"]["config"]["variants"]:
        comparisons[f"semantic/{method} - lexical/{method}"] = paired(measures["semantic"][method], measures["lexical"][method], cases)
    for reference in ("linear", "mlp", "fixed_graph", "dense_unconditioned", "dense_no_prior", "dense_shuffled"):
        comparisons[f"semantic/dense - semantic/{reference}"] = paired(measures["semantic"]["dense"], measures["semantic"][reference], cases)
    comparisons["semantic/dense - fixed/rrf"] = paired(measures["semantic"]["dense"], measures["fixed"]["rrf"], cases)
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    result = {"n": len(cases), "groups": len({c["group"] for c in cases.values()}),
              "methods": summaries, "comparisons": comparisons, "encoding": encoding,
              "verification": {"historical_files_unchanged": len(historical), "source_files_unchanged": source_count,
                               "repository_and_task_disjoint": True, "matched_parameter_budget": True, **checks}}
    save(dest / "comparison.json", result)
    save(dest / "per_case_metrics.json", measures)
    with (dest / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["representation", "method", *METRICS, "zero_root_mean_hits"])
        writer.writeheader()
        for name, row in summaries.items():
            representation, method = name.split("/", 1)
            writer.writerow({"representation": representation, "method": method,
                             **{k: row[k] for k in (*METRICS, "zero_root_mean_hits")}})
    names = ["linear", "mlp", "fixed_graph", "dense_unconditioned", "dense", "dense_no_prior"]
    display = ["线性排序", "MLP", "固定图 attention", "关系不注入失败", "Attention map", "无图先验"]
    dense = summaries["semantic/dense"]
    delta = comparisons["semantic/dense - lexical/dense"]
    lines = ["# 冻结语义表示对照结果", "",
             f"语义 Attention map 的 Top-1 为 {dense['top1']:.2%}，Top-5 为 {dense['top5']:.2%}，MRR 为 {dense['mrr']:.4f}。"
             f"相对旧词法表示，Top-1 变化 {delta['top1_delta'] * 100:+.2f} 个百分点。", "",
             "102 条、35 个仓库/任务组、5 折、3 种子。两种表示各运行 90 次训练；模型架构、可训练参数量、初始化、训练预算、数值特征与图输入相同。", "",
             "| 方法 | 词法 Top-1 | 语义 Top-1 | 词法 Top-5 | 语义 Top-5 | 词法 MRR | 语义 MRR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, label in zip(names, display):
        a, b = summaries[f"lexical/{method}"], summaries[f"semantic/{method}"]
        lines.append(f"| {label} | {a['top1']:.2%} | {b['top1']:.2%} | {a['top5']:.2%} | {b['top5']:.2%} | {a['mrr']:.4f} | {b['mrr']:.4f} |")
    fixed = summaries["fixed/rrf"]
    lines += ["", f"固定 RRF：Top-1 {fixed['top1']:.2%}，Top-5 {fixed['top5']:.2%}，MRR {fixed['mrr']:.4f}。", "",
              "## 种子与配对变化", "", "| 语义方法 | 种子 17 命中 | 种子 29 命中 | 种子 43 命中 | 旧传播零分 54 例平均命中 |", "|---|---:|---:|---:|---:|"]
    for method, label in zip(names, display):
        row = summaries[f"semantic/{method}"]
        counts = " | ".join(str(r["top1_count"]) for r in row["per_seed"])
        lines.append(f"| {label} | {counts} | {row['zero_root_mean_hits']:.2f} |")
    lines += ["", "配对区间按仓库/任务联合组重采样 2,000 次；每个案例先取三种子平均。", "",
              "| 比较（前者减后者） | Top-1 差值（百分点） | 95% 组 bootstrap 区间 | 各种子新增/丢失 |", "|---|---:|---:|---|"]
    for name, row in comparisons.items():
        low, high = row["group_bootstrap_95ci"]
        counts = "; ".join(f"{p['seed']}: +{p['gained']}/−{p['lost']}" for p in row["per_seed"])
        lines.append(f"| {name} | {100 * row['top1_delta']:+.2f} | [{100 * low:.2f}, {100 * high:.2f}] | {counts} |")
    lines += ["", "## 机制消融", "", "| 语义 Attention map | Top-1 | Top-5 | MRR |", "|---|---:|---:|---:|"]
    for method in ("dense", "dense_shuffled", "dense_drop50", "dense_drop100"):
        row = summaries[f"semantic/{method}"]
        lines.append(f"| {method} | {row['top1']:.2%} | {row['top5']:.2%} | {row['mrr']:.4f} |")
    lines += ["", "## 实现与复现核验", "",
              f"冻结 MiniLM：{encoding['unique_texts']:,} 段唯一文本，{encoding['content_tokens']:,} 个内容 token，{encoding['chunks']:,} 个块；编码 {encoding['seconds']:.1f} 秒。所有块均参与加权合并，384 维以固定随机正交投影压为 128 维。预训练编码器可训练参数为 0。", "",
              f"两组训练累计 {checks['lexical']['fit_seconds'] + checks['semantic']['fit_seconds']:.1f} 秒；排序器 CPU 单线程，编码器 {encoding['identity']['device']}。新增推理 API 请求 0；首次下载公开模型权重。已有 LLM 图/q 缓存生成成本仍为复用成本。", "",
              f"原实验 {len(historical):,} 个文件及 {source_count} 份来源保持原哈希。词法重跑的 {checks['lexical']['old_predictions_identical']} 份逐例预测及扰动分数与旧实验完全相同。共 {sum(c['replayed_predictions'] for c in checks.values())} 份预测由保存模型逐一重放，{sum(c['verified_maps'] for c in checks.values())} 张 attention 矩阵与重放逐元素一致。", "",
              "RootSE 已参与历史开发，本轮是冻结配置的分组开发比较。独立新任务评估、受控重执行与等端到端预算实验仍在原研究计划中。", "",
              "![表示对照](comparison.png)", "",
              "协议：experiments/attentionmap/SEMANTIC_PROTOCOL.md；入口：scripts/reproduce_attentionmap_semantic.sh。comparison.json 保留全指标与区间；per_case_metrics.json 保留逐案例/种子明细。"]
    (dest / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels_en = ["Linear", "MLP", "Fixed graph", "No failure query", "Attention map", "No graph prior"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    y = np.arange(len(names))
    for ax, metric in zip(axes, ("top1", "top5")):
        for representation, shift, color in (("lexical", -.18, "#9da9b6"), ("semantic", .18, "#247b88")):
            values = [summaries[f"{representation}/{method}"][metric] * 100 for method in names]
            ax.barh(y + shift, values, height=.33, color=color, label=representation.capitalize())
            for i, method in enumerate(names):
                for row in summaries[f"{representation}/{method}"]["per_seed"]:
                    ax.plot(row[metric] * 100, y[i] + shift, ".", color="#1a2834", markersize=3)
                label_x = max(row[metric] * 100 for row in summaries[f"{representation}/{method}"]["per_seed"]) + 1
                ax.text(label_x, y[i] + shift, f"{values[i]:.1f}", va="center", fontsize=8)
        ax.axvline(fixed[metric] * 100, ls="--", lw=1, color="#aa5c37", label="Fixed RRF")
        ax.set_yticks(y, labels_en)
        ax.invert_yaxis()
        ax.set_title("Top-1 root localization" if metric == "top1" else "Top-5 candidate coverage")
        ax.set_xlabel("Percent of 102 trajectories")
        ax.set_xlim(0, max(summaries[f"{r}/{m}"][metric] * 100 for r in ("lexical", "semantic") for m in names) + 10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Frozen semantic representation · matched 128-d inputs and rankers\nRootSE grouped development OOF · 5 folds · 3 seeds; points: seeds", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    # Reuse the previous round's example, fixed before inspecting new outcomes.
    example = read(original / "analysis/illustrative_case.json")
    cid = example["case_id"]
    case = cases[cid]
    gold = labels[cid][0]
    seed = seeds[0]
    case_result = {"case_id": cid, "trajectory_id": case["trajectory_id"],
                   "selection": "Same case as v1, fixed before semantic results",
                   "seed_displayed": seed, "root_step_id": case["step_ids"][gold],
                   "ranks": {f"{r}/{m}": [measures[r][m][s][cid]["rank"] for s in seeds]
                             for r in ("lexical", "semantic") for m in names}}
    save(dest / "fixed_case.json", case_result)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    n = case["n_steps"]
    ticks = np.unique(np.linspace(0, n - 1, min(n, 7)).astype(int))
    for ax, representation in zip(axes[:2], ("lexical", "semantic")):
        with np.load(out / representation / "maps/dense" / f"seed-{seed}" / f"{cid}.npz") as z:
            attention = z["attention"]
        ax.imshow(attention / attention.max(axis=1, keepdims=True), cmap="viridis", vmin=0, vmax=1)
        rank = measures[representation]["dense"][seed][cid]["rank"]
        ax.set_title(f"{representation.capitalize()} attention · root rank {rank}")
        ax.axvline(gold, color="#e8594e", ls="--", lw=1.2)
        ax.set_xticks(ticks, [case["step_ids"][i] for i in ticks])
        ax.set_yticks(ticks, [case["step_ids"][i] for i in ticks])
        ax.set_xlabel("Earlier source step ID")
        ax.set_ylabel("Later target step ID")
    for representation, color in (("lexical", "#8795a1"), ("semantic", "#247b88")):
        scores = predictions[representation]["dense"][seed][cid]["scores"]
        probability = torch.softmax(torch.tensor(scores), dim=0).numpy()
        axes[2].plot(np.arange(n), probability, label=representation.capitalize(), color=color)
    axes[2].axvline(gold, color="#e8594e", ls="--", label=f"Gold step {case['step_ids'][gold]}")
    axes[2].set_xticks(ticks, [case["step_ids"][i] for i in ticks])
    axes[2].set_xlabel("Original step ID")
    axes[2].set_ylabel("Softmax of final ranking logits")
    axes[2].set_title("Final score distribution")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{case['task_id'][:70]} · seed {seed}\nSame example as v1; maps normalized by each row maximum", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(dest / f"fixed_case.{suffix}", dpi=180)
    plt.close(fig)
    with (dest / "report.md").open("a") as handle:
        handle.write("\n## 固定案例\n\n沿用上轮案例，展示种子 17；图按每行最大值缩放。各种子与六模型的原始根因排名保存在 fixed_case.json。\n\n![同一案例](fixed_case.png)\n")
    print(json.dumps(result["verification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
