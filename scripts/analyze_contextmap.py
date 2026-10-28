#!/usr/bin/env python3
"""Verify recorded responses and analyze matched context-reader experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_contextmap import ROOT, OUT, GRAPH, METHODS, read, sha, prepare
from suite_transport import digest, save
from loghound_agent_debug.context_map import parse_answer

NAMES = {"full": "全文", "window": "窗口/检索", "graph": "原图上下文", "attention": "Attention map", "attention_flat": "Map 同事实平铺"}
ENGLISH = {"full": "Full trajectory", "window": "Window / retrieval", "graph": "Original graph",
           "attention": "Attention map", "attention_flat": "Same excerpts, flat"}


def grouped(values: dict[str, float], cases: dict, paired: bool = False) -> dict:
    groups = sorted({c["group"] for c in cases.values()})
    total = np.array([sum(values[c] for c in cases if cases[c]["group"] == g) for g in groups])
    sizes = np.array([sum(cases[c]["group"] == g for c in cases) for g in groups])
    rng = np.random.default_rng(20261028)
    chosen = rng.integers(len(groups), size=(5000, len(groups)))
    bootstrap = total[chosen].sum(1) / sizes[chosen].sum(1)
    observed = float(total.sum() / sizes.sum())
    result = {"mean": observed, "ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    if paired:
        null = rng.choice([-1, 1], size=(20000, len(groups))) @ total / sizes.sum()
        result["p"] = float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (len(null) + 1))
    return result


def account(directory: Path) -> dict:
    totals = Counter()
    for p in directory.glob("**/calls/*/state.json"):
        state = read(p)
        totals["logical_calls"] += 1
        for a in state["attempts"]:
            totals["requests"] += 1
            if a.get("usage") is None:
                totals["unknown_usage_requests"] += 1
            else:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    totals[k] += a["usage"].get(k, 0)
    return dict(totals)


def read_sse(path: Path) -> tuple[str, str]:
    content, reasoning = [], []
    done = False
    for line in path.read_text().splitlines():
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            done = True
            break
        chunk = json.loads(value)
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            if any(delta.get(k) for k in ("tool_calls", "function_call", "web_search_call")):
                raise ValueError("Unexpected tool output")
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
    if not done:
        raise ValueError("Incomplete SSE")
    return "".join(content), "".join(reasoning)


def analyze(out: Path) -> dict:
    manifest = prepare(out)
    if read(out / "run_status.json").get("completed") != len(METHODS) * manifest["n"]:
        raise ValueError("Require complete experiment before analysis")
    cases = {r["case_id"]: r for r in manifest["cases"]}
    labels = read(out / "labels.json")
    historical = read(out / "historical_hashes.json")
    if sha(out / "historical_hashes.json") != manifest["historical_hashes_sha256"]:
        raise ValueError("Historical manifest drift")
    for p, expected in historical.items():
        if sha(Path(p)) != expected:
            raise ValueError("Historical artifact modified: " + p)
    rows, predictions = [], {}
    verified_sse = verified_answers = 0
    raw_hashes = {}
    for method in METHODS:
        predictions[method] = {}
        for cid, c in cases.items():
            prepared = read(out / "prepared" / method / f"{cid}.json")
            directory = out / "runs" / method / cid
            state = read(directory / "state.json")
            call = directory / "calls/0001"
            request, callstate = read(call / "request.json"), read(call / "state.json")
            if request["messages"] != prepared["messages"] or digest(request) != callstate["request_sha256"]:
                raise ValueError("Actual request differs from prepared evidence")
            if request.get("model") != manifest["config"]["model"] or request.get("enable_thinking") is not True:
                raise ValueError("Model/thinking protocol mismatch")
            if any(request.get(k) for k in ("tools", "functions", "enable_search", "search_options")):
                raise ValueError("Unexpected search/tool request")
            if request["max_tokens"] != 32768 or request["temperature"] != 0:
                raise ValueError("Output protocol mismatch")
            decision = None
            reasoning_chars = 0
            if callstate["status"] == "received":
                responsepath = call / callstate["response_file"]
                response = read(responsepath)
                message = response["choices"][0]["message"]
                rawcontent, rawreasoning = read_sse(responsepath.with_suffix(".sse"))
                if rawcontent != (message.get("content") or "") or rawreasoning != (message.get("reasoning_content") or ""):
                    raise ValueError("SSE differs from normalized response")
                verified_sse += 1
                reasoning_chars = len(rawreasoning)
                if state["status"] == "valid":
                    decision = parse_answer(rawcontent, prepared["step_ids"])
                    if decision != state["decision"] or response["choices"][0]["finish_reason"] != "stop":
                        raise ValueError("Parsed result differs from raw response")
                    verified_answers += 1
            for p in directory.rglob("*"):
                if p.is_file():
                    raw_hashes[str(p.relative_to(out))] = sha(p)
            gold = labels[cid]
            ranking = decision["ranked_step_ids"] if decision else []
            rank = min((ranking.index(g) + 1 for g in gold if g in ranking), default=None)
            selected = prepared["view"]["selected_ids"]
            full = read(out / "prepared/full" / f"{cid}.json")
            gold_full_chars = sum(len(f["text"]) for f in full["view"]["fragments"] if f["step_id"] in gold)
            gold_read_chars = sum(len(f["text"]) for f in prepared["view"]["fragments"] if f["step_id"] in gold)
            usage = Counter()
            unknown_usage = 0
            seconds = 0.
            for attempt in callstate["attempts"]:
                seconds += attempt.get("elapsed_seconds", 0)
                if attempt.get("usage") is None:
                    unknown_usage += 1
                else:
                    usage.update({k: attempt["usage"].get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
            row = {"method": method, "case_id": cid, "group": c["group"], "n_steps": c["n_steps"],
                   "status": state["status"], "gold": gold, "prediction": ranking[0] if ranking else None,
                   "ranked_step_ids": ranking, "rank": rank, "top1": int(rank == 1),
                   "top5": int(rank is not None), "mrr_at5": 1 / rank if rank else 0.,
                   "root_expanded": int(bool(set(gold) & set(selected))),
                   "root_excerpt_fraction": gold_read_chars / max(1, gold_full_chars),
                   "selected_steps": len(selected), "context_tokens_proxy": prepared["view"]["context_tokens"],
                   "message_tokens_proxy": prepared["message_tokens_proxy"], "requests": len(callstate["attempts"]),
                   "unknown_usage_requests": unknown_usage, "seconds": seconds,
                   "reasoning_chars": reasoning_chars, **dict(usage)}
            rows.append(row)
            predictions[method][cid] = row
    for cid in cases:
        a = read(out / "prepared/attention" / f"{cid}.json")
        b = read(out / "prepared/attention_flat" / f"{cid}.json")
        if a["view"]["fragments"] != b["view"]["fragments"]:
            raise ValueError("Same-fact control differs")
    summaries = {}
    for method in METHODS:
        selected = list(predictions[method].values())
        valid = [r for r in selected if r["status"] == "valid"]
        expanded = [r for r in selected if r["root_expanded"]]
        summaries[method] = {"n": len(selected), "valid": len(valid), "hits": sum(r["top1"] for r in selected),
            "top5_hits": sum(r["top5"] for r in selected), "status_counts": dict(Counter(r["status"] for r in selected)),
            "metrics": {metric: grouped({r["case_id"]: r[metric] for r in selected}, cases)
                        for metric in ("top1", "top5", "mrr_at5", "root_expanded")},
            "valid_top1": sum(r["top1"] for r in valid) / max(1, len(valid)),
            "expanded_top1": sum(r["top1"] for r in expanded) / max(1, len(expanded)),
            "expanded_n": len(expanded), "unexpanded_hits": sum(r["top1"] for r in selected if not r["root_expanded"]),
            "root_excerpt_fraction_mean": float(np.mean([r["root_excerpt_fraction"] for r in selected])),
            "input_tokens_median": float(np.median([r["prompt_tokens"] for r in selected if "prompt_tokens" in r])) if valid else None,
            "response_seconds_median": float(np.median([r["seconds"] for r in selected])),
            "cost": account(out / "runs" / method)}
    contrasts = {}
    for other in ("graph", "window", "attention_flat", "full"):
        left, right = predictions["attention"], predictions[other]
        contrast = grouped({c: left[c]["top1"] - right[c]["top1"] for c in cases}, cases, paired=True)
        contrast.update(gained=sum(left[c]["top1"] > right[c]["top1"] for c in cases),
                        lost=sum(left[c]["top1"] < right[c]["top1"] for c in cases))
        contrasts[other] = contrast
    running = 0.
    ordered = sorted(contrasts, key=lambda k: contrasts[k]["p"])
    for i, k in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - i) * contrasts[k]["p"]))
        contrasts[k]["p_holm"] = running
    result = {"experiment": manifest["fingerprint"], "n": len(cases), "groups": manifest["groups"], "runs_per_method": 1,
              "methods": summaries, "contrasts_attention_minus": contrasts,
              "prepare_seconds": manifest["prepare_seconds"],
              "historical_graph_cost": {"formal": account(GRAPH / "targets"), "pilot": account(GRAPH / "pilot")},
              "rejected_credential_preflight_cost": account(out / "preflight_rejected"),
              "credential_probe_cost": account(ROOT / "outputs/attentionmap-context-credential-probe-20261028")}
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    save(analysis / "summary.json", result)
    save(analysis / "predictions.json", predictions)
    save(analysis / "raw_hashes.json", raw_hashes)
    save(analysis / "verification.json", {"responses_reparsed": verified_answers, "sse_verified": verified_sse,
        "prepared_inputs_checked": len(rows), "same_fact_pairs": len(cases), "source_files_unchanged": len(manifest["source_hashes"]),
        "historical_files_unchanged": len(historical), "actual_payloads_match": True, "search_or_tool_requests": 0,
        "analysis_script_sha256": sha(Path(__file__))})
    with (analysis / "paired_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    write_reports(result, analysis)
    figures(result, analysis)
    print(json.dumps({"hits": {m: summaries[m]["hits"] for m in METHODS}, "contrasts": contrasts,
                      "verification": read(analysis / "verification.json")}, ensure_ascii=False, indent=2))
    return result


def write_reports(result: dict, out: Path) -> None:
    lines = ["# 上下文组织端到端实验", "", "固定 Qwen、thinking 与诊断指令，比较上下文选择和组织。102 条、35 个仓库/任务组，每组一次服务运行。", "",
             "| 上下文 | 有效输出 | Top-1 | Top-5 | MRR@5 | 根因展开 | 实际输入 token 中位数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        s = result["methods"][method]
        lines.append(f"| {NAMES[method]} | {s['valid']}/102 | {s['hits']}/102 ({s['metrics']['top1']['mean']:.2%}) | {s['top5_hits']}/102 ({s['metrics']['top5']['mean']:.2%}) | {s['metrics']['mrr_at5']['mean']:.4f} | {s['expanded_n']}/102 | {s['input_tokens_median']} |")
    lines += ["", "## 预先指定的配对比较", "", "| Attention map 减去 | Top-1 差值 | 组 bootstrap 95% CI | 新增/丢失命中 | Holm p |", "|---|---:|---|---:|---:|"]
    for method, c in result["contrasts_attention_minus"].items():
        lines.append(f"| {NAMES[method]} | {100*c['mean']:+.2f} pp | [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] pp | +{c['gained']}/−{c['lost']} | {c['p_holm']:.4f} |")
    lines += ["", "## 可据此解释的范围", "",
              "Top-1/Top-5 以全部 102 条为分母；无有效输出计未命中。MRR@5 对根因未进入返回列表的案例计 0。根因展开仅表示其部分原文可见，完整证据链另需逐例审阅。",
              "", "有界组的 16,384 token 是 cl100k_base 轨迹视图预算，真实 Qwen token 数另列。任务、测试和共享提示词在所有组中保持相同。全文组是全量参照。",
              "", "原图和 attention 复用已有图与冻结表示；新增诊断成本不能代表从零运行总成本。历史图的生成用量见 summary.json。", "",
              "## Claim Candidates", ""]
    for method, c in result["contrasts_attention_minus"].items():
        lines += [f"- Claim: 本轮 Attention map 相对 {NAMES[method]} 的 Top-1 差值为 {100*c['mean']:+.2f} 个百分点。",
                  "  - Source evidence: paired_cases.csv、summary.json 与已保存请求/响应。",
                  "  - Allowed wording: 描述本轮配对效应及上述区间。",
                  "  - Forbidden stronger wording: 不据一次开发集服务运行声称跨任务普遍优势。",
                  f"  - Uncertainty: 组 bootstrap 区间 {c['ci95']}，Holm p={c['p_holm']:.4f}。",
                  "  - Next check: 审阅分歧案例和独立数据复现。", "  - Decision: keep"]
    (out / "analysis-report.md").write_text("\n".join(lines) + "\n")
    stats = ["# 统计附录", "", "分析单位为轨迹，相关性单位为仓库/任务联合组。102 条属于 35 组。每个方法运行一次，不将五组方法当独立样本。",
             "", "Top-1 是二元变量，组大小不等，使用非参数方法。95% 区间采用 5,000 次组 bootstrap，抽取整组后按抽中轨迹数加权。配对检验在每组内整体交换两个方法的标签，20,000 次随机化，双侧 p 含加一校正。四个预先指定比较采用 Holm 校正。随机种子 20261028。",
             "", "差值为配对 Top-1 风险差（百分点）；新增和丢失命中辅助解释方向。单次运行没有模型重复种子的均值±标准差，区间不估计服务随机性。缺失/无效输出按主评估分母计 0，条件有效率单列。", "",
             "完整描述统计、置信区间、原始与校正 p 值、成本和有效率均见 summary.json。"]
    (out / "stats-appendix.md").write_text("\n".join(stats) + "\n")


def figures(result: dict, out: Path) -> None:
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, offset, color in (("top1", -.18, "#235789"), ("top5", .18, "#72A98F")):
        means = np.array([result["methods"][m]["metrics"][metric]["mean"] for m in METHODS])
        ci = np.array([result["methods"][m]["metrics"][metric]["ci95"] for m in METHODS])
        ax.bar(x + offset, means * 100, width=.34, color=color, label=metric.upper().replace("TOP", "Top-"))
        ax.errorbar(x + offset, means * 100, yerr=np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means])) * 100,
                    fmt="none", ecolor="#444444", capsize=3)
    ax.set_xticks(x, [ENGLISH[m].replace(" ", "\n", 1) for m in METHODS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Localization accuracy (%)")
    ax.set_title("Same diagnostic LLM, different trajectory context")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"01-localization.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    selected = METHODS[1:]
    axes[0].barh(np.arange(len(selected)), [100 * result["methods"][m]["expanded_n"] / result["n"] for m in selected], color="#9CC5A1", label="Root excerpt shown")
    axes[0].scatter([100 * result["methods"][m]["expanded_top1"] for m in selected], np.arange(len(selected)), color="#B44436", marker="D", s=45, label="Top-1 when shown")
    axes[0].set_yticks(np.arange(len(selected)), [ENGLISH[m] for m in selected])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Percentage (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    for i, m in enumerate(METHODS):
        s = result["methods"][m]
        known = s["cost"].get("prompt_tokens", 0)
        axes[1].scatter(known / result["n"] / 1000, s["metrics"]["top1"]["mean"] * 100, s=60)
        axes[1].annotate(ENGLISH[m], (known / result["n"] / 1000, s["metrics"]["top1"]["mean"] * 100),
                         xytext=(4, (i % 3 - 1) * 13 + 5), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Known prompt tokens / case (thousands)")
    axes[1].set_ylabel("Top-1 (%)")
    axes[1].margins(x=.25, y=.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"02-coverage-cost.{ext}", dpi=180)
    plt.close(fig)
    flat = result["contrasts_attention_minus"]["attention_flat"]
    (out / "figure-catalog.md").write_text(
        "# 图表目录\n\n"
        "## 01-localization.png / .pdf\n\n"
        "目的：比较相同诊断模型下的定位准确率。数据来自 summary.json；柱表示全 102 条的 Top-1/Top-5，误差条是 35 组 bootstrap 95% CI。图注需注明每组一次运行、全文为全量参照、其余为有界视图。\n\n"
        f"解读：同事实组织对照 Top-1 差值为 {flat['mean']*100:+.2f} pp，Holm p={flat['p_holm']:.4f}。结合配对差值判断组织收益，不用两根柱的区间是否重叠代替配对检验。\n\n"
        "## 02-coverage-cost.png / .pdf\n\n"
        "目的：区分根因是否进入展开原文与读到后是否判断正确，并展示真实 API 输入量。左侧绿色条是根因展开率，红点是已展开子集 Top-1；二者分母不同。右侧显示已知输入 token / 全部案例与 Top-1。\n\n"
        "解读：展开率高而条件定位率低时，单纯增加候选覆盖不能解释差距；还需审阅证据完整性和模型推理。右图仅为新增诊断成本，复用图的历史生成成本另列。缺失用量不能当作零。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    analyze(args.out)


if __name__ == "__main__":
    main()
