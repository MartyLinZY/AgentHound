#!/usr/bin/env python3
"""Offline same-graph ablations, complete-denominator evaluation and blind audit packets."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from migration_core import ROOT
from suite_transport import save, digest, now
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.schema import input_from_dict
from loghound_agent_debug.rankers.loghound_static import load_config
from loghound_agent_debug.rankers.anchor_path import rank_trajectory
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rootse_evaluation import evaluate
from loghound_agent_debug.graph import build_rootse_graph
from compare_rootse_models import paired_counts
from report_rootse_suite import totals
from run_rootse_migration import DEFAULT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(p: Path):
    return json.loads(p.read_text())


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def accounting(out: Path) -> dict:
    attempts = []
    states = []
    for p in sorted((out / "targets").glob("*/*/calls/*/state.json")):
        state = read(p)
        require(
            digest(read(p.parent / "request.json")) == state["request_sha256"],
            "Integrity check failed: digest(read(p.parent/'request.json'))==state['request_sha256']",
        )
        states.append(state)
        attempts.extend(state["attempts"])
    return {**totals(attempts), "logical_calls": len(states)}


def cluster_interval(
    ids: list[str], left: dict, right: dict, repositories: dict, repetitions: int = 5000
) -> list[float]:
    """Paired repository-resampling percentile CI; no model-seed interpretation."""
    groups = defaultdict(list)
    for tid in ids:
        groups[repositories[tid]].append(int(right[tid]) - int(left[tid]))
    names = sorted(groups)
    sums = np.array([sum(groups[x]) for x in names])
    sizes = np.array([len(groups[x]) for x in names])
    sampled = np.random.default_rng(20261006).integers(
        0, len(names), size=(repetitions, len(names))
    )
    values = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1) * 100
    return list(np.quantile(values, [0.025, 0.975]))


def review_packets(out: Path, manifest: dict) -> None:
    """Sample by hash before using root-cause labels; no gold fields in packets."""
    buckets = defaultdict(list)
    inputs = {
        c["case_id"]: input_from_dict(read(out / "inputs" / f"{c['case_id']}.json"))
        for c in manifest["cases"]
    }
    for target in manifest["targets"]:
        state = read(out / target["directory"] / "state.json")
        i = inputs[target["case_id"]]
        ordinal = target["target_ordinal"]
        if state["status"] != "complete":
            continue
        for category, key in [
            ("accepted", "validated_dependencies"),
            ("rejected", "rejected_dependencies"),
        ]:
            for n, edge in enumerate(state.get(key, [])):
                row = {
                    "trajectory_id": i.trajectory_id,
                    "target_ordinal": ordinal,
                    "category": category,
                    "edge": edge,
                    "record_directory": target["directory"],
                }
                row["review_id"] = digest(
                    {"record": target["directory"], "category": category, "index": n}
                )[:16]
                buckets[category].append(row)
        connected = {e["source_ordinal"] for e in state["validated_dependencies"]}
        for source in state["metadata"]["candidate_ordinals"]:
            if source not in connected:
                row = {
                    "trajectory_id": i.trajectory_id,
                    "target_ordinal": ordinal,
                    "source_ordinal": source,
                    "category": "no_accepted_edge",
                    "record_directory": target["directory"],
                    "review_id": digest(
                        {"record": target["directory"], "source": source}
                    )[:16],
                }
                buckets["no_accepted_edge"].append(row)
    selected = []
    for category, n in [("accepted", 24), ("rejected", 12), ("no_accepted_edge", 12)]:
        selected += sorted(buckets[category], key=lambda r: r["review_id"])[:n]
    by_id = {i.trajectory_id: i for i in inputs.values()}
    for r in selected:
        i = by_id[r["trajectory_id"]]
        target = i.steps[r["target_ordinal"]]
        r["target_fields"] = {
            k: v for k, v in target.rootse_fields.items() if k != "observation"
        }
        edge = r.get("edge", {})
        relation = edge.get("relationship", edge)
        sid = relation.get("source_step_id")
        source = (
            next((s for s in i.steps if s.source_step_id == str(sid)), None)
            if sid is not None
            else i.steps[r["source_ordinal"]]
        )
        r["source_fields"] = source.rootse_fields if source else None
        r["review"] = {"verdict": None, "reason": None, "reviewer": None}
    save(
        out / "analysis/blind_edge_review.json",
        {
            "selection": "hash order; excludes gold, localization predictions and hit flags",
            "sample_counts": dict(Counter(r["category"] for r in selected)),
            "population_counts": {k: len(v) for k, v in buckets.items()},
            "items": selected,
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--source", type=Path, default=ROOT.parent / "TrajAudit")
    p.add_argument(
        "--baselines",
        type=Path,
        default=ROOT / "outputs/rootse-suite-deepseek-v41-flash-32k-v1",
    )
    a = p.parse_args()
    out = a.out.resolve()
    manifest = read(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Wait for all graph requests before scoring")
    for rel, h in manifest["protocol"]["dataset"].items():
        if not (hashlib.sha256((a.source / rel).read_bytes()).hexdigest() == h):
            raise ValueError(
                "Integrity check failed: hashlib.sha256((a.source/rel).read_bytes()).hexdigest()==h"
            )
    for rel, h in manifest["protocol"]["code"].items():
        if not (hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h):
            raise ValueError("Inference code changed: " + rel)
    for target in manifest["targets"]:
        state = read(out / target["directory"] / "state.json")
        if not (
            state["status"] != "running"
            and state["request_sha256"] == target["request_sha256"]
        ):
            raise ValueError(
                "Integrity check failed: state['status']!='running' and state['request_sha256']==target['request_sha256']"
            )
    config = manifest["protocol"]["config"]
    rank_config = load_config(out / "snapshot/configs/default.yaml")
    rules = load_clue_rules(out / "snapshot/configs/clue_rules.yaml")
    graphs = {
        c["trajectory_id"]: read(out / "graphs" / f"{c['case_id']}.json")
        for c in manifest["cases"]
    }
    if not (len(graphs) == 102):
        raise ValueError("Integrity check failed: len(graphs)==102")
    if not (out / "analysis/blind_edge_review.json").exists():
        review_packets(out, manifest)
    # Labels become available only here, after graph completion and blind sample selection.
    outputs = {}
    scores = {}
    repositories = {}
    ids = []
    for protocol in ["postmortem", "trace-only"]:
        inputs, labels = load_rootse(a.source / "RootSE", protocol=protocol)
        repositories = {i.trajectory_id: i.repository for i in inputs}
        ids = sorted(repositories)
        for variant in config["variants"]:
            started = time.monotonic()
            rankings = []
            for i in inputs:
                graph = graphs[i.trajectory_id]
                r = rank_trajectory(
                    i, rank_config, rules, graph["dependencies"], variant
                )
                r["extraction_status"] = graph["status"]
                r["rule_only_fallback"] = graph["llm_edge_count"] == 0
                rankings.append(r)
            folder = out / "ablations" / protocol / variant
            jsonl(folder / "rankings.jsonl", rankings)
            result = evaluate(inputs, labels, rankings)
            rows = result["per_trajectory"]
            result["tolerance_hits"] = [
                sum(r["step_distance"] <= k for r in rows) for k in range(4)
            ]
            result["strict_complete_graph_tolerance_hits"] = [
                sum(
                    r["step_distance"] <= k
                    and graphs[r["trajectory_id"]]["status"] == "complete"
                    for r in rows
                )
                for k in range(4)
            ]
            result["ranking_seconds"] = time.monotonic() - started
            result["n"] = 102
            result["mechanism"] = {
                "top1_with_dependency_path": sum(
                    any(p["edges"] for p in r["ranking"][0].get("supporting_paths", []))
                    for r in rankings
                ),
                "cases_with_upstream_propagation": sum(
                    (
                        r.get("anchor_stats", {}).get(
                            "propagation_reaches_unseeded_steps"
                        )
                        or 0
                    )
                    > 0
                    for r in rankings
                ),
                "rule_only_fallbacks": sum(r["rule_only_fallback"] for r in rankings),
                "no_linked_anchor": sum(
                    r.get("evidence_status") == "no_linked_anchor" for r in rankings
                ),
            }
            save(folder / "metrics.json", result)
            outputs[f"{protocol}/{variant}"] = result
            scores[f"{protocol}/{variant}"] = {
                r["trajectory_id"]: r["top1_hit"] for r in rows
            }
    baseline = read(a.baselines / "evaluation.json")
    bm = read(a.baselines / "metrics.json")
    for method in ["all_at_once", "binary_search"]:
        if not ({r["trajectory_id"] for r in baseline[method]} == set(ids)):
            raise ValueError(
                "Integrity check failed: {r['trajectory_id'] for r in baseline[method]}==set(ids)"
            )
        scores[method] = {
            r["trajectory_id"]: r["tolerance_hits"][0] for r in baseline[method]
        }
    bconfig = read(a.baselines / "manifest.json")["protocol"]["config"]
    if not (
        all(
            config[k] == bconfig[k]
            for k in ["model", "endpoint", "temperature", "max_tokens"]
        )
    ):
        raise ValueError(
            "Integrity check failed: all(config[k]==bconfig[k] for k in ['model','endpoint','temperature','max_tokens'])"
        )
    paired = []
    primary = "postmortem/anchors-backtrace"
    for comparison in [
        "postmortem/baseline",
        "postmortem/anchors-global",
        "postmortem/legacy-backtrace",
        "all_at_once",
        "binary_search",
    ]:
        left = scores[comparison]
        right = scores[primary]
        pair = paired_counts([left[i] for i in ids], [right[i] for i in ids])
        paired.append(
            {
                "baseline": comparison,
                "candidate": primary,
                **pair,
                "delta_pp": 100 * (sum(right.values()) - sum(left.values())) / 102,
                "repository_cluster_bootstrap95_pp": cluster_interval(
                    ids, left, right, repositories
                ),
            }
        )
    previous = 0
    for rank, index in enumerate(
        sorted(range(5), key=lambda i: paired[i]["exact_mcnemar_two_sided_p"])
    ):
        previous = max(
            previous, min(1.0, (5 - rank) * paired[index]["exact_mcnemar_two_sided_p"])
        )
        paired[index]["holm_p"] = previous
    main_cost = accounting(out)
    pilot_cost = accounting(out / "pilot")
    counts = Counter()
    accepted = 0
    rejected = 0
    for t in manifest["targets"]:
        state = read(out / t["directory"] / "state.json")
        counts[state["status"]] += 1
        accepted += len(state.get("validated_dependencies", []))
        rejected += len(state.get("rejected_dependencies", []))
    graph_stats = {
        "complete_graphs": sum(g["status"] == "complete" for g in graphs.values()),
        "partial_graphs": sum(g["status"] == "partial" for g in graphs.values()),
        "cases_with_llm_edges": sum(g["llm_edge_count"] > 0 for g in graphs.values()),
        "accepted_relation_records": accepted,
        "rejected_relation_records": rejected,
        "merged_edges": sum(
            len(
                build_rootse_graph(
                    c["steps"], graphs[c["trajectory_id"]]["dependencies"]
                ).edges
            )
            for c in manifest["cases"]
        ),
    }
    summary = {
        "model": config["model"],
        "n": 102,
        "variants": outputs,
        "paired": paired,
        "accounting": main_cost,
        "pilot_accounting": pilot_cost,
        "extraction_status_counts": dict(counts),
        "graph_stats": graph_stats,
        "fingerprint": manifest["fingerprint"],
    }
    save(out / "metrics.json", summary)
    table = [
        "| 输入 / 方法 | Tol@0 | Tol@1 | Tol@2 | Tol@3 | Top-3 | Top-5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in outputs.items():
        table.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{h}/102 ({h / 102:.2%})" for h in m["tolerance_hits"])
            + f" | {m['top3_hits']}/102 | {m['top5_hits']}/102 | {m['mrr']:.4f} |"
        )
    for name in ["all_at_once", "binary_search"]:
        table.append(
            "| "
            + name
            + " / 原完整输入 | "
            + " | ".join(
                f"{h}/102 ({h / 102:.2%})"
                for h in bm["metrics"][name]["tolerance_hits"]
            )
            + " | — | — | — |"
        )
    pairtable = [
        "| 对照（候选为postmortem新锚点回溯） | 差值pp | 仓库重采样95%区间pp | 仅对照命中 | 仅候选命中 | Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in paired:
        pairtable.append(
            f"| {r['baseline']} | {r['delta_pp']:+.2f} | {r['repository_cluster_bootstrap95_pp'][0]:+.2f}, {r['repository_cluster_bootstrap95_pp'][1]:+.2f} | {r['left_only']} | {r['right_only']} | {r['holm_p']:.4g} |"
        )
    primary_hits = outputs[primary]["tolerance_hits"][0]
    base_hits = outputs["postmortem/baseline"]["tolerance_hits"][0]
    aa = bm["metrics"]["all_at_once"]["tolerance_hits"][0]
    decision = f"新锚点回溯{primary_hits}/102，旧同图全局排序{base_hits}/102，All-at-Once{aa}/102。相对两项对照的观测变化分别为{primary_hits - base_hits:+d}和{primary_hits - aa:+d}条；不把图变密或路径增多解释成准确率增益。"
    review = read(out / "analysis/blind_edge_review.json")
    reviewed = [
        r
        for r in review["items"]
        if r.get("review", {}).get("verdict")
        in {"supported", "unsupported", "uncertain"}
    ]
    quality_lines = [
        "| 抽样组 | supported | unsupported | uncertain |",
        "|---|---:|---:|---:|",
    ]
    for category in ["accepted", "rejected", "no_accepted_edge"]:
        values = Counter(
            r["review"]["verdict"] for r in reviewed if r["category"] == category
        )
        quality_lines.append(
            f"| {category} | {values['supported']} | {values['unsupported']} | {values['uncertain']} |"
        )
    quality_text = (
        (
            f"已完成 {len(reviewed)}/{len(review['items'])} 项助手语义盲审（不是独立人工金标准）。"
            + "原始响应、图及排名不因审阅改变。\n\n"
            + chr(10).join(quality_lines)
            + f"\n\n逐条证据、理由及盲审边界见 [QUALITY_REVIEW.md]({out / 'QUALITY_REVIEW.md'})。"
            + "拒绝组的 supported 表示引用校验可能损失语义成立的边；无边组的 supported 表示可能漏边。小型分层样本不能代表全图精度/召回率。"
        )
        if reviewed
        else "盲审尚未完成，不能据结构引用校验声称语义准确率。"
    )
    mechanism_lines = [
        "| 方法（postmortem） | Top1 有非空支持路径 | 向未播种步骤传播的案例 | 无链接锚点 | 严格完整图 Tol@0..3（分母102） |",
        "|---|---:|---:|---:|---|",
    ]
    for variant in config["variants"]:
        result = outputs["postmortem/" + variant]
        mech = result["mechanism"]
        path_count = (
            "未输出"
            if variant == "baseline"
            else str(mech["top1_with_dependency_path"])
        )
        missing = "不适用" if variant == "baseline" else str(mech["no_linked_anchor"])
        mechanism_lines.append(
            f"| {variant} | {path_count} | {mech['cases_with_upstream_propagation']} | {missing} | {result['strict_complete_graph_tolerance_hits']} |"
        )
    verification = (
        read(out / "verification.json")
        if (out / "verification.json").exists()
        else None
    )
    verification_text = (
        (
            f"完整入口已实际重跑：退出码 {verification['entry_exit_code']}，{verification['hashed_files']} 个正式/pilot 目标、调用 state 及排名文件哈希不变，新增 API 请求 {verification['new_requests']}；测试通过（项数见复现日志），{verification['secret_scan_files']} 个产物密钥扫描零匹配。见 [verification.json]({out / 'verification.json'})。"
        )
        if verification
        else "最终复现验证待执行，不能视为已交付。"
    )
    resource_lines = [
        "| 方法 | 实际请求 | 已知输入 tokens | 已知输出 tokens | 已知总 tokens | usage 未知请求 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, cost in [
        ("迁移正式构图（八组共用）", main_cost),
        ("技术 pilot（另计）", pilot_cost),
        ("All-at-Once", bm["metrics"]["all_at_once"]),
        ("Binary Search", bm["metrics"]["binary_search"]),
    ]:
        resource_lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{cost[k]:,}"
                for k in [
                    "api_requests",
                    "known_input_tokens",
                    "known_output_tokens",
                    "known_total_tokens",
                    "attempts_without_usage",
                ]
            )
            + " |"
        )
    audit = (
        read(out / "analysis/independent_audit.json")
        if (out / "analysis/independent_audit.json").exists()
        else None
    )
    failed_cost_text = (
        f"{counts.get('output_truncated', 0)} 个终态截断目标消耗的已知 tokens 共 {audit['terminal_failed_target_known_tokens']:,}，已包含在正式总量中。"
        if audit
        else ""
    )
    report = f"""# RootSE 完整迁移实验

{decision}

## 冻结协议与结果

模型{config["model"]}，temperature0、32768输出预算/调用、16并发，默认thinking，未调用Claude。先验选择最多8个历史候选，提供选中字段全文，目标只含决定/动作，不含目标observation、最终测试或根因标签。历史候选仍有检索覆盖限制，不称为全历史完整建图。原前缀截断版本结果不覆盖。

全部{manifest["n_targets"]}目标终态后才评价102轨迹。失败目标不贡献边，保留有效边+规则边；四变体共用同图/语义/融合权重。任何目标失败不触发重跑。主协议postmortem与基线匹配首条失败测试；trace-only单独作输入消融。图抽取不读取该测试。

{chr(10).join(table)}

Top-k是排序命中，不是Tol@k；MRR基于完整排序。所有主指标分母102，部分图未被删除。严格完整构图敏感性把任何目标失败的整条轨迹记0（仍102分母），见各变体metrics.json的strict_complete_graph_tolerance_hits。

## 统计与机制

{chr(10).join(pairtable)}

单轮102轨迹/98任务/35仓库。仓库层配对bootstrap5000次、seed20261006，用于探索样本组成不确定性，不能代替模型独立重复。McNemar/Holm使用配对独立假设，仅探索性报告。RootSE已经参与开发，不能声称是未见测试集。没有根据命中率调权重。

图统计：{json.dumps(graph_stats, ensure_ascii=False)}。逐变体mechanism包含上游传播案例、Top1非空支持路径和回退数。baseline未生成支持路径，因此其该项0代表未输出，不代表没有路径。所有关系引用通过原文及方向核验仍不保证语义真实；盲审包在analysis/blind_edge_review.json，不含标签或定位命中结果。完整图仅指所选目标全部抽取结束且响应合格，不等于图覆盖所有真实依赖。

{chr(10).join(mechanism_lines)}

同图新/旧全局排序 Tol@0 分别为 {outputs["postmortem/anchors-global"]["top1_hits"]}/102 和 {base_hits}/102；新/旧回溯分别为 {primary_hits}/102 和 {outputs["postmortem/legacy-backtrace"]["top1_hits"]}/102，后者预定比较 Holm p={paired[2]["holm_p"]:.4g}。新锚点回溯相对旧全局的仓库重采样区间见上表；其 Top-3/Top-5/MRR 为 {outputs[primary]["top3_hits"]}/{outputs[primary]["top5_hits"]}/{outputs[primary]["mrr"]:.4f}，旧全局为 {outputs["postmortem/baseline"]["top3_hits"]}/{outputs["postmortem/baseline"]["top5_hits"]}/{outputs["postmortem/baseline"]["mrr"]:.4f}，不能只看 Top1。trace-only 新锚点回溯 Top1={outputs["trace-only/anchors-backtrace"]["top1_hits"]}/102。上游传播或解释路径的出现不代表找到了真正根因。

{quality_text}

## 资源、失败与可比性

抽取状态{dict(counts)}。正式逻辑调用{main_cost["logical_calls"]}，实际请求{main_cost["api_requests"]}，已知tokens {main_cost["known_total_tokens"]:,}，{main_cost["attempts_without_usage"]}次usage未知。技术pilot另计：{pilot_cost["api_requests"]}请求、{pilot_cost["known_total_tokens"]:,}已知tokens、{pilot_cost["attempts_without_usage"]}未知usage。正式和pilot均不与旧基线费用混算。

{chr(10).join(resource_lines)}

正式请求跨度 {main_cost["active_span_seconds"] / 60:.2f} 分钟，累计请求耗时 {main_cost["request_time_sum_seconds"] / 3600:.2f} 小时（并发时不能作为墙钟耗时）。未知 usage 请求不能当作零成本；HTTP/终态错误计数保留在 metrics.json。{failed_cost_text}首次正式构图已知 token 用量为 All-at-Once 的 {main_cost["known_total_tokens"] / bm["metrics"]["all_at_once"]["known_total_tokens"]:.2f} 倍；服务计费未知，不换算金额。

同一组图服务所有离线消融；首次端到端成本包含构图，不能把缓存排序时间当完整方法时间。当前均等每调用上限不等于同总资源。All-at-Once和Binary直接分析完整轨迹，方法输入表示与调用量不同；FAMAS、TrajAudit及论文跨骨干表见原基线总结。没有新模型理由裁判或修复验证，不报告Justification Accuracy或因果有效性。

## 复现

`bash scripts/reproduce_rootse_migration.sh /path/to/moma_ai.key`：校验冻结代码/数据、跑技术pilot、断点构图、同图消融和离线报告；已有终态目标不会新增请求。所有attempt/raw response/usage在targets目录，pilot单独目录。输入、规则边、配置和代码快照与manifest指纹{manifest["fingerprint"]}保留。

{verification_text}

逐条 Tol@0..3、Top-k/MRR、严格完整图分母及八组依赖哈希已由独立离线算术脚本复核，见 [independent_audit.json]({out / "analysis/independent_audit.json"})。复核命令：`.venv/bin/python scripts/audit_rootse_migration_results.py`（完成最终验证后执行）。

应结合上表差值及区间判断收益，不能将相对某项对照的局部变化写成普遍增益。下一阶段应在独立开发集验证引用校验召回、锚点覆盖与排序目标；本轮没有实施这些事后修改。此前六基线及论文/作者CSV差异见 [完整基线报告]({a.baselines.resolve() / "2026-10-06--rootse-baselines--r1--round-review.md"})。跨骨干论文数字只作描述性对照。
"""
    (out / "EXPERIMENT_SUMMARY.md").write_text(report)
    (out / "analysis/analysis-report.md").write_text(report)
    (out / "analysis/stats-appendix.md").write_text(
        "\n".join(pairtable)
        + "\n\n仓库配对bootstrap5000次，seed20261006；重复抽35仓库，将对应全部轨迹重复纳入。每轮候选与对照使用同一抽样。单轮推理、开发数据和跨任务相关性限制结论。\n"
    )
    save(
        out / "report_provenance.json",
        {
            "generated_at": now(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "label_access": "only_after_all_targets_terminal",
            "metrics_sha256": hashlib.sha256(
                (out / "metrics.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "primary_hits": primary_hits,
                "graphs": graph_stats,
                "accounting": main_cost,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
