"""Failure-conditioned dependency ranking, separate from the frozen baseline."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from loghound_agent_debug.anchors import ANCHOR_VERSION, AnchorEvidence, extract_anchors
from loghound_agent_debug.features.semantic import minmax
from loghound_agent_debug.graph import Graph, build_rootse_graph, pagerank
from loghound_agent_debug.rankers.loghound_static import _hash, rank_trajectory as rank_static
from loghound_agent_debug.schema import TrajectoryInput

VARIANTS = ("baseline", "anchors-global", "legacy-backtrace", "anchors-backtrace")


def supporting_paths(item: TrajectoryInput, graph: Graph, evidence: AnchorEvidence, candidates: list[int], limit: int = 3) -> dict[int, list[dict[str, Any]]]:
    """One shortest upstream witness per seed/candidate; not a causal proof."""
    incoming: dict[int, list] = {}
    for edge in graph.edges:
        if edge.relation != "temporal":
            incoming.setdefault(edge.target, []).append(edge)
    witnesses: dict[int, list[dict[str, Any]]] = {i: [] for i in candidates}
    wanted = set(candidates)
    for link in evidence.links:
        queue = deque([(link.step_ordinal, [])])
        seen = {link.step_ordinal}
        while queue:
            node, reverse_edges = queue.popleft()
            if node in wanted and len(witnesses[node]) < limit:
                forward = list(reversed(reverse_edges))
                witnesses[node].append({
                    "anchor_id": link.anchor_id,
                    "link": {**asdict(link), "source_step_id": item.steps[link.step_ordinal].source_step_id},
                    "step_ids": [item.steps[node].source_step_id] + [item.steps[e.target].source_step_id for e in forward],
                    "edges": [{"source_step_id": item.steps[e.source].source_step_id,
                               "target_step_id": item.steps[e.target].source_step_id,
                               "relation": e.relation, "evidence": e.evidence or []} for e in forward],
                    "evidence_kind": "retrieval_plus_dependency" if link.relation == "entity_match" else "observed_plus_dependency",
                })
            for edge in incoming.get(node, []):
                if edge.source not in seen:
                    seen.add(edge.source)
                    queue.append((edge.source, reverse_edges + [edge]))
    return witnesses


def rank_trajectory(item: TrajectoryInput, config: dict[str, Any], rules: dict[str, Any], dependencies: list[dict[str, Any]], variant: str = "anchors-backtrace") -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    # These historical variants implement linear fusion. Keep the upstream
    # RootSE RRF override confined to the static ranker instead of mixing formulas.
    if variant != "baseline":
        config = {**config, "fusion_rootse": {}, "fusion": {**config.get("fusion", {}), "method": "linear"}}
    result = rank_static(item, config, rules, dependencies)
    result.update({"variant": variant, "protocol": item.protocol, "algorithm_version": ANCHOR_VERSION})
    result["dependency_hash"] = _hash(dependencies)
    result["rules_hash"] = _hash({k: v for k, v in rules.items() if k != "rules"} | {
        "rules": [{k: v for k, v in rule.items() if k != "compiled"} for rule in rules.get("rules", [])]})
    if variant == "baseline":
        return result
    mode = "legacy" if variant == "legacy-backtrace" else "structured"
    evidence = extract_anchors(item, rules, mode)
    graph = build_rootse_graph(len(item.steps), dependencies)
    by_id = {row["source_step_id"]: row for row in result["ranking"]}
    global_scores = [by_id[step.source_step_id]["components"]["pagerank"] for step in item.steps]
    scores = global_scores
    fallback = None
    conditioned = variant != "anchors-global"
    if conditioned and any(evidence.seeds):
        scores, converged, iterations = pagerank(
            graph, alpha=float(config.get("graph", {}).get("alpha", 0.85)),
            tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
            max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
            personalization=evidence.seeds,
        )
        result["graph_stats"].update(pagerank_converged=converged, pagerank_iterations=iterations, pagerank_sum=sum(scores))
    elif conditioned:
        # No seed means no failure-conditioned claim: retrieve by the same
        # semantic channel, without silently reverting to global importance.
        scores = [0.0] * len(item.steps)
        fallback = "semantic_only_no_linked_anchor"
    normalized = minmax(scores)
    weights = result["graph_stats"]["fusion_weights"]
    for ordinal, step in enumerate(item.steps):
        row = by_id[step.source_step_id]
        clue = evidence.local_scores[ordinal] if mode == "structured" else row["components"]["clue"]
        raw = weights["semantic"] * row["components"]["semantic_normalized"] + weights["clue"] * clue + weights["structure"] * normalized[ordinal]
        row.update(score_raw=round(raw, 9), suspiciousness=round(raw / weights["denominator"] if weights["denominator"] else 0.0, 9))
        row["components"].update(clue=clue, global_pagerank=global_scores[ordinal], pagerank=scores[ordinal], path_normalized=normalized[ordinal])
        row["evidence"] = (["observed_failure_anchor"] if clue else []) + (["failure_conditioned_score"] if conditioned and scores[ordinal] > 0 else [])
    ordinal_by_id = {step.source_step_id: step.ordinal for step in item.steps}
    result["ranking"].sort(key=lambda row: (-row["score_raw"], ordinal_by_id[row["source_step_id"]]))
    for rank, row in enumerate(result["ranking"], 1):
        row["rank"] = rank
    candidates = [ordinal_by_id[row["source_step_id"]] for row in result["ranking"][:5]]
    paths = supporting_paths(item, graph, evidence, candidates)
    for row in result["ranking"][:5]:
        row["supporting_paths"] = paths[ordinal_by_id[row["source_step_id"]]]
        if conditioned and row["components"]["pagerank"] > 0 and not any(path["edges"] for path in row["supporting_paths"]):
            row["evidence"].append("anchor_seed_without_dependency_path")
    linked = {link.anchor_id for link in evidence.links}
    result["anchors"] = [asdict(anchor) for anchor in evidence.anchors]
    result["anchor_links"] = [{**asdict(link), "source_step_id": item.steps[link.step_ordinal].source_step_id} for link in evidence.links]
    result["anchor_stats"] = {
        "total": len(evidence.anchors), "linked": len(linked),
        "unlinked": len(evidence.anchors) - len(linked), "seeded_steps": sum(v > 0 for v in evidence.seeds),
        "external": sum(a.step_ordinal is None for a in evidence.anchors), "fallback": fallback,
        "propagation_reaches_unseeded_steps": sum(score > 0 and evidence.seeds[i] == 0 for i, score in enumerate(scores)) if conditioned else None,
        "supported_dependency_edges": sum(scores[edge.target] > 0 for edge in graph.edges) if conditioned else None,
    }
    result["graph_stats"]["ranking_mode"] = "personalized" if conditioned and not fallback else (fallback or "global")
    result["clue_hits"] = [{"source_step_id": a.source_step_id, "score": a.strength, "category": a.category, "anchor_id": a.anchor_id} for a in evidence.anchors if a.source_step_id is not None]
    result["evidence_status"] = "no_linked_anchor" if not linked else "ok"
    result["evidence_tier"] = "A" if any(a.category != "negative_confession" for a in evidence.anchors) else ("B" if evidence.anchors else "C")
    return result
