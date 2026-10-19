"""The registered three-channel LogHound-Agent-Debug ranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from loghound_agent_debug.evidence import EvidenceResult, extract_evidence, load_clue_rules
from loghound_agent_debug.features import anchor as anchor_scores
from loghound_agent_debug.features.semantic import minmax, scores as semantic_scores
from loghound_agent_debug.graph import build_graph, build_rootse_graph, pagerank, pagerank_dangling_nodes, pagerank_edge_count
from loghound_agent_debug.schema import TrajectoryInput, input_to_dict


def _hash(value: Any) -> str:
    """Hash canonical JSON so an output records exactly which config it used."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML ranking configuration, treating an empty file as empty config."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _failure_text(input_item: TrajectoryInput) -> str:
    """Flatten structured failure observations into the semantic query text."""
    return " ".join(observation.content for observation in input_item.failure_observations)


def _ranks(values: list[float]) -> list[int]:
    """1-based ranks, ties broken by the earlier ordinal for determinism."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


def _is_constant(values: list[float]) -> bool:
    """A channel with one repeated score carries no ordering information."""
    return not values or max(values) == min(values)


def _rrf_fuse(channels: dict[str, tuple[list[float], float]], k: int) -> tuple[list[float], dict[str, float]]:
    """Reciprocal Rank Fusion over named channels.

    Constant channels are skipped: a channel that gives every step the same
    score carries no ordering information, and RRF would otherwise turn its
    arbitrary tie order into a spurious signal.
    """
    if not channels:
        return [], {}
    size = len(next(iter(channels.values()))[0])
    fused = [0.0] * size
    active: dict[str, float] = {}
    for name, (values, weight) in channels.items():
        if weight <= 0 or _is_constant(values):
            continue
        active[name] = weight
        for index, rank in enumerate(_ranks(values)):
            fused[index] += weight / (k + rank)
    return fused, active


def _evidence_for_step(ordinal: int, evidence: EvidenceResult, graph_edges: list, path_norm: list[float]) -> list[str]:
    """Explain which local or graph-derived signals contributed to a step."""
    result = []
    hits = [hit for hit in evidence.hits if hit.step_ordinal == ordinal]
    if any(hit.category == "terminal_failure" for hit in hits):
        result.append("terminal_failure_observation")
    if any(hit.category == "tool_env_failure" for hit in hits):
        result.append("tool_environment_failure_observation")
    if any(hit.category == "negative_confession" for hit in hits):
        result.append("agent_negative_confession")
    if any(edge.target == ordinal and edge.relation == "artifact" for edge in graph_edges):
        result.append("reads_or_modifies_reused_artifact")
    if path_norm[ordinal] >= 0.8:
        result.append("high_graph_importance")
    return result


def rank_trajectory(input_item: TrajectoryInput, config: dict[str, Any], clue_rules: dict[str, Any], rootse_dependencies: list[dict[str, Any]] | None = None, misstep_scores: list[float] | None = None) -> dict[str, Any]:
    """Rank every step using semantic, anchor, clue, and global PageRank channels.

    ``fusion.method: linear`` (the global default) keeps the historical weighted
    sum.  ``fusion_rootse.method: rrf`` enables reciprocal rank fusion on the
    RootSE path only, so the AgenTracer baseline stays reproducible.
    """
    evidence = extract_evidence(input_item, clue_rules)
    semantic = semantic_scores(
        input_item,
        _failure_text(input_item),
        task_weight=float(config.get("semantic", {}).get("query_task_weight", 0.4)),
        include_observation=bool(config.get("semantic", {}).get("include_observation", False)),
    )
    anchor = anchor_scores.scores(input_item)
    graph = build_rootse_graph(len(input_item.steps), rootse_dependencies or []) if rootse_dependencies is not None else build_graph(input_item)
    pagerank_scores, converged, iterations = pagerank(
        graph,
        alpha=float(config.get("graph", {}).get("alpha", 0.85)),
        tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
        max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
    )
    if misstep_scores is not None and len(misstep_scores) != len(input_item.steps):
        raise ValueError("misstep_scores must align with trajectory steps")

    # fusion_rootse overrides fusion only on the RootSE path, and only when it
    # explicitly asks for RRF.  Every other caller (AgenTracer, rollback to
    # linear) keeps the global fusion section untouched.
    rootse_fusion = config.get("fusion_rootse") or {}
    if rootse_dependencies is not None and str(rootse_fusion.get("method", "")).lower() == "rrf":
        fusion = rootse_fusion
    else:
        fusion = config.get("fusion") or {}
    method = str(fusion.get("method", "linear")).lower()

    semantic_norm = minmax(semantic)
    path_norm = minmax(pagerank_scores)
    if method == "rrf":
        k = max(1, int(fusion.get("k", 60)))
        weights = {
            "semantic": float(fusion.get("semantic", 1.0)),
            "path": float(fusion.get("path", 1.0)),
            "anchor": float(fusion.get("anchor", 1.0)),
            "clue": float(fusion.get("clue", 0.0)),
            "misstep": float(fusion.get("misstep", 1.0 if misstep_scores is not None else 0.0)),
        }
        channels = {
            "semantic": (semantic, weights["semantic"]),
            "path": (pagerank_scores, weights["path"]),
            "anchor": (anchor, weights["anchor"]),
            "clue": (evidence.clue_scores, weights["clue"]),
        }
        if misstep_scores is not None:
            channels["misstep"] = (misstep_scores, weights["misstep"])
        raw_scores, active = _rrf_fuse(channels, k)
        # The best reachable score puts every active channel at rank 1, so the
        # fused score can be reported on a 0-1 scale without pretending to be a
        # calibrated probability.
        max_possible = sum(active.values()) / (k + 1) if active else 0.0
        score_scale = max_possible or 1.0
        fusion_weights = {"method": "rrf", "k": k, **weights, "active_channels": sorted(active), "denominator": score_scale}
    else:
        if misstep_scores is not None:
            if "fusion_misstep" not in config:
                raise ValueError("misstep_scores provided but fusion_misstep configuration is missing")
            active_fusion = config["fusion_misstep"]
            semantic_weight = float(active_fusion["semantic"])
            clue_weight = float(active_fusion["misstep"])
            path_weight = float(active_fusion["path"])
            denominator = float(active_fusion.get("denominator", semantic_weight + clue_weight + path_weight))
            third_channel = [float(x) for x in misstep_scores]
            channel_name = "misstep"
        else:
            active_fusion = fusion
            semantic_weight = float(active_fusion.get("semantic", 0.12))
            clue_weight = float(active_fusion.get("clue", 1.0))
            path_weight = float(active_fusion.get("path", 1.0))
            denominator = float(active_fusion.get("denominator", semantic_weight + clue_weight + path_weight))
            third_channel = evidence.clue_scores
            channel_name = "clue"
        raw_scores = [
            semantic_weight * semantic_norm[i] + clue_weight * third_channel[i] + path_weight * path_norm[i]
            for i in range(len(input_item.steps))
        ]
        score_scale = denominator or 1.0
        fusion_weights = {"method": "linear", "semantic": semantic_weight, channel_name: clue_weight, "structure": path_weight, "denominator": denominator}
    semantic_rank = _ranks(semantic)
    path_rank = _ranks(pagerank_scores)
    anchor_rank = _ranks(anchor)
    ranking = []
    for ordinal, step in enumerate(input_item.steps):
        # Clue scores stay on their fixed rule scale; semantic and path have
        # already been normalized within this trajectory.
        ranking.append({
            "rank": 0,
            "source_step_id": step.source_step_id,
            "agent": step.agent,
            "score_raw": round(raw_scores[ordinal], 9),
            "suspiciousness": round(raw_scores[ordinal] / score_scale if score_scale else 0.0, 9),
            "components": {
                "semantic_normalized": round(semantic_norm[ordinal], 9),
                "semantic_rank": semantic_rank[ordinal],
                "clue": round(evidence.clue_scores[ordinal], 9),
                "pagerank": round(pagerank_scores[ordinal], 9),
                "path_normalized": round(path_norm[ordinal], 9),
                "path_rank": path_rank[ordinal],
                "anchor": round(anchor[ordinal], 9),
                "anchor_rank": anchor_rank[ordinal],
                **({"misstep": round(float(misstep_scores[ordinal]), 9)} if misstep_scores is not None else {}),
            },
            "evidence": _evidence_for_step(ordinal, evidence, graph.edges, path_norm),
        })
    # Earliest ordinal wins exact score ties, making repeated runs stable.
    ordinal_by_source = {step.source_step_id: step.ordinal for step in input_item.steps}
    ranking.sort(key=lambda item: (-item["score_raw"], int(ordinal_by_source[item["source_step_id"]])))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    config_hash = _hash(config)
    return {
        "trajectory_id": input_item.trajectory_id,
        "evidence_status": evidence.status,
        "evidence_tier": evidence.tier,
        "ranking": ranking,
        "clue_hits": [
            {
                "source_step_id": input_item.steps[hit.step_ordinal].source_step_id,
                "score": hit.strength,
                "category": hit.category,
                "rule_names": list(hit.rule_names),
            }
            for hit in evidence.hits
        ],
        "graph_stats": {
            "nodes": graph.nodes,
            "edges": len(graph.edges),
            "causal_edges": len(graph.edges),
            "pagerank_edges": pagerank_edge_count(graph),
            "edge_relations": {relation: sum(edge.relation == relation for edge in graph.edges) for relation in sorted({edge.relation for edge in graph.edges})},
            "pagerank_converged": converged,
            "pagerank_iterations": iterations,
            "pagerank_sum": round(sum(pagerank_scores), 12),
            "dangling_nodes": pagerank_dangling_nodes(graph),
            "fusion_weights": fusion_weights,
            "pagerank_alpha": float(config.get("graph", {}).get("alpha", 0.85)),
            "pagerank_tolerance": float(config.get("graph", {}).get("tolerance", 1e-6)),
            "pagerank_max_iterations": int(config.get("graph", {}).get("max_iterations", 1000)),
            "failed_occurrences": evidence.failed_occurrences,
            "failed_clusters": evidence.failed_clusters,
        },
        "config_hash": config_hash,
        "input_hash": _hash(input_to_dict(input_item)),
    }


def rank_file(input_path: str | Path, config_path: str | Path, rules_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    """Rank a JSONL input file and write one JSON result per trajectory."""
    config = load_config(config_path)
    rules = load_clue_rules(rules_path)
    results = []
    with open(input_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                # Import lazily so importing the ranker does not impose the
                # JSON adapter on callers that use rank_trajectory directly.
                from loghound_agent_debug.schema import input_from_dict
                results.append(rank_trajectory(input_from_dict(json.loads(line)), config, rules))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return results
