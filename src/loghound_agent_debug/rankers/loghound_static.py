"""The registered three-channel LogHound-Agent-Debug ranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from loghound_agent_debug.evidence import EvidenceResult, extract_evidence, load_clue_rules
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
    """Rank every step using semantic, clue, and global PageRank channels."""
    evidence = extract_evidence(input_item, clue_rules)
    semantic = semantic_scores(
        input_item,
        _failure_text(input_item),
        task_weight=float(config.get("semantic", {}).get("query_task_weight", 0.4)),
    )
    graph = build_rootse_graph(len(input_item.steps), rootse_dependencies or []) if rootse_dependencies is not None else build_graph(input_item)
    pagerank_scores, converged, iterations = pagerank(
        graph,
        alpha=float(config.get("graph", {}).get("alpha", 0.85)),
        tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
        max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
    )
    path = pagerank_scores
    semantic_norm = minmax(semantic)
    path_norm = minmax(path)
    fusion = config.get("fusion", {})
    if misstep_scores is not None:
        if len(misstep_scores) != len(input_item.steps):
            raise ValueError("misstep_scores must align with trajectory steps")
        if "fusion_misstep" not in config:
            raise ValueError("misstep_scores provided but fusion_misstep configuration is missing")
        active_fusion = config["fusion_misstep"]
        semantic_weight = float(active_fusion["semantic"])
        clue_weight = float(active_fusion["misstep"])
        path_weight = float(active_fusion["path"])
        denominator = float(active_fusion.get("denominator", semantic_weight + clue_weight + path_weight))
        channel_name = "misstep"
    else:
        active_fusion = fusion
        semantic_weight = float(active_fusion.get("semantic", 0.12))
        clue_weight = float(active_fusion.get("clue", 1.0))
        path_weight = float(active_fusion.get("path", 1.0))
        denominator = float(active_fusion.get("denominator", semantic_weight + clue_weight + path_weight))
        channel_name = "clue"
    ranking = []
    for ordinal, step in enumerate(input_item.steps):
        # Clue scores stay on their fixed rule scale; semantic and path have
        # already been normalized within this trajectory.
        clue_or_misstep = evidence.clue_scores[ordinal] if misstep_scores is None else float(misstep_scores[ordinal])
        raw = semantic_weight * semantic_norm[ordinal] + clue_weight * clue_or_misstep + path_weight * path_norm[ordinal]
        ranking.append({
            "rank": 0,
            "source_step_id": step.source_step_id,
            "agent": step.agent,
            "score_raw": round(raw, 9),
            "suspiciousness": round(raw / denominator if denominator else 0.0, 9),
            "components": {
                "semantic_normalized": round(semantic_norm[ordinal], 9),
                "clue": round(evidence.clue_scores[ordinal], 9),
                **({"misstep": round(float(misstep_scores[ordinal]), 9)} if misstep_scores is not None else {}),
                "pagerank": round(path[ordinal], 9),
                "path_normalized": round(path_norm[ordinal], 9),
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
            "pagerank_sum": round(sum(path), 12),
            "dangling_nodes": pagerank_dangling_nodes(graph),
            "fusion_weights": {"semantic": semantic_weight, channel_name: clue_weight, "structure": path_weight, "denominator": denominator},
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
