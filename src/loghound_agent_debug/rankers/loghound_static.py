"""The registered three-channel LogHound-Agent-Debug ranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from loghound_agent_debug.evidence import EvidenceResult, extract_evidence, load_clue_rules
from loghound_agent_debug.features.semantic import minmax, scores as semantic_scores
from loghound_agent_debug.features.specificity import scores_for
from loghound_agent_debug.graph import build_graph, ppr_from_source_ids
from loghound_agent_debug.schema import TrajectoryInput, input_to_dict


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _failure_text(input_item: TrajectoryInput) -> str:
    return " ".join(observation.content for observation in input_item.failure_observations)


def _evidence_for_step(ordinal: int, evidence: EvidenceResult, graph_edges: list) -> list[str]:
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
    if any(edge.target == ordinal and edge.relation in {"temporal", "same_agent", "message"} for edge in graph_edges) and ordinal not in {hit.step_ordinal for hit in evidence.hits}:
        result.append("upstream_of_failure_evidence")
    return result


def rank_trajectory(input_item: TrajectoryInput, config: dict[str, Any], clue_rules: dict[str, Any], specificity: list[float] | None = None) -> dict[str, Any]:
    evidence = extract_evidence(input_item, clue_rules)
    semantic = semantic_scores(
        input_item,
        _failure_text(input_item),
        task_weight=float(config.get("semantic", {}).get("query_task_weight", 0.4)),
    )
    graph = build_graph(input_item)
    ordinal_by_source = {step.source_step_id: step.ordinal for step in input_item.steps}
    ppr, converged, iterations = ppr_from_source_ids(
        graph,
        ordinal_by_source,
        evidence.symptom_seeds,
        alpha=float(config.get("graph", {}).get("alpha", 0.85)),
        tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
        max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
    )
    if specificity is None:
        specificity = [1.0] * len(input_item.steps)
    path = [upstream * spec for upstream, spec in zip(ppr, specificity)]
    semantic_norm = minmax(semantic)
    path_norm = minmax(path)
    fusion = config.get("fusion", {})
    semantic_weight = float(fusion.get("semantic", 0.12))
    clue_weight = float(fusion.get("clue", 1.0))
    path_weight = float(fusion.get("path", 1.0))
    denominator = float(fusion.get("denominator", semantic_weight + clue_weight + path_weight))
    ranking = []
    for ordinal, step in enumerate(input_item.steps):
        raw = semantic_weight * semantic_norm[ordinal] + clue_weight * evidence.clue_scores[ordinal] + path_weight * path_norm[ordinal]
        ranking.append({
            "rank": 0,
            "source_step_id": step.source_step_id,
            "agent": step.agent,
            "score_raw": round(raw, 9),
            "suspiciousness": round(raw / denominator if denominator else 0.0, 9),
            "components": {
                "semantic_normalized": round(semantic_norm[ordinal], 9),
                "clue": round(evidence.clue_scores[ordinal], 9),
                "path_normalized": round(path_norm[ordinal], 9),
            },
            "evidence": _evidence_for_step(ordinal, evidence, graph.edges),
        })
    ranking.sort(key=lambda item: (-item["score_raw"], int(ordinal_by_source[item["source_step_id"]])))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    config_hash = _hash(config)
    return {
        "trajectory_id": input_item.trajectory_id,
        "evidence_status": evidence.status,
        "evidence_tier": evidence.tier,
        "ranking": ranking,
        "symptom_seeds": evidence.symptom_seeds,
        "graph_stats": {
            "nodes": graph.nodes,
            "edges": len(graph.edges),
            "edge_relations": {relation: sum(edge.relation == relation for edge in graph.edges) for relation in sorted({edge.relation for edge in graph.edges})},
            "ppr_converged": converged,
            "ppr_iterations": iterations,
            "failed_occurrences": evidence.failed_occurrences,
            "failed_clusters": evidence.failed_clusters,
        },
        "config_hash": config_hash,
        "input_hash": _hash(input_to_dict(input_item)),
    }


def rank_file(input_path: str | Path, config_path: str | Path, rules_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rules = load_clue_rules(rules_path)
    results = []
    with open(input_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                from loghound_agent_debug.schema import input_from_dict
                results.append(rank_trajectory(input_from_dict(json.loads(line)), config, rules))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return results
