"""The RootSE step-level ranker.

Three label-free channels are fused with weighted Reciprocal Rank Fusion:

* ``semantic`` — TF-IDF similarity between a step document and the task plus
  the observed failure text;
* ``path`` — ordinary PageRank over the reversed RootSE dependency graph;
* ``anchor`` — overlap between the step text and symbols named by the failing
  tests.

Everything here runs on one trajectory at a time and never reads an
annotation field.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from loghound_agent_debug.features import action_type as action_channel
from loghound_agent_debug.features import anchor as anchor_scores
from loghound_agent_debug.features.semantic import minmax, scores as semantic_scores
from loghound_agent_debug.graph import build_rootse_graph, pagerank, pagerank_dangling_nodes, pagerank_edge_count
from loghound_agent_debug.schema import TrajectoryInput, input_to_dict


def _hash(value: Any) -> str:
    """Hash canonical JSON so an output records exactly which config it used."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Any) -> dict[str, Any]:
    """Load YAML ranking configuration, treating an empty file as empty config."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _ranks(values: list[float]) -> list[int]:
    """1-based ranks, ties broken by the earlier ordinal.

    The tie rule is deliberate: it encodes a weak "earlier steps are more
    suspicious" prior.  Replacing it with average ranks measurably hurts
    Top-1, so it must not be "fixed" without an explicit position feature.
    """
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


def _step_evidence(ordinal: int, graph_edges: list, path_norm: list[float]) -> list[str]:
    """Explain which structural signals contributed to a step."""
    result = []
    if any(edge.target == ordinal for edge in graph_edges):
        result.append("prerequisite_of_later_step")
    if path_norm[ordinal] >= 0.8:
        result.append("high_graph_importance")
    return result


def rank_trajectory(input_item: TrajectoryInput, config: dict[str, Any], rootse_dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank every step of one RootSE trajectory with weighted RRF."""
    fusion = config.get("fusion") or {}
    semantic = semantic_scores(
        input_item,
        " ".join(observation.content for observation in input_item.failure_observations),
        task_weight=float(config.get("semantic", {}).get("query_task_weight", 0.4)),
        include_observation=bool(config.get("semantic", {}).get("include_observation", False)),
    )
    anchor = anchor_scores.scores(input_item)
    action_settings = config.get("action_type") or {}
    action_kinds = action_channel.kinds(input_item.steps)
    action_prior = action_channel.scores(action_kinds, action_settings)
    graph = build_rootse_graph(len(input_item.steps), rootse_dependencies)
    pagerank_scores, converged, iterations = pagerank(
        graph,
        alpha=float(config.get("graph", {}).get("alpha", 0.85)),
        tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
        max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
    )

    k = max(1, int(fusion.get("k", 60)))
    weights = {
        "semantic": float(fusion.get("semantic", 1.0)),
        "path": float(fusion.get("path", 1.0)),
        "anchor": float(fusion.get("anchor", 1.0)),
        "action": float(fusion.get("action", 0.0)),
    }
    channels = {
        "semantic": (semantic, weights["semantic"]),
        "path": (pagerank_scores, weights["path"]),
        "anchor": (anchor, weights["anchor"]),
    }
    # Compressed trajectories have no meaningful explore/decision split, so the
    # prior is undefined there; leaving it out keeps them on three channels.
    if len(input_item.steps) >= int(action_settings.get("min_steps", 0)):
        channels["action"] = (action_prior, weights["action"])
    raw_scores, active = _rrf_fuse(channels, k)
    # The best reachable score puts every active channel at rank 1, so the
    # fused score is reported on 0-1 scale without pretending to be calibrated.
    max_possible = sum(active.values()) / (k + 1) if active else 0.0
    score_scale = max_possible or 1.0
    fusion_weights = {"method": "rrf", "k": k, **weights, "active_channels": sorted(active), "denominator": score_scale}

    semantic_rank = _ranks(semantic)
    path_rank = _ranks(pagerank_scores)
    anchor_rank = _ranks(anchor)
    semantic_norm = minmax(semantic)
    path_norm = minmax(pagerank_scores)
    ranking = []
    for ordinal, step in enumerate(input_item.steps):
        ranking.append({
            "rank": 0,
            "source_step_id": step.source_step_id,
            "agent": step.agent,
            "score_raw": round(raw_scores[ordinal], 9),
            "suspiciousness": round(raw_scores[ordinal] / score_scale if score_scale else 0.0, 9),
            "components": {
                "semantic_normalized": round(semantic_norm[ordinal], 9),
                "semantic_rank": semantic_rank[ordinal],
                "pagerank": round(pagerank_scores[ordinal], 9),
                "path_normalized": round(path_norm[ordinal], 9),
                "path_rank": path_rank[ordinal],
                "anchor": round(anchor[ordinal], 9),
                "anchor_rank": anchor_rank[ordinal],
                "action_type": action_kinds[ordinal],
                "action_prior": round(action_prior[ordinal], 9),
            },
            "evidence": _step_evidence(ordinal, graph.edges, path_norm),
        })
    # Earliest ordinal wins exact score ties, making repeated runs stable.
    ordinal_by_source = {step.source_step_id: step.ordinal for step in input_item.steps}
    ranking.sort(key=lambda item: (-item["score_raw"], int(ordinal_by_source[item["source_step_id"]])))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    return {
        "trajectory_id": input_item.trajectory_id,
        "ranking": ranking,
        "graph_stats": {
            "nodes": graph.nodes,
            "edges": len(graph.edges),
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
        },
        "config_hash": _hash(config),
        "input_hash": _hash(input_to_dict(input_item)),
    }
