"""Trajectory dependency graph and reverse personalized PageRank."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from loghound_agent_debug.schema import TrajectoryInput


@dataclass(frozen=True)
class Edge:
    """A directed influence relation from one ordinal step to another."""
    source: int
    target: int
    weight: float
    relation: str


@dataclass(frozen=True)
class Graph:
    """Compact trajectory graph represented by its node count and edges."""
    nodes: int
    edges: list[Edge]


def _add(edges: dict[tuple[int, int], Edge], source: int, target: int, weight: float, relation: str) -> None:
    """Add one edge, keeping only the strongest relation for a node pair."""
    if source == target:
        return
    key = (source, target)
    existing = edges.get(key)
    if existing is None or weight > existing.weight:
        edges[key] = Edge(source, target, weight, relation)


def build_graph(input_item: TrajectoryInput) -> Graph:
    """Build conservative temporal, agent, artifact, and message dependencies."""
    edges: dict[tuple[int, int], Edge] = {}
    by_artifact: dict[str, list[int]] = {}
    # Index writers first so later reads/writes can connect to prior producers.
    for step in input_item.steps:
        for artifact in step.artifacts_written:
            by_artifact.setdefault(artifact.lower(), []).append(step.ordinal)
    # Temporal edges ensure even otherwise unlinked steps retain sequence
    # context; stronger semantic relations can replace them for a pair.
    for index in range(len(input_item.steps) - 1):
        _add(edges, index, index + 1, 0.2, "temporal")
    for index, step in enumerate(input_item.steps):
        for previous in range(index - 1, -1, -1):
            if input_item.steps[previous].agent == step.agent:
                _add(edges, previous, index, 0.5, "same_agent")
                break
        for artifact in set(step.artifacts_read + step.artifacts_written):
            for source in by_artifact.get(artifact.lower(), []):
                if source < index:
                    _add(edges, source, index, 1.0, "artifact")
        # Explicit messages are kept conservative: only a named recipient is
        # used, and only a later step by that agent is connected.
        recipients = re.findall(r"\bto\s+([A-Za-z][A-Za-z ]{1,40}?)(?::|\s+please|\s+implement|\s+complete|$)", step.decision_text, re.I)
        for recipient in recipients:
            recipient = re.sub(r"\s+", " ", recipient).strip().lower()
            for later in input_item.steps[index + 1 :]:
                if later.agent.lower() == recipient:
                    _add(edges, index, later.ordinal, 0.8, "message")
                    break
    return Graph(len(input_item.steps), list(edges.values()))


def reverse_pagerank(graph: Graph, seeds: list[dict], alpha: float = 0.85, tolerance: float = 1e-6, max_iterations: int = 1000) -> tuple[list[float], bool, int]:
    """Run reverse PPR when callers already have ordinal seed IDs."""
    if graph.nodes == 0 or not seeds:
        return [0.0] * graph.nodes, False, 0
    seed_vector = [0.0] * graph.nodes
    for seed in seeds:
        ordinal = int(seed["ordinal"])
        if 0 <= ordinal < graph.nodes:
            seed_vector[ordinal] += float(seed["strength"])
    total = sum(seed_vector)
    if total == 0:
        return [0.0] * graph.nodes, False, 0
    # Normalization makes seed strengths comparable across trajectories.
    seed_vector = [value / total for value in seed_vector]
    reverse: list[list[tuple[int, float]]] = [[] for _ in range(graph.nodes)]
    for edge in graph.edges:
        reverse[edge.target].append((edge.source, edge.weight))
    # A dangling node gets a self-loop so probability mass is conserved.
    for node in range(graph.nodes):
        if not reverse[node]:
            reverse[node].append((node, 1.0))
    ranks = seed_vector[:]
    for iteration in range(1, max_iterations + 1):
        next_ranks = [(1.0 - alpha) * value for value in seed_vector]
        for node, outgoing in enumerate(reverse):
            normalizer = sum(weight for _, weight in outgoing)
            for predecessor, weight in outgoing:
                next_ranks[predecessor] += alpha * ranks[node] * weight / normalizer
        if sum(abs(left - right) for left, right in zip(next_ranks, ranks)) < tolerance:
            return next_ranks, True, iteration
        ranks = next_ranks
    return ranks, False, max_iterations


def ppr_from_source_ids(graph: Graph, source_step_ids: dict[str, int], seeds: list[dict], alpha: float = 0.85, tolerance: float = 1e-6, max_iterations: int = 1000) -> tuple[list[float], bool, int]:
    """Map source IDs to ordinals and run PPR on the reversed graph."""
    if graph.nodes == 0 or not seeds:
        return [0.0] * graph.nodes, False, 0
    seed_vector = [0.0] * graph.nodes
    for seed in seeds:
        # Seeds are persisted with source IDs so rankings remain auditable even
        # when source IDs are not contiguous integers.
        ordinal = source_step_ids.get(str(seed["source_step_id"]))
        if ordinal is not None:
            seed_vector[ordinal] += float(seed["strength"])
    total = sum(seed_vector)
    if total == 0:
        return [0.0] * graph.nodes, False, 0
    seed_vector = [value / total for value in seed_vector]
    reverse: list[list[tuple[int, float]]] = [[] for _ in range(graph.nodes)]
    for edge in graph.edges:
        # In the reversed graph target -> source.
        reverse[edge.target].append((edge.source, edge.weight))
    for node in range(graph.nodes):
        if not reverse[node]:
            reverse[node].append((node, 1.0))
    ranks = seed_vector[:]
    for iteration in range(1, max_iterations + 1):
        next_ranks = [(1.0 - alpha) * value for value in seed_vector]
        for node, outgoing in enumerate(reverse):
            normalizer = sum(weight for _, weight in outgoing)
            for predecessor, weight in outgoing:
                next_ranks[predecessor] += alpha * ranks[node] * weight / normalizer
        # L1 distance is the convergence criterion for the probability vector.
        distance = sum(abs(left - right) for left, right in zip(next_ranks, ranks))
        ranks = next_ranks
        if distance < tolerance:
            return ranks, True, iteration
    return ranks, False, max_iterations
