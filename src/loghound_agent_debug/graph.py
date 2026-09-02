"""Trajectory dependency graph and non-personalized PageRank."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    for step in input_item.steps:
        for artifact in step.artifacts_written:
            by_artifact.setdefault(artifact.lower(), []).append(step.ordinal)

    # Temporal edges remain available for audit and visualization.  The main
    # PageRank routine deliberately excludes them because row normalization
    # would turn a lone low-weight temporal edge into probability one.
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
        recipients = re.findall(
            r"\bto\s+([A-Za-z][A-Za-z ]{1,40}?)(?::|\s+please|\s+implement|\s+complete|$)",
            step.decision_text,
            re.I,
        )
        for recipient in recipients:
            recipient = re.sub(r"\s+", " ", recipient).strip().lower()
            for later in input_item.steps[index + 1 :]:
                if later.agent.lower() == recipient:
                    _add(edges, index, later.ordinal, 0.8, "message")
                    break
    return Graph(len(input_item.steps), list(edges.values()))


def pagerank(
    graph: Graph,
    alpha: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[list[float], bool, int]:
    """Run ordinary weighted PageRank on the reversed dependency graph.

    ``Graph.edges`` point from an influence source to its later consequence.
    PageRank runs on the reverse of every non-temporal edge so a consequence
    votes for the prerequisite it depends on.  The teleport vector is always
    uniform and never depends on evidence, labels, or suspicious steps.
    """
    nodes = graph.nodes
    if nodes == 0:
        return [], False, 0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")

    outgoing: list[list[tuple[int, float]]] = [[] for _ in range(nodes)]
    for edge in graph.edges:
        if edge.relation == "temporal":
            continue
        # Reverse dependency edge: source -> target becomes target -> source.
        outgoing[edge.target].append((edge.source, edge.weight))

    uniform = 1.0 / nodes
    teleport = [uniform] * nodes
    ranks = teleport[:]
    for iteration in range(1, max_iterations + 1):
        next_ranks = [(1.0 - alpha) * value for value in teleport]
        dangling_mass = sum(ranks[node] for node, links in enumerate(outgoing) if not links)
        if dangling_mass:
            contribution = alpha * dangling_mass * uniform
            next_ranks = [value + contribution for value in next_ranks]
        for node, links in enumerate(outgoing):
            if not links:
                continue
            normalizer = sum(weight for _, weight in links)
            for destination, weight in links:
                next_ranks[destination] += alpha * ranks[node] * weight / normalizer
        distance = sum(abs(left - right) for left, right in zip(next_ranks, ranks))
        ranks = next_ranks
        if distance < tolerance:
            return ranks, True, iteration
    return ranks, False, max_iterations


def pagerank_edge_count(graph: Graph) -> int:
    """Return the number of edges used by the main PageRank calculation."""
    return sum(edge.relation != "temporal" for edge in graph.edges)


def pagerank_dangling_nodes(graph: Graph) -> int:
    """Return dangling-node count in the reversed dependency graph."""
    outgoing = [0] * graph.nodes
    for edge in graph.edges:
        if edge.relation != "temporal":
            outgoing[edge.target] += 1
    return sum(value == 0 for value in outgoing)
