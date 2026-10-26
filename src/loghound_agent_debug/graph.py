"""RootSE dependency graph and non-personalized PageRank."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    """A directed influence relation from one ordinal step to another."""

    source: int
    target: int
    weight: float
    relation: str
    evidence: list[dict] | None = None


@dataclass(frozen=True)
class Graph:
    """Compact trajectory graph represented by its node count and edges."""

    nodes: int
    edges: list[Edge]


def build_rootse_graph(nodes: int, dependencies: list) -> Graph:
    """Merge RootSE evidence by step pair and assign one fixed edge weight."""
    merged: dict[tuple[int, int], dict] = {}
    for dependency in dependencies:
        source = int(dependency["source_ordinal"] if isinstance(dependency, dict) else dependency.source_ordinal)
        target = int(dependency["target_ordinal"] if isinstance(dependency, dict) else dependency.target_ordinal)
        if source < 0 or target < 0 or source >= nodes or target >= nodes or source >= target:
            continue
        key = (source, target)
        relation = dependency["relation"] if isinstance(dependency, dict) else dependency.relation
        evidence = dependency.get("evidence", dependency) if isinstance(dependency, dict) else dependency.evidence
        item = merged.setdefault(key, {"relations": set(), "evidence": []})
        item["relations"].add(str(relation))
        if evidence not in item["evidence"]:
            item["evidence"].append(evidence)
    edges = [Edge(a, b, 0.1, "+".join(sorted(value["relations"])), value["evidence"]) for (a, b), value in sorted(merged.items())]
    return Graph(nodes, edges)


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
