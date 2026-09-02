"""Tests for dependency direction and global PageRank importance."""

from loghound_agent_debug.graph import Graph, Edge, build_graph, pagerank
from loghound_agent_debug.schema import Step, TrajectoryInput


def test_pagerank_votes_for_upstream_dependency_without_a_seed():
    # Step b depends on the artifact written by step a. Ordinary PageRank on
    # the reversed dependency graph therefore ranks a higher, independent of
    # any failure observation.
    steps = [
        Step(0, "a", ["a"], "A", "assistant", "", "write", "", "", None, [], ["x.py"]),
        Step(1, "b", ["b"], "A", "assistant", "", "run", "", "Traceback", None, [], []),
    ]
    graph = build_graph(TrajectoryInput("0.1.0", "x", "t", "task", "coding", "failure", [], steps))
    ranks, converged, _ = pagerank(graph)
    assert converged
    assert ranks[0] > ranks[1]


def test_pagerank_ignores_temporal_edges():
    graph = Graph(3, [Edge(0, 1, 0.2, "temporal"), Edge(1, 2, 0.2, "temporal")])
    ranks, converged, _ = pagerank(graph)
    assert converged
    assert ranks == [1 / 3, 1 / 3, 1 / 3]


def test_pagerank_uses_uniform_dangling_redistribution():
    graph = Graph(2, [Edge(0, 1, 1.0, "artifact")])
    ranks, converged, _ = pagerank(graph)
    assert converged
    assert abs(sum(ranks) - 1.0) < 1e-9
    assert ranks[0] > ranks[1]
