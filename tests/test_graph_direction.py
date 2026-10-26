"""Tests for dependency direction and global PageRank importance."""

from loghound_agent_debug.graph import Graph, Edge, build_rootse_graph, pagerank


def test_pagerank_votes_for_upstream_dependency_without_a_seed():
    # Step 1 depends on the artifact written by step 0. Ordinary PageRank on
    # the reversed dependency graph therefore ranks 0 higher, independent of
    # any failure observation.
    graph = build_rootse_graph(2, [
        {"source_ordinal": 0, "target_ordinal": 1, "relation": "artifact", "evidence": []},
    ])
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
