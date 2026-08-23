from loghound_agent_debug.graph import build_graph, reverse_pagerank
from loghound_agent_debug.schema import Step, TrajectoryInput


def test_reverse_ppr_moves_mass_from_symptom_to_upstream_step():
    steps = [
        Step(0, "a", ["a"], "A", "assistant", "", "write", "", "", None, [], ["x.py"]),
        Step(1, "b", ["b"], "A", "assistant", "", "run", "", "Traceback", None, [], []),
    ]
    graph = build_graph(TrajectoryInput("0.1.0", "x", "t", "task", "coding", "failure", [], steps))
    ranks, converged, _ = reverse_pagerank(graph, [{"ordinal": 1, "strength": 1.0}])
    assert converged
    assert ranks[0] > ranks[1]
