"""Tests for normalization and channel independence in the RootSE ranker."""

from loghound_agent_debug.features.semantic import minmax
from loghound_agent_debug.rankers.loghound_static import rank_trajectory
from loghound_agent_debug.schema import FailureObservation, Step, TrajectoryInput


def test_minmax_constant_channel_is_zero():
    # A constant channel carries no within-trajectory ordering information.
    assert minmax([2.0, 2.0]) == [0.0, 0.0]


def test_ranker_runs_pagerank_without_failure_evidence():
    # No observed failure text must not stop the structural channel, and no
    # channel may be invented from the trajectory-level `outcome` field.
    item = TrajectoryInput("0.1.0", "rootse", "t", "write code", "coding", "failure", [], [
        Step(0, "0", ["0"], "A", "assistant", "", "write", "", "", None, [], []),
    ])
    result = rank_trajectory(item, {"semantic": {}, "graph": {}, "fusion": {}}, [])
    assert result["graph_stats"]["pagerank_iterations"] > 0
    assert set(result["ranking"][0]["components"]) == {
        "semantic_normalized", "semantic_rank", "pagerank", "path_normalized",
        "path_rank", "anchor", "anchor_rank",
    }


def test_pagerank_is_independent_of_failure_observations():
    steps = [
        Step(0, "0", ["0"], "A", "assistant", "", "write", "", "", None, [], ["x.py"]),
        Step(1, "1", ["1"], "A", "assistant", "", "run", "", "", None, ["x.py"], []),
    ]
    clean = TrajectoryInput("0.1.0", "rootse", "t", "write code", "coding", "failure", [], steps)
    noisy = TrajectoryInput(
        "0.1.0", "rootse", "t", "write code", "coding", "failure",
        [FailureObservation("1", "trace_failure", "Traceback: boom")], steps,
    )
    config = {"semantic": {}, "graph": {}, "fusion": {}}
    deps = [{"source_ordinal": 0, "target_ordinal": 1, "relation": "artifact", "evidence": []}]
    clean_path = {row["source_step_id"]: row["components"]["pagerank"] for row in rank_trajectory(clean, config, deps)["ranking"]}
    noisy_path = {row["source_step_id"]: row["components"]["pagerank"] for row in rank_trajectory(noisy, config, deps)["ranking"]}
    assert clean_path == noisy_path
