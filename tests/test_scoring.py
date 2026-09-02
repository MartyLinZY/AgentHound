"""Tests for normalization and no-evidence behavior in ranking channels."""

import re

from loghound_agent_debug.features.semantic import minmax
from loghound_agent_debug.schema import Step, TrajectoryInput
from loghound_agent_debug.rankers.loghound_static import rank_trajectory


def test_minmax_constant_channel_is_zero():
    # A constant channel carries no within-trajectory ordering information.
    assert minmax([2.0, 2.0]) == [0.0, 0.0]


def test_no_evidence_does_not_create_clue_and_still_runs_pagerank():
    # Without observed failure text, the ranker must not invent a terminal
    # clue merely because the trajectory outcome is marked as failure.
    item = TrajectoryInput("0.1.0", "x", "t", "write code", "coding", "failure", [], [
        Step(0, "0", ["0"], "A", "assistant", "", "write", "", "", None, [], []),
    ])
    result = rank_trajectory(item, {"semantic": {}, "graph": {}, "fusion": {}}, {"rules": []})
    assert result["evidence_status"] == "no_evidence"
    assert result["clue_hits"] == []
    assert result["graph_stats"]["pagerank_iterations"] > 0
    assert result["ranking"][0]["components"]["clue"] == 0.0


def test_pagerank_component_is_independent_of_clue_hits():
    steps = [
        Step(0, "0", ["0"], "A", "assistant", "", "write", "", "", None, [], ["x.py"]),
        Step(1, "1", ["1"], "A", "assistant", "", "run", "", "Traceback", None, ["x.py"], []),
    ]
    item = TrajectoryInput("0.1.0", "x", "t", "write code", "coding", "failure", [], steps)
    base = {"semantic": {}, "graph": {}, "fusion": {}}
    clean = rank_trajectory(item, base, {"rules": []})
    hit_rules = {"rules": [{"name": "terminal", "category": "terminal_failure", "weight": 1.0, "fields": ["observation_text"], "compiled": [re.compile("Traceback")]}]}
    with_hit = rank_trajectory(item, base, hit_rules)
    clean_path = {row["source_step_id"]: row["components"]["pagerank"] for row in clean["ranking"]}
    hit_path = {row["source_step_id"]: row["components"]["pagerank"] for row in with_hit["ranking"]}
    assert clean_path == hit_path
