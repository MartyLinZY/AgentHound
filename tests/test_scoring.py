"""Tests for normalization and no-evidence behavior in ranking channels."""

from loghound_agent_debug.features.semantic import minmax
from loghound_agent_debug.schema import Step, TrajectoryInput
from loghound_agent_debug.rankers.loghound_static import rank_trajectory


def test_minmax_constant_channel_is_zero():
    # A constant channel carries no within-trajectory ordering information.
    assert minmax([2.0, 2.0]) == [0.0, 0.0]


def test_no_evidence_does_not_create_clue_or_ppr_seed():
    # Without observed failure text, the ranker must not invent a terminal
    # seed merely because the trajectory outcome is marked as failure.
    item = TrajectoryInput("0.1.0", "x", "t", "write code", "coding", "failure", [], [
        Step(0, "0", ["0"], "A", "assistant", "", "write", "", "", None, [], []),
    ])
    result = rank_trajectory(item, {"semantic": {}, "graph": {}, "fusion": {}}, {"rules": []})
    assert result["evidence_status"] == "no_evidence"
    assert result["symptom_seeds"] == []
    assert result["graph_stats"]["ppr_iterations"] == 0
    assert result["ranking"][0]["components"]["clue"] == 0.0
