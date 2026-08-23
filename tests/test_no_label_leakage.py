import inspect

from loghound_agent_debug.rankers.loghound_static import rank_trajectory
from loghound_agent_debug.schema import FailureObservation, Step, TrajectoryInput


def test_ranker_accepts_only_canonical_input_and_does_not_reference_labels():
    signature = inspect.signature(rank_trajectory)
    assert "label" not in signature.parameters
    source = inspect.getsource(rank_trajectory)
    assert "ground_truth" not in source
    assert "mistake_step" not in source
    item = TrajectoryInput("0.1.0", "test", "t", "fix a bug", "coding", "failure", [FailureObservation("1", "trace_failure", "Traceback")], [
        Step(0, "0", ["0"], "Engineer", "assistant", "", "fix", "", "", None, [], []),
        Step(1, "1", ["1"], "Engineer", "assistant", "", "", "", "Traceback", None, [], []),
    ])
    rules = {"version": "test", "rules": [{"name": "terminal", "category": "terminal_failure", "weight": 1.0, "fields": ["observation_text"], "compiled": []}]}
    # A hand-built no-regex configuration is sufficient to establish the API
    # boundary without loading labels.
    result = rank_trajectory(item, {"semantic": {"query_task_weight": 0.4}, "graph": {}, "fusion": {}}, rules)
    assert result["trajectory_id"] == "t"
