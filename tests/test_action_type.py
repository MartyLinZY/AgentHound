"""Tests for the action-type prior channel."""

from loghound_agent_debug.features.action_type import classify, kinds, scores
from loghound_agent_debug.rankers.loghound_static import rank_trajectory
from loghound_agent_debug.schema import Step, TrajectoryInput

SETTINGS = {"decision": ["edit", "reasoning"], "min_steps": 0}


def _step(ordinal=0, raw=None, action="", agent="swe-agent"):
    fields = {"raw_action": raw} if raw is not None else {}
    return Step(ordinal, str(ordinal), [str(ordinal)], agent, "assistant", "",
                action_text=action, rootse_fields=fields)


def test_openhands_subcommands_are_split():
    assert classify({"tool": "str_replace_editor", "input": {"command": "view"}}) == "explore"
    assert classify({"tool": "str_replace_editor", "input": {"command": "str_replace"}}) == "edit"
    assert classify({"tool": "str_replace_editor", "input": {"command": "create"}}) == "edit"
    assert classify({"tool": "execute_bash", "input": {"command": "ls"}}) == "shell"
    assert classify({"tool": "finish"}) == "other"


def test_empty_action_is_reasoning_not_missing_data():
    # OpenHands encodes a pure-reasoning step as an empty action.
    assert classify("") == "reasoning"
    assert classify(None) == "reasoning"


def test_acr_function_names():
    assert classify([{"func_name": "write_patch"}]) == "edit"
    assert classify([{"func_name": "search_class"}, {"func_name": "search_method"}]) == "explore"
    assert classify([{"func_name": "run_test"}]) == "shell"


def test_swe_string_commands():
    assert classify("str_replace_editor view /app") == "explore"
    assert classify("str_replace_editor str_replace 1:2 /app/a.py") == "edit"
    assert classify("python -m pytest tests/") == "shell"
    assert classify("grep -r 'x' /app") == "explore"
    assert classify("apply_patch <<'EOF'") == "edit"


def test_write_redirect_is_edit_not_explore():
    # `cat > fix.py << EOF` and `grep ... > patch.py` are writes even though
    # they start with a read-looking command.
    assert classify("cat > fix.py << 'EOF'") == "edit"
    assert classify("grep -n foo /app/a.py > patch.py") == "edit"


def test_scores_are_two_level():
    assert scores(["edit", "reasoning", "shell", "explore", "other"], SETTINGS) == [1.0, 1.0, 0.0, 0.0, 0.0]
    # The decision set is the only knob; unknown types fall to the evidence side.
    assert scores(["edit", "other"], {"decision": ["edit"]}) == [1.0, 0.0]


def test_kinds_prefers_raw_action_and_falls_back_to_action_text():
    steps = [
        _step(0, raw={"tool": "str_replace_editor", "input": {"command": "view"}}),
        _step(1, action='[{"func_name": "write_patch"}]'),
    ]
    assert kinds(steps) == ["explore", "edit"]


def test_action_channel_enters_components_and_skips_when_flat():
    steps = [
        Step(0, "0", ["0"], "swe-agent", "assistant", "", "edit a.py", "", "", None, [], ["a.py"],
             rootse_fields={"raw_action": "str_replace_editor str_replace 1:2 /a.py"}),
        Step(1, "1", ["1"], "swe-agent", "assistant", "", "grep -r x /app", "", "", None, ["a.py"], [],
             rootse_fields={"raw_action": "grep -r x /app"}),
    ]
    item = TrajectoryInput("0.1.0", "rootse", "t", "fix", "coding", "failure", [], steps)
    config = {
        "semantic": {"query_task_weight": 0.4},
        "graph": {},
        "action_type": SETTINGS,
        "fusion": {"method": "rrf", "k": 60, "semantic": 1.0, "path": 0.5, "anchor": 0.5, "action": 1.0},
    }
    ranked = rank_trajectory(item, config, [])
    assert ranked["ranking"][0]["components"]["action_type"] == "edit"
    assert ranked["ranking"][0]["components"]["action_prior"] == 1.0
    assert "action" in ranked["graph_stats"]["fusion_weights"]["active_channels"]

    config["fusion"]["action"] = 0.0
    flat = rank_trajectory(item, config, [])
    assert "action" not in flat["graph_stats"]["fusion_weights"]["active_channels"]


def test_compressed_trajectory_skips_the_action_channel():
    steps = [
        Step(0, "0", ["0"], "auto-code-rover", "assistant", "", "search_class RST", "", "", None, [], []),
        Step(1, "1", ["1"], "auto-code-rover", "assistant", "", "write_patch", "", "", None, [], []),
    ]
    item = TrajectoryInput("0.1.0", "rootse", "t", "fix", "coding", "failure", [], steps)
    config = {
        "semantic": {"query_task_weight": 0.4},
        "graph": {},
        "action_type": {"decision": ["edit", "reasoning"], "min_steps": 6},
        "fusion": {"method": "rrf", "k": 60, "semantic": 1.0, "path": 0.5, "anchor": 0.5, "action": 1.0},
    }
    ranked = rank_trajectory(item, config, [])
    assert "action" not in ranked["graph_stats"]["fusion_weights"]["active_channels"]


def test_step_fields_never_join_the_raw_action_object():
    # Dependency extraction joins the four text fields; the raw action object
    # now lives beside them and must be filtered out instead of stringified.
    from loghound_agent_debug.rootse_dependencies import _step_fields

    step = _step(0, raw={"tool": "str_replace_editor", "input": {"command": "view"}})
    fields = _step_fields(step)
    assert "raw_action" not in fields
    assert all(isinstance(value, str) for value in fields.values())
    assert " ".join(fields.values()).strip() == ""

    step = _step(0, raw=[{"func_name": "write_patch"}], action="write_patch")
    step.rootse_fields["thought"] = "inspect"
    fields = _step_fields(step)
    assert fields == {"thought": "inspect"}
