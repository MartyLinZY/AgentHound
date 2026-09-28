"""RootSE file operations use only explicit tool and observation evidence."""
import json

from loghound_agent_debug.rootse_operations import parse_operations, track_state_dependencies
from loghound_agent_debug.schema import Step


def step(n, action, observation="", cwd=""):
    action_text = action if isinstance(action, str) else json.dumps(action)
    return Step(n, str(n), [str(n)], "agent", "assistant", "", action_text=action_text,
                observation_text=observation, working_dir=cwd)


def test_openhands_nested_view_and_edit_create_version_edge():
    path = "/workspace/repo/file.py"
    steps = [
        step(0, {"tool": "str_replace_editor", "input": {"command": "str_replace", "path": path}}, f"The file {path} has been edited."),
        step(1, {"tool": "str_replace_editor", "input": {"command": "view", "path": path}}, f"Here's the result of running `cat -n` on {path}: contents"),
    ]
    ops = parse_operations(steps[0])
    assert len(ops) == 1 and ops[0].kind == "modify" and ops[0].status == "applied"
    edges = track_state_dependencies(steps)
    assert [(e.source_ordinal, e.target_ordinal, e.relation) for e in edges] == [(0, 1, "state_dependency")]


def test_partial_edit_chains_versions_but_full_create_replaces():
    path = "/app/a.py"
    steps = [
        step(0, "echo first > /app/a.py", "<returncode>0</returncode>"),
        step(1, "sed -i s/first/second/ /app/a.py", "updated 1 file"),
        step(2, {"tool": "str_replace_editor", "input": {"command": "create", "path": path}}, f"The file {path} has been created."),
        step(3, "cat /app/a.py", "<returncode>0</returncode> result"),
    ]
    edges = track_state_dependencies(steps)
    assert [(e.source_ordinal, e.target_ordinal) for e in edges] == [(0, 1), (2, 3)]


def test_failed_or_ambiguous_action_does_not_preserve_stale_version():
    path = "/app/a.py"
    steps = [
        step(0, "echo first > /app/a.py", "<returncode>0</returncode>"),
        step(1, {"tool": "str_replace_editor", "input": {"command": "str_replace", "path": path}}, "Error: old text not found"),
        step(2, "cat /app/a.py", "<returncode>0</returncode> result"),
    ]
    assert track_state_dependencies(steps) == []


def test_unknown_explicit_patch_invalidates_known_file_state():
    path = "/app/a.py"
    steps = [
        step(0, "echo first > /app/a.py", "<returncode>0</returncode>"),
        step(1, [{"func_name": "write_patch", "arguments": {}, "call_ok": True}], ""),
        step(2, "cat /app/a.py", "<returncode>0</returncode> result"),
    ]
    assert track_state_dependencies(steps) == []
