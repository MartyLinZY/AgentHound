"""Evidence provenance regressions found during the PPT/code audit."""

import json

import pytest

from loghound_agent_debug.rootse_operations import (
    parse_operations,
    track_state_dependencies,
)
from loghound_agent_debug.schema import Step


def step(n, action, observation="", cwd=""):
    text = action if isinstance(action, str) else json.dumps(action)
    return Step(
        n,
        str(n),
        [str(n)],
        "agent",
        "assistant",
        "",
        action_text=text,
        observation_text=observation,
        working_dir=cwd,
    )


def pairs(steps):
    return [
        (e.source_ordinal, e.target_ordinal) for e in track_state_dependencies(steps)
    ]


@pytest.mark.parametrize("tool", ["copy_file", "move_file"])
def test_explicit_copy_and_move_use_source_version(tool):
    steps = [
        step(0, "echo old > /app/a.py", "<returncode>0</returncode>"),
        step(
            1,
            {
                "tool": tool,
                "input": {"source": "/app/a.py", "destination": "/app/b.py"},
                "success": True,
            },
        ),
        step(2, "cat /app/b.py", "<returncode>0</returncode>"),
    ]
    assert pairs(steps) == [(0, 1), (1, 2)]


def test_shell_wrapper_uses_explicit_call_working_directory():
    action = {"tool": "execute_bash", "input": {"command": "cat a.py", "cwd": "/app"}}
    ops = parse_operations(step(1, action, "<returncode>0</returncode>"))
    assert [(o.kind, o.path, o.status) for o in ops] == [
        ("read", "/app/a.py", "applied")
    ]


def test_fallback_success_does_not_certify_failed_overwrite():
    steps = [
        step(0, "echo old > /app/a.py", "<returncode>0</returncode>"),
        step(1, "echo new > /app/a.py || true", "<returncode>0</returncode>"),
        step(2, "cat /app/a.py", "<returncode>0</returncode>"),
    ]
    assert pairs(steps) == []


def test_quoted_semicolon_does_not_split_editor_payload():
    action = "str_replace_editor create /app/a.py --file_text 'x = 1; y = 2'"
    ops = parse_operations(step(0, action, "File created successfully at: /app/a.py"))
    assert len(ops) == 1 and ops[0].path == "/app/a.py" and ops[0].mode == "overwrite"


def test_swe_agent_string_editor_reads_current_version():
    steps = [
        step(
            0,
            'str_replace_editor create /app/a.py --file_text "x=1"',
            "File created successfully at: /app/a.py",
        ),
        step(
            1,
            "str_replace_editor view /app/a.py --view_range 1 10",
            "Here's the result of running `cat -n` on /app/a.py:",
        ),
    ]
    assert pairs(steps) == [(0, 1)]


def test_tee_append_keeps_prior_version_dependency():
    steps = [
        step(0, "echo old > /app/a.py", "<returncode>0</returncode>"),
        step(1, "tee -a /app/a.py", "<returncode>0</returncode>"),
    ]
    assert pairs(steps) == [(0, 1)]


def test_unparsed_mutating_script_invalidates_prior_version():
    steps = [
        step(0, "echo old > /app/a.py", "<returncode>0</returncode>"),
        step(
            1,
            "python3 -c \"open('/app/a.py','w').write('new')\"",
            "<returncode>0</returncode>",
        ),
        step(2, "cat /app/a.py", "<returncode>0</returncode>"),
    ]
    assert pairs(steps) == []


def test_multiple_path_aliases_are_deterministic():
    action = {
        "tool": "read_file",
        "input": {"path": "/app/a.py", "file": "/app/b.py"},
        "success": True,
    }
    ops = parse_operations(step(0, action))
    assert [o.path for o in ops] == ["/app/a.py"]


def test_sed_program_is_not_a_second_file():
    ops = parse_operations(
        step(0, "sed -i s/first/second/ /app/a.py", "updated 1 file")
    )
    assert [(o.kind, o.path) for o in ops] == [("modify", "/app/a.py")]


def test_leading_cd_and_single_command_can_confirm_a_write():
    ops = parse_operations(
        step(0, "cd /app && echo new > a.py", "<returncode>0</returncode>")
    )
    assert [(o.path, o.status) for o in ops] == [("/app/a.py", "applied")]


def test_mixed_action_list_does_not_reexecute_previous_command():
    action = [
        {"tool": "execute_bash", "input": {"command": "cat /app/a.py"}},
        "cat /app/b.py",
    ]
    ops = parse_operations(step(0, action, "<returncode>0</returncode>"))
    assert [o.path for o in ops] == ["/app/a.py", "/app/b.py"]


def test_sed_n_and_grep_n_do_not_consume_the_pattern_as_option_value():
    for command in ["sed -n '1,10p' /app/a.py", "grep -n 'thing' /app/a.py"]:
        ops = parse_operations(step(0, command, "<returncode>0</returncode>"))
        assert [(o.kind, o.path) for o in ops] == [("read", "/app/a.py")]


def test_find_exec_is_not_assumed_read_only():
    ops = parse_operations(
        step(0, "find /app -name '*.py' -delete", "<returncode>0</returncode>")
    )
    assert len(ops) == 1 and ops[0].kind == "unknown"


def test_no_dependency_between_operations_within_one_original_step():
    actions = [
        {"tool": "write_file", "input": {"path": "/app/a.py"}, "success": True},
        {"tool": "read_file", "input": {"path": "/app/a.py"}, "success": True},
    ]
    assert pairs([step(0, actions)]) == []
