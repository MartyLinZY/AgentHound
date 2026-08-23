from loghound_agent_debug.adapters.agentracer import parse_history_entry


def test_command_and_terminal_output_stay_in_one_decision_step():
    entry = {
        "step": 7,
        "name": "Engineer",
        "role": "assistant",
        "content": "Engineer thinking: fix it\n```json\n[{\"command_name\": \"Terminal.run\", \"args\": {\"command\": \"pytest\"}}]\n```\nTerminal output: [command]: pytest\n[command output] : AssertionError: Lists differ",
    }
    step = parse_history_entry(entry, 0)
    assert step.source_step_id == "7"
    assert step.action_text == 'Terminal.run {"command": "pytest"}'
    assert "AssertionError" in step.observation_text
    assert step.ordinal == 0


def test_solution_filename_is_normalized():
    step = parse_history_entry({"step": "0", "name": "A", "content": "write Foo_attacked_solution.py"}, 0)
    assert "Foo_attacked_solution.py" not in step.decision_text
    assert "<SOLUTION_FILE>" in step.decision_text
