"""Behavioral checks for evidence preservation, isolation, and matched views."""
import copy

import numpy as np
import pytest

pytest.importorskip("tiktoken")

from loghound_agent_debug.context_map import build_map, build_views, messages, safe_case, parse_answer


def fixture_case():
    raw = {"original_mission_prompt": "Use a lock context manager", "failure_id": 2,
           "failure_reason": "SECRET_REASON", "gold_patch": "SECRET_PATCH",
           "failed_tests": {"one": {"code": "assert lock.__enter__.called", "error_description": "AssertionError"}},
           "original_traj": [{"index": i, "thought": f"Inspect lock step {i}",
                              "action": "edit /app/lock.py", "observation": "result\n" * (1000 if i == 3 else 10)}
                             for i in range(8)]}
    rng = np.random.default_rng(10)
    vectors = rng.normal(size=(8, 128))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    graph = [{"source_ordinal": 0, "target_ordinal": 7, "relation": "state_dependency"}]
    safe = safe_case(raw)
    mapping = build_map(safe, graph, vectors, vectors, vectors[0], vectors[1])
    return raw, safe, mapping


def test_labels_do_not_change_any_view_or_prompt():
    raw, safe, mapping = fixture_case()
    poison = copy.deepcopy(raw)
    for k in ("failure_id", "failure_reason", "gold_patch"):
        poison[k] = "FORBIDDEN"
    assert safe_case(poison) == safe
    original, changed = build_views(safe, mapping, 3000), build_views(safe_case(poison), mapping, 3000)
    for method in original:
        assert messages(safe, original[method], "rules") == messages(safe_case(poison), changed[method], "rules")
        assert "SECRET_" not in original[method]["content"]


def test_budget_exact_spans_and_same_fact_control():
    _, safe, mapping = fixture_case()
    views = build_views(safe, mapping, 3000)
    for method, view in views.items():
        assert method == "full" or view["context_tokens"] <= 3000
        for fragment in view["fragments"]:
            assert fragment["text"] == safe["steps"][int(fragment["step_id"])][fragment["field"]][fragment["start"]:fragment["end"]]
    assert views["attention"]["fragments"] == views["attention_flat"]["fragments"]
    assert set(views["attention"]["selected_ids"]) == set(views["attention_flat"]["selected_ids"])
    assert views["attention_flat"]["presentation_order"] == sorted(views["attention_flat"]["presentation_order"], key=int)


def test_reader_can_follow_past_dependencies_and_later_effects():
    _, _, mapping = fixture_case()
    a = np.array(mapping["attention"])
    assert np.allclose(a.sum(axis=1), 1)
    assert a[0, 7] > 0 and a[7, 0] > 0
    assert np.all(a >= 0)


def test_parser_never_uses_presentation_positions_or_silent_fallback():
    valid = '{"ranked_step_ids": ["5", "2"], "reason": "cause", "evidence_step_ids": ["2"]}'
    assert parse_answer(valid, ["2", "5"])["ranked_step_ids"] == ["5", "2"]
    with pytest.raises(ValueError):
        parse_answer(valid.replace('"5"', '"0"'), ["2", "5"])
    with pytest.raises(ValueError):
        parse_answer(valid.replace('"5"', '"2"'), ["2", "5"])
