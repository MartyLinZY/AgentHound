from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.rootse_misstep import (
    MisstepJudge,
    _canonical,
    _messages,
    _citation_check,
    make_chunks,
)
from loghound_agent_debug.rankers.loghound_static import load_config, rank_trajectory
from loghound_agent_debug.evidence import load_clue_rules


def _step(ordinal: int, sid: str, *, thought: str = "", response: str = "", action: str = "", observation: str = ""):
    return SimpleNamespace(ordinal=ordinal, source_step_id=sid, rootse_fields={
        "thought": thought, "response": response, "action": action, "observation": observation,
    })


def _trajectory(steps):
    return SimpleNamespace(trajectory_id="test/trajectory", task="The output must be square.", steps=steps)


def test_chunking_keeps_steps_atomic_and_reports_unfit_step():
    trajectory = _trajectory([_step(0, "0", action="a" * 100), _step(1, "1", action="b" * 100)])
    one_step_chars = len(_canonical(_messages(trajectory.task, "", [{"ordinal": 0, "source_step_id": "0", "fields": {"action": "a" * 100}}])))
    chunks, failures = make_chunks(trajectory, {"max_field_chars": 1000, "max_input_chars": one_step_chars + 20, "max_summary_chars": 10})
    assert not any(x.get("status") == "budget_insufficient" for x in failures)
    assert [x["source_step_id"] for c in chunks for x in c["steps"]] == ["0", "1"]
    chunks, failures = make_chunks(trajectory, {"max_field_chars": 1000, "max_input_chars": one_step_chars - 1, "max_summary_chars": 10})
    assert not chunks
    assert failures[0]["status"] == "budget_insufficient"


def test_decision_time_citation_rejects_own_observation_and_future():
    steps = [_step(0, "0", observation="class A is already present"), _step(1, "1", observation="FAILED"), _step(2, "2", thought="I know it works")]
    visible = [{"ordinal": s.ordinal, "source_step_id": s.source_step_id, "fields": s.rootse_fields} for s in steps]
    target = {"ordinal": 1, "source_step_id": "1"}
    own_observation = {"origin": "step", "field": "observation", "step_id": "1", "quote": "FAILED"}
    future = {"origin": "step", "field": "thought", "step_id": "2", "quote": "I know it works"}
    prior = {"origin": "step", "field": "observation", "step_id": "0", "quote": "class A is already present"}
    assert _citation_check(own_observation, "evidence_contradiction", target, steps, visible, "task", "")[1]
    assert _citation_check(future, "invalid_action", target, steps, visible, "task", "")[1]
    assert _citation_check(prior, "evidence_contradiction", target, steps, visible, "task", "")[0]["char_span"] == [0, 26]


def test_self_assumption_citation_and_task_quote_are_checked():
    steps = [_step(0, "0", thought="The API always squares images.", action="edit()")]
    visible = [{"ordinal": 0, "source_step_id": "0", "fields": steps[0].rootse_fields}]
    target = {"ordinal": 0, "source_step_id": "0"}
    c = {"origin": "step", "field": "thought", "step_id": "0", "quote": "always squares images"}
    accepted, reason = _citation_check(c, "unfounded_assumption", target, steps, visible, "The output must be square.", "")
    assert reason is None and accepted["char_span"] == [8, 29]
    invalid_task = {"origin": "task", "field": "action", "step_id": "0", "quote": "The output must be square."}
    assert _citation_check(invalid_task, "requirement_misreading", target, steps, visible, "The output must be square.", "")[1]


def test_ranker_fusion_branch_and_legacy_formula(tmp_path):
    inputs, _ = load_rootse("data/raw/rootse")
    item = inputs[0]
    config = load_config("configs/default.yaml")
    rules = load_clue_rules("configs/clue_rules.yaml")
    # The legacy linear formula is only reachable without fusion_rootse; that is
    # the rollback path and still requires fusion_misstep when misstep is given.
    old = dict(config)
    old.pop("fusion_misstep")
    old.pop("fusion_rootse", None)
    baseline = rank_trajectory(item, old, rules, [])
    legacy_again = rank_trajectory(item, old, rules, [])
    assert baseline == legacy_again
    with pytest.raises(ValueError, match="fusion_misstep"):
        rank_trajectory(item, old, rules, [], [0.0] * len(item.steps))
    # RootSE now fuses with RRF and records the configured channel weights.
    ranked = rank_trajectory(item, config, rules, [], [0.0] * len(item.steps))
    assert "misstep" in ranked["ranking"][0]["components"]
    assert ranked["graph_stats"]["fusion_weights"]["method"] == "rrf"
    assert ranked["graph_stats"]["fusion_weights"]["misstep"] == 1.0


def test_chunk_cache_hit_reuses_response(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "unit-test-key")
    monkeypatch.setenv("MODEL_NAME", "unit-test-model")
    trajectory = _trajectory([_step(0, "0", thought="Need to edit file", action="edit()")])
    config = {"prompt_version": "test", "format_version": "test", "weights": {}, "max_input_chars": 60000, "max_field_chars": 2000, "max_summary_chars": 500}
    cache = tmp_path / "misstep_cache"
    judge = MisstepJudge(tmp_path, cache, config)
    calls = []

    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgments": [{"source_step_id": "0", "verdict": "no_misstep", "facets": [], "brief_reason": "reasonable"}], "established_state_summary": "No confirmed changes."})}}], "usage": {"total_tokens": 1}}

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("loghound_agent_debug.rootse_misstep.requests.post", post)
    first_record, first_status = judge.judge(trajectory)
    second_record, second_status = judge.judge(trajectory)
    first = {"cache_hit": first_status["cache_hits"] > 0, "requests": first_status["requests"]}
    second = {"cache_hit": second_status["cache_hits"] > 0, "requests": second_status["requests"]}
    assert first == {"cache_hit": False, "requests": 1}
    assert second == {"cache_hit": True, "requests": 0}
    assert len(calls) == 1
    serialized = json.dumps(calls[0]["messages"], ensure_ascii=False)
    for label in ("failure_id", "failure_reason", "gold_patch", "test_patch", "failed_tests"):
        assert label not in serialized


def test_truncated_chunk_is_split_before_rejudging(tmp_path, monkeypatch):
    trajectory = _trajectory([_step(i, str(i), action=f"edit_{i}()") for i in range(4)])
    config = {"weights": {}, "max_input_chars": 60000, "max_field_chars": 2000, "max_summary_chars": 500}
    judge = MisstepJudge(tmp_path, tmp_path / "cache", config, cache_only=True)
    calls = []

    def response(item, chunk, summary):
        calls.append([x["source_step_id"] for x in chunk["steps"]])
        if len(chunk["steps"]) > 1:
            error = RuntimeError("JSONDecodeError: malformed response JSON")
            error.request_attempts = 1
            raise error
        step = chunk["steps"][0]
        return ({"cache_key": f"key-{step['source_step_id']}", "parsed_response": {"judgments": [{"source_step_id": step["source_step_id"], "verdict": "no_misstep", "facets": []}], "established_state_summary": "state"}, "usage": {}}, {"cache_hit": True, "requests": 0})

    monkeypatch.setattr(judge, "_request_chunk", response)
    result, status = judge.judge(trajectory)
    assert status["status"] == "complete"
    assert result["chunk_splits"]
    assert len(result["steps"]) == 4
    assert calls[0] == ["0", "1", "2", "3"]


def test_single_step_truncation_retries_with_lower_reasoning(tmp_path, monkeypatch):
    trajectory = _trajectory([_step(0, "0", thought="Need to edit", action="edit()")])
    judge = MisstepJudge(tmp_path, tmp_path / "cache", {"max_input_chars": 60000, "max_field_chars": 2000, "max_summary_chars": 500, "weights": {}})
    seen = []

    def request(item, chunk, summary, params_override=None):
        seen.append(params_override)
        if params_override is None:
            error = RuntimeError("output_truncated: finish_reason=length")
            error.request_attempts = 1
            raise error
        return ({"cache_key": "low-reasoning", "parsed_response": {"judgments": [{"source_step_id": "0", "verdict": "no_misstep", "facets": []}], "established_state_summary": "Checked."}, "usage": {}}, {"cache_hit": False, "requests": 1})

    monkeypatch.setattr(judge, "_request_chunk", request)
    result, status = judge.judge(trajectory)
    assert result["status"] == "complete"
    assert status["requests"] == 2
    assert seen[0] is None
    assert seen[1]["reasoning_effort"] == "low"
    assert seen[1]["max_tokens"] == 65536

