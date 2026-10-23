"""Behavioral checks for evidence isolation and failure-conditioned paths."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from loghound_agent_debug.adapters.rootse import load_rootse_file
from loghound_agent_debug.anchors import extract_anchors
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.graph import Edge, Graph, pagerank
from loghound_agent_debug.rankers.anchor_path import rank_trajectory
from loghound_agent_debug.rankers.loghound_static import load_config, rank_trajectory as rank_static
from loghound_agent_debug.rootse_dependencies import _build_request
from loghound_agent_debug.rootse_evaluation import evaluate
from loghound_agent_debug.schema import DiagnosticTest, Label, Step, TrajectoryInput, input_from_dict, input_to_dict

ROOT = Path(__file__).resolve().parents[1]


def rules():
    return load_clue_rules(ROOT / "configs/clue_rules.yaml")


def trajectory(observations, actions=None):
    return TrajectoryInput("0.1.0", "rootse", "case", "fix parser", "coding", "failure", [], [
        Step(i, str(i), [str(i)], "agent", "assistant", "", action_text=(actions or [""] * len(observations))[i], observation_text=text)
        for i, text in enumerate(observations)])


def raw_case(tmp_path):
    path = tmp_path / "data/agent/model/case.json"
    path.parent.mkdir(parents=True)
    raw = {"instance_id": "task", "repo": "a/b", "base_commit": "abc", "original_mission_prompt": "fix parser",
           "agent": "agent", "original_traj": [{"index": "step-A", "thought": "inspect", "response": "", "action": "cat parser.py", "observation": "content"},
                                                  {"index": "step-C", "thought": "edit parser", "action": "", "observation": ""}],
           "failed_tests": {"test_parser": {"code": "def test_parser(): pass", "error_description": "ASSERTION_ONLY_IN_POSTMORTEM", "expected": "NOT_ENABLED"},
                            "test_second": {"code": "SECOND_TEST_NOT_ENABLED", "error_description": "second"}},
           "gold_patch": "GOLD_CANARY", "failure_id": "step-A", "failure_reason": "REASON_CANARY"}
    path.write_text(json.dumps(raw))
    return path, raw


def test_protocol_isolates_test_evidence_and_labels_and_preserves_original_ids(tmp_path):
    path, raw = raw_case(tmp_path)
    trace, _ = load_rootse_file(path)
    post, _ = load_rootse_file(path, protocol="postmortem")
    assert trace.diagnostic_tests == []
    assert trace.failure_anchors == []
    assert "ASSERTION_ONLY_IN_POSTMORTEM" in post.failure_anchors
    assert "second" not in post.failure_anchors
    assert [s.source_step_id for s in trace.steps] == ["step-A", "step-C"]
    assert len(post.diagnostic_tests) == 1
    assert post.steps == trace.steps and post.failure_observations == trace.failure_observations
    for item in (trace, post):
        text = json.dumps(input_to_dict(item))
        assert "GOLD_CANARY" not in text and "REASON_CANARY" not in text
        assert "NOT_ENABLED" not in text and "SECOND_TEST_NOT_ENABLED" not in text
        assert input_from_dict(input_to_dict(item)) == item
    assert "ASSERTION_ONLY_IN_POSTMORTEM" not in json.dumps(input_to_dict(trace))
    request_trace, _ = _build_request(trace.steps, 1, [0])
    request_post, _ = _build_request(post.steps, 1, [0])
    assert request_trace == request_post
    before = rank_trajectory(post, load_config(ROOT / "configs/default.yaml"), rules(), [])
    raw.update(gold_patch="CHANGED_GOLD", failure_id="step-C", failure_reason="CHANGED_REASON")
    path.write_text(json.dumps(raw))
    after_input, _ = load_rootse_file(path, protocol="postmortem")
    assert rank_trajectory(after_input, load_config(ROOT / "configs/default.yaml"), rules(), []) == before


def test_trace_protocol_rejects_external_test_injection():
    with pytest.raises(ValueError, match="trace-only"):
        replace(trajectory([""]), diagnostic_tests=[DiagnosticTest("t", "", "failure")])


def test_hf_and_github_layouts_share_trajectory_identity(tmp_path):
    path, raw = raw_case(tmp_path)
    other = tmp_path / "RootSE/agent/model/case.json"
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(raw))
    hf, _ = load_rootse_file(path)
    github, _ = load_rootse_file(other)
    assert hf == github
    assert hf.trajectory_id == "rootse/agent/model/case"


def test_anchor_quotes_are_exact_and_task_or_plan_errors_are_not_observed():
    item = trajectory(["normal output", "x" * 4000 + "\n--- FAIL: TestParser (0.1s)\nexit code 1"])
    item = replace(item, task="FAILED tests/test_parser.py", steps=[replace(item.steps[0], decision_text="AssertionError: proposed failure"), item.steps[1]])
    evidence = extract_anchors(item, rules())
    assert evidence.anchors
    assert all(a.source_step_id == "1" for a in evidence.anchors)
    for anchor in evidence.anchors:
        original = getattr(item.steps[anchor.step_ordinal], anchor.field)
        assert original[anchor.start:anchor.end] == anchor.quote
    assert any(a.start > 3000 for a in evidence.anchors)


@pytest.mark.parametrize("output", ["0 failed, 24 passed", "exit code 0", "FAILURE is a word in documentation"])
def test_zero_failure_summaries_do_not_create_anchors(output):
    assert extract_anchors(trajectory([output]), rules()).anchors == []


@pytest.mark.parametrize("output", ["--- FAIL: TestThing (0.01s)", "npm ERR! build failed", "bash: syntax error near unexpected token `('", "<returncode>-9</returncode>"])
def test_non_python_failure_formats(output):
    assert extract_anchors(trajectory([output]), rules()).anchors


def test_later_success_is_qualified_and_does_not_erase_failure():
    item = trajectory(["AssertionError: x\nexit code 1", "passed\nexit code 0", "passed\nexit code 0"], ["pytest a.py", "pytest b.py", "pytest a.py"])
    evidence = extract_anchors(item, rules())
    assert evidence.anchors[0].recovery_step_id == "2"
    assert evidence.anchors[0].recovery_status == "later_same_command_success_unverified"
    assert evidence.seeds[0] > 0


def test_external_test_remains_external_with_explicit_weak_entity_links():
    item = replace(trajectory(["read tests/test_parser.py", "done"]), protocol="postmortem",
                   diagnostic_tests=[DiagnosticTest("test_parser", "", "FAILED tests/test_parser.py")])
    evidence = extract_anchors(item, rules())
    external = next(a for a in evidence.anchors if a.provenance != "trace")
    assert external.step_ordinal is None and external.source_step_id is None
    links = [link for link in evidence.links if link.anchor_id == external.anchor_id]
    assert links and all(link.relation == "entity_match" and link.step_ordinal == 0 for link in links)
    assert evidence.local_scores == [0.0, 0.0]
    assert evidence.seeds[1] == 0


def test_unmatched_external_anchor_falls_back_without_attaching_to_last_step():
    item = replace(trajectory(["hello", "done"]), protocol="postmortem", diagnostic_tests=[DiagnosticTest("unknown", "", "unexpected value")])
    result = rank_trajectory(item, {}, rules(), [])
    assert result["anchor_stats"]["unlinked"] == 1
    assert result["anchor_stats"]["fallback"] == "semantic_only_no_linked_anchor"
    assert result["anchor_links"] == []
    assert all(row["components"]["pagerank"] == 0 for row in result["ranking"])


def test_personalized_pagerank_only_reaches_dependencies_not_unrelated_hubs():
    graph = Graph(5, [Edge(0, 2, 1, "state_dependency"), Edge(1, 3, 1, "state_dependency"), Edge(1, 4, 1, "state_dependency")])
    scores, converged, _ = pagerank(graph, personalization=[0, 0, 1, 0, 0])
    assert converged and sum(scores) == pytest.approx(1)
    assert scores[0] > 0 and scores[2] > 0
    assert scores[1] == scores[3] == scores[4] == 0


@pytest.mark.parametrize("seeds", [[0, 0], [1], [-1, 2], [float("nan"), 1], [float("inf"), 0]])
def test_invalid_personalization_is_rejected(seeds):
    with pytest.raises(ValueError, match="personalization"):
        pagerank(Graph(2, []), personalization=seeds)


def test_returned_path_has_forward_step_ids_and_real_edge_evidence():
    item = trajectory(["", "", "AssertionError: wrong value"])
    dependencies = [{"source_ordinal": 0, "target_ordinal": 1, "relation": "plan_adoption", "evidence": {"quote": "use the proposed parser"}},
                    {"source_ordinal": 1, "target_ordinal": 2, "relation": "state_dependency", "evidence": {"path": "parser.py"}}]
    result = rank_trajectory(item, {}, rules(), dependencies)
    upstream = next(row for row in result["ranking"] if row["source_step_id"] == "0")
    witness = upstream["supporting_paths"][0]
    assert witness["step_ids"] == ["0", "1", "2"]
    assert witness["edges"][0]["evidence"] == [{"quote": "use the proposed parser"}]


def test_baseline_variant_preserves_static_rankings():
    item = trajectory(["", "Traceback"])
    config = load_config(ROOT / "configs/default.yaml")
    assert rank_trajectory(item, config, rules(), [], "baseline")["ranking"] == rank_static(item, config, rules(), [])["ranking"]


def test_strict_evaluation_rejects_partial_or_invalid_gold_and_separates_tolerance():
    item = trajectory(["", "", ""])
    result = rank_trajectory(item, {}, rules(), [], "baseline")
    labels = [Label("case", ["1"])]
    metrics = evaluate([item], labels, [result])
    assert metrics["top1_hits"] == 0 and metrics["top3_hits"] == 1 and metrics["tolerance_1"] == 1
    with pytest.raises(ValueError, match="complete"):
        evaluate([item], labels, [])
    with pytest.raises(ValueError, match="gold step"):
        evaluate([item], [Label("case", ["bad-ID"])], [result])
    with pytest.raises(ValueError, match="unique"):
        evaluate([item], labels * 2, [result])


def offline_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("offline_experiment", ROOT / "scripts/run_rootse_offline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_runner_never_calls_api_and_preserves_previous_results(tmp_path, monkeypatch):
    raw_case(tmp_path)
    def reject(*args, **kwargs):
        raise AssertionError("offline run attempted network access")
    monkeypatch.setattr("requests.post", reject)
    run = offline_runner()
    args = ["--root", str(tmp_path), "--out-dir", str(tmp_path / "result")]
    assert run.main(args) == 0
    manifest = json.loads((tmp_path / "result/manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["api_requests_this_run"] == 0
    summaries = json.loads((tmp_path / "result/summary.json").read_text())
    assert len(summaries) == 8 and all(r["trajectories"] == 1 for r in summaries)
    original = (tmp_path / "result/summary.json").read_bytes()
    with pytest.raises(SystemExit):
        run.main(args)
    assert (tmp_path / "result/summary.json").read_bytes() == original


def test_missing_cache_marks_run_incomplete_without_partial_metrics(tmp_path, monkeypatch):
    raw_case(tmp_path)
    monkeypatch.setattr("requests.post", lambda *a, **k: pytest.fail("cache-only must never call API"))
    args = ["--root", str(tmp_path), "--out-dir", str(tmp_path / "result"),
            "--graph-source", "cache", "--cache-dir", str(tmp_path / "empty-cache")]
    with pytest.raises(ValueError, match="cache incomplete"):
        offline_runner().main(args)
    manifest = json.loads((tmp_path / "result/manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert not (tmp_path / "result/summary.json").exists()
