"""Regression tests for the Day 1-3 semantic/anchor/RRF changes."""

from loghound_agent_debug.adapters.rootse import extract_failure_anchors
from loghound_agent_debug.features import anchor as anchor_channel
from loghound_agent_debug.features.semantic import semantic_document
from loghound_agent_debug.rankers.loghound_static import _is_constant, _ranks, _rrf_fuse, rank_trajectory
from loghound_agent_debug.schema import Step, TrajectoryInput, input_from_dict


def _step(ordinal, decision="", action="", observation=""):
    return Step(
        ordinal=ordinal, source_step_id=str(ordinal), source_event_ids=[str(ordinal)],
        agent="A", role="assistant", raw_content="", decision_text=decision,
        action_text=action, observation_text=observation,
    )


def test_semantic_document_excludes_observation():
    step = _step(0, decision="fix parser", action="edit parser.py", observation="x" * 5000)
    document = semantic_document(step, include_observation=False)
    assert "fix parser" in document
    assert "x" * 100 not in document


def test_semantic_document_can_include_observation_for_rollback():
    step = _step(0, decision="fix parser", action="edit parser.py", observation="unique_marker")
    assert "unique_marker" in semantic_document(step, include_observation=True)


def test_anchor_scores_use_decision_and_action_only():
    item = TrajectoryInput(
        "0.1.0", "rootse", "t", "task", "coding", "failure", [],
        [
            _step(0, decision="inspect TemporaryUploadedFile", action="search_class"),
            _step(1, decision="", action="write_patch", observation="TemporaryUploadedFile"),
        ],
        failure_anchors=["TemporaryUploadedFile", "FileSystemStorage"],
    )
    scores = anchor_channel.scores(item)
    assert scores[0] > 0
    assert scores[1] == 0  # observation must not contribute


def test_rrf_skips_constant_channel():
    fused, active = _rrf_fuse(
        {"a": ([1.0, 2.0, 3.0], 1.0), "b": ([0.0, 0.0, 0.0], 1.0)}, k=60
    )
    assert active == {"a": 1.0}
    assert _ranks(fused) == [3, 2, 1]


def test_rrf_prefers_channel_agreement():
    # step 1 is top-2 in both channels; steps 0 and 2 are top-1 in only one.
    fused, _ = _rrf_fuse(
        {"a": ([0.9, 0.8, 0.1], 1.0), "b": ([0.1, 0.9, 0.8], 1.0)}, k=60
    )
    assert fused[1] == max(fused)


def test_is_constant_handles_empty():
    assert _is_constant([])
    assert _is_constant([2.0, 2.0])
    assert not _is_constant([1.0, 2.0])


def test_extract_failure_anchors_reads_error_description_only():
    raw = {
        "failed_tests": {
            "pkg/test_mod.py::test_case": {
                "error_description": 'FAILED pkg/test_mod.py::test_case - assert None == Unit("nm")',
                "expected": "LeakedFromExpected",
            }
        },
        "failure_id": "3",
        "failure_reason": "LeakedFromReason should not be an anchor",
        "gold_patch": "LeakedFromPatch()",
    }
    anchors = extract_failure_anchors(raw)
    assert "test_case" in anchors
    assert "Unit" in anchors
    for leaked in ("LeakedFromExpected", "LeakedFromReason", "LeakedFromPatch", "FAILED", "assert", "None"):
        assert leaked not in anchors


def test_extract_failure_anchors_without_failed_tests_is_empty():
    assert extract_failure_anchors({"failure_reason": "no tests"}) == []


def test_input_roundtrip_preserves_failure_anchors():
    item = TrajectoryInput(
        "0.1.0", "rootse", "t", "task", "coding", "failure", [], [_step(0)], failure_anchors=["Alpha", "Beta"]
    )
    from loghound_agent_debug.schema import input_to_dict

    assert input_from_dict(input_to_dict(item)).failure_anchors == ["Alpha", "Beta"]


def test_rrf_channel_records_ranks_and_skips_missing_anchor_channel():
    # No anchors and a flat semantic channel leave only path and clue ordering;
    # RRF must not invent a signal from a constant channel.
    steps = [
        Step(0, "0", ["0"], "A", "assistant", "", "read a.py", "", "", None, ["a.py"], []),
        Step(1, "1", ["1"], "A", "assistant", "", "edit a.py", "", "", None, [], ["a.py"]),
    ]
    item = TrajectoryInput("0.1.0", "rootse", "t", "fix", "coding", "failure", [], steps)
    config = {
        "semantic": {"query_task_weight": 0.4},
        "graph": {},
        "fusion": {"semantic": 0.5, "clue": 0.05, "path": 0.45, "denominator": 1},
        "fusion_rootse": {"method": "rrf", "k": 60, "semantic": 0.4, "path": 1.0, "anchor": 1.0, "clue": 0.0},
    }
    ranked = rank_trajectory(item, config, {"rules": []}, [])
    weights = ranked["graph_stats"]["fusion_weights"]
    assert weights["method"] == "rrf"
    assert "anchor" not in weights["active_channels"]
    assert {"semantic_rank", "path_rank", "anchor", "anchor_rank"} <= set(ranked["ranking"][0]["components"])
