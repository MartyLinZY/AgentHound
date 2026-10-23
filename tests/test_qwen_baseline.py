"""Qwen launch guard, prompt isolation and fail-closed run accounting."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("qwen_runner", Path(__file__).resolve().parents[1] / "scripts/run_qwen_baseline.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def setup(tmp_path):
    source = tmp_path / "upstream"
    (source / "baseline/who_and_when").mkdir(parents=True)
    (source / "utils.py").write_text("annotation_guidelines = 'earliest decisive error'\n")
    (source / "baseline/who_and_when/all_at_once.py").write_text("system_prompt = 'diagnose'\nuser_prompt = '{problem_statement} {trajectories} {test_code} {error_message} {annotation_guidelines} {ground_truth}'\n")
    root = tmp_path / "dataset"
    raw = {"instance_id": "case", "agent": "agent", "original_mission_prompt": "fix bug",
           "original_traj": [{"index": "A", "action": "inspect", "observation": "failed"}],
           "failed_tests": {"test": {"code": "assert ok", "error_description": "bad value"}},
           "gold_patch": "PRIVATE_GOLD", "failure_reason": "PRIVATE_REASON", "failure_id": "A"}
    path = root / "data/agent/model/case.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw))
    args = ["--root", str(root), "--trajaudit-source", str(source), "--out-dir", str(tmp_path / "out")]
    return args, raw, source


def test_qwen_prompt_does_not_forward_annotations(tmp_path):
    _, raw, source = setup(tmp_path)
    text = json.dumps(runner.messages_for(raw, source))
    assert "PRIVATE_GOLD" not in text and "PRIVATE_REASON" not in text
    assert "failure_id" not in text


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "gpt-5", "qwen-fallback-claude"])
def test_other_models_are_rejected(model):
    with pytest.raises(ValueError):
        runner.validate_model(model)


def test_dry_run_cannot_send_requests(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path)
    monkeypatch.delenv("QWEN_MODEL_NAME", raising=False)
    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: pytest.fail("dry run sent request"))
    assert runner.main(args + ["--dry-run"]) == 0
    result = json.loads((tmp_path / "out/manifest.json").read_text())
    assert result["status"] == "dry_run_complete" and result["api_requests"] == 0
    assert result["model"] == "Qwen3.5-27B"
    assert not (tmp_path / "out/metrics.json").exists()


def test_missing_credentials_does_not_claim_a_run(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        runner.main(args)
    assert not (tmp_path / "out").exists()


def test_qwen_request_and_complete_result_exclude_secrets(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path)
    monkeypatch.setenv("QWEN_API_KEY", "unit-secret")
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"model": "qwen-test", "choices": [{"finish_reason": "stop", "message": {"content": '{"failure_id":"A","failure_reason":"wrong assumption"}'}}], "usage": {"total_tokens": 20}}
    post = Mock(return_value=response)
    monkeypatch.setattr(runner.requests, "post", post)
    assert runner.main(args + ["--model", "qwen-test", "--base-url", "https://example.test/v1"]) == 0
    assert post.call_args.kwargs["json"]["model"] == "qwen-test"
    assert "response_format" not in post.call_args.kwargs["json"]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer unit-secret"
    assert all("unit-secret" not in p.read_text() for p in (tmp_path / "out").glob("*.json"))
    assert json.loads((tmp_path / "out/metrics.json").read_text())["exact_hits"] == 1


def test_truncated_response_fails_without_metrics(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path)
    monkeypatch.setenv("QWEN_API_KEY", "unit-secret")
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": '{"failure_id":"A","failure_reason":"partial"}'}}], "usage": {"total_tokens": 20}}
    monkeypatch.setattr(runner.requests, "post", Mock(return_value=response))
    assert runner.main(args + ["--base-url", "https://example.test/v1"]) == 2
    assert json.loads((tmp_path / "out/manifest.json").read_text())["status"] == "incomplete"
    assert not (tmp_path / "out/metrics.json").exists()


def test_key_file_and_http_error_are_redacted(tmp_path, monkeypatch):
    args, _, _ = setup(tmp_path)
    key = tmp_path / "moma.key"
    key.write_text("file-secret\n")
    post = Mock(return_value=Mock(ok=False, status_code=400, text="error with file-secret"))
    monkeypatch.setattr(runner.requests, "post", post)
    assert runner.main(args + ["--api-key-file", str(key), "--base-url", "https://example.test/v1"]) == 2
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer file-secret"
    assert post.call_args.kwargs["allow_redirects"] is False
    records = json.loads((tmp_path / "out/records.json").read_text())
    assert records[0]["http_status"] == 400
    assert records[0]["provider_error"] == "error with [REDACTED]"
    assert all("file-secret" not in p.read_text() for p in (tmp_path / "out").rglob("*.json"))


def test_reason_quote_recovery_preserves_id_and_marks_result():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"failure_id":"A","failure_reason":"Unit("nm") failed"}'}}]}
    with pytest.raises(json.JSONDecodeError):
        runner.parse_prediction(response, {"A"})
    result = runner.parse_prediction(response, {"A"}, recover_reason_quotes=True)
    assert result == {"failure_id": "A", "failure_reason": 'Unit("nm") failed', "format_recovered": True}
    with pytest.raises(ValueError):
        runner.parse_prediction(response, {"B"}, recover_reason_quotes=True)


def test_ambiguous_malformed_prediction_is_not_recovered():
    response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"failure_id":"A","failure_reason":"bad "quote", "failure_id":"B"}'}}]}
    with pytest.raises(ValueError):
        runner.parse_prediction(response, {"A", "B"}, recover_reason_quotes=True)
