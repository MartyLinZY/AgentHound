"""Regression checks for reasoning exhaustion, retry budgets and cache replay."""
import json
from unittest.mock import Mock, patch

import pytest
import requests

from loghound_agent_debug.rootse_dependencies import DependencyExtractor, _parse_model_json
from loghound_agent_debug.schema import Step, TrajectoryInput


def extractor(tmp_path):
    item = DependencyExtractor(tmp_path, tmp_path / "cache", request_params={"thinking": {"type": "disabled"}})
    item.api_key, item.model = "unit-test-key", "test-model"
    return item


def response(finish="stop", content='{"dependencies":[]}'):
    result = Mock()
    result.json.return_value = {"choices": [{"finish_reason": finish, "message": {"content": content}}], "usage": {}}
    return result


def request():
    return {"request_params": {"max_tokens": 4096, "thinking": {"type": "disabled"}}, "model_messages": []}


def test_truncated_even_parseable_json_is_not_retried(tmp_path):
    e = extractor(tmp_path)
    with patch("requests.post", return_value=response("length")) as post:
        with pytest.raises(RuntimeError, match="output_truncated") as caught:
            e._call(request(), 3, "t")
    assert post.call_count == 1
    assert caught.value.request_attempts == 1
    assert e.requests_made["t"] == 1
    again = extractor(tmp_path)
    assert again.requests_made["t"] == 1


def test_repairs_only_unescaped_quotes_inside_json_strings():
    raw = r'{"dependencies":[{"source_evidence":{"quote":"\\">{}".format(path)"}}]}'
    parsed, offsets = _parse_model_json(raw)
    assert parsed["dependencies"][0]["source_evidence"]["quote"] == r'\">{}".format(path)'
    assert len(offsets) >= 1


def test_transient_failure_retries_and_counts_each_http(tmp_path):
    e = extractor(tmp_path)
    with patch("requests.post", side_effect=[requests.Timeout("temporary"), response()]) as post, patch("time.sleep"):
        result = e._call(request(), 3, "t")
    assert result[3] == 2 and post.call_count == 2
    assert e.requests_made["t"] == 2
    assert post.call_args.kwargs["json"]["thinking"] == {"type": "disabled"}


def test_budget_exhaustion_does_not_send(tmp_path):
    e = extractor(tmp_path)
    e.max_requests = 1
    e.requests_made["t"] = 1
    with patch("requests.post") as post, pytest.raises(RuntimeError):
        e._call(request(), 3, "t")
    post.assert_not_called()


def test_complete_empty_result_replays_offline_and_params_change_key(tmp_path):
    steps = [Step(i, str(i), [str(i)], "a", "assistant", "", decision_text="inspect file", rootse_fields={"thought": "inspect file"}) for i in range(2)]
    t = TrajectoryInput("v1", "rootse", "t", "task", "coding", "failure", [], steps)
    e = extractor(tmp_path)
    with patch("requests.post", return_value=response()) as post:
        _, first = e.extract_for_trajectory(t)
    assert first["status"] == "complete" and post.call_count == 1
    e.cache_only = True
    with patch("requests.post") as post:
        _, second = e.extract_for_trajectory(t)
        assert second["status"] == "complete" and second["cache_hits"] == 1
        e.request_params = {"thinking": {"type": "enabled"}}
        _, changed = e.extract_for_trajectory(t)
    post.assert_not_called()
    assert changed["status"] == "incomplete"
