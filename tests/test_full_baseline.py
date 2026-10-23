"""Regression checks for auditability, strict evaluation and safe resumption."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_rootse_full_baseline as full
from report_rootse_baseline import summarize, tolerance


def setup_case(tmp_path):
    for name in ('states', 'requests', 'responses'):
        (tmp_path / name).mkdir()
    payload = {'model': 'Qwen2.5-14B-Instruct-1M', 'messages': [{'role': 'user', 'content': 'diagnose'}]}
    full.save(tmp_path / 'requests/case.json', payload)
    state = {'case_id': 'case', 'status': 'prepared', 'attempts': [], 'step_ids': ['2', '5'],
             'request_file': 'requests/case.json', 'request_sha256': full.sha(full.canonical(payload))}
    path = tmp_path / 'states/case.json'
    full.save(path, state)
    config = {'endpoint': 'https://example.test/v1/chat/completions', 'timeout_seconds': 10, 'transport_attempts': 3}
    return path, config


def response(content='{"failure_id":"2","failure_reason":"bad plan"}', finish='stop'):
    data = {'model': 'default', 'choices': [{'finish_reason': finish, 'message': {'content': content}}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}
    return Mock(status_code=200, json=lambda: data)


def test_completed_resume_sends_no_new_requests(tmp_path):
    path, config = setup_case(tmp_path)
    post = Mock(return_value=response())
    first = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    assert first['status'] == 'valid'
    again = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    assert again == first and post.call_count == 1
    assert not any('secret' in p.read_text() for p in tmp_path.rglob('*.json'))


def test_overflow_is_terminal_and_in_full_denominator(tmp_path):
    path, config = setup_case(tmp_path)
    post = Mock(return_value=Mock(status_code=400, text='maximum context exceeded secret'))
    result = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    assert result['status'] == 'context_overflow' and post.call_count == 1
    rows = [{'status': result['status'], 'tolerance_hits': tolerance(None, '2', False)},
            {'status': 'valid', 'tolerance_hits': tolerance('2', '2', True)}]
    assert summarize(rows)['tolerance_accuracy'] == [0.5] * 4
    assert 'secret' not in path.read_text()


def test_transport_retry_accounts_for_unknown_attempt(tmp_path):
    path, config = setup_case(tmp_path)
    post = Mock(side_effect=[requests.ReadTimeout('private details'), response()])
    result = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post, sleep=lambda _: None)
    assert result['status'] == 'valid' and len(result['attempts']) == 2
    assert result['attempts'][0]['usage_unknown'] is True
    assert 'private details' not in path.read_text()


def test_invalid_json_is_not_repaired_or_retried(tmp_path):
    path, config = setup_case(tmp_path)
    post = Mock(return_value=response('{"failure_id":"2","failure_reason":"Unit("nm")"}'))
    result = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    assert result['status'] == 'format_error' and post.call_count == 1
    assert result['attempts'][0]['usage']['total_tokens'] == 30


def test_request_tamper_refuses_network(tmp_path):
    path, config = setup_case(tmp_path)
    full.save(tmp_path / 'requests/case.json', {'model': 'claude'})
    post = Mock()
    with pytest.raises(ValueError, match='request changed'):
        full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    post.assert_not_called()


def test_saved_response_recovers_without_network(tmp_path):
    path, config = setup_case(tmp_path)
    state = json.loads(path.read_text())
    state['attempts'] = [{'status': 'started', 'number': 1, 'response_file': 'responses/saved.json'}]
    full.save(path, state)
    full.save(tmp_path / 'responses/saved.json', {'metadata': {'usage': {'total_tokens': 30}}, 'data': response().json()})
    post = Mock()
    result = full.run_case(path, tmp_path, config, 'secret', lambda text: text, post=post)
    assert result['status'] == 'valid'
    post.assert_not_called()


def test_truncated_or_non_original_step_never_scores():
    assert full.classify_response(response(finish='length').json(), ['2'], lambda text: text)['status'] == 'output_truncated'
    assert full.classify_response(response().json(), ['5'], lambda text: text)['status'] == 'invalid_step'
    assert tolerance(None, '2', False) == [False] * 4
    assert tolerance('5', '2', True) == [False, False, False, True]
