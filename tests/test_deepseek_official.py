"""Offline checks for SSE completion, cached replay, and interrupted streams."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from deepseek_suite_transport import RecordedClient, ExperimentAbort
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'experiments/rootse_suite/config_deepseek_official_flash.json').read_text())


def response(lines):
    r = Mock(status_code=200)
    r.iter_lines.return_value = iter(lines)
    return r


def chunk(**fields):
    return ('data: ' + json.dumps(fields)).encode()


def test_sse_reasoning_usage_and_replay(tmp_path):
    lines = [chunk(model=CONFIG['model'], choices=[{'index': 0, 'delta': {'reasoning_content': 'private trace'}}]),
             chunk(model=CONFIG['model'], choices=[{'index': 0, 'delta': {'content': 'answer'}, 'finish_reason': 'stop'}]),
             chunk(choices=[], usage={'prompt_tokens': 5, 'completion_tokens': 9, 'total_tokens': 14}), b'data: [DONE]']
    post = Mock(return_value=response(lines))
    client = RecordedClient(tmp_path, CONFIG, 'fake-test-key', post=post)
    args = {'model': CONFIG['model'], 'messages': []}
    result = client.create(**args)
    assert result.choices[0].message.content == 'answer'
    assert result.usage.total_tokens == 14
    saved = json.loads((tmp_path / 'calls/0001/response-1.json').read_text())
    assert saved['choices'][0]['message']['reasoning_content'] == 'private trace'
    payload = post.call_args.kwargs['json']
    assert payload['thinking'] == {'type': 'enabled'}
    assert payload['reasoning_effort'] == 'high'
    assert 'enable_thinking' not in payload
    assert payload['stream_options']['include_usage']
    assert post.call_args.kwargs['stream']
    replay = RecordedClient(tmp_path, CONFIG, 'fake-test-key', post=post).create(**args)
    assert replay.choices[0].message.content == 'answer' and post.call_count == 1


def test_incomplete_stream_is_not_accepted(tmp_path):
    post = Mock(return_value=response([chunk(model=CONFIG['model'], choices=[{'delta': {'content': 'partial'}}])]))
    with pytest.raises(ExperimentAbort):
        RecordedClient(tmp_path, CONFIG, 'fake-test-key', post=post).create(model=CONFIG['model'], messages=[])
    state = json.loads((tmp_path / 'calls/0001/state.json').read_text())
    assert state['status'] == 'response_error'
    assert (tmp_path / 'calls/0001/response-1.sse').exists()


def test_reasoning_only_is_not_a_prediction(tmp_path):
    post = Mock(return_value=response([chunk(model=CONFIG['model'], choices=[{'delta': {'reasoning_content': 'trace'}, 'finish_reason': 'length'}]), b'data: [DONE]']))
    with pytest.raises(ExperimentAbort, match='Unfinished'):
        RecordedClient(tmp_path, CONFIG, 'fake-test-key', post=post).create(model=CONFIG['model'], messages=[])
