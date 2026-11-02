"""Ensure raw-call auditing rejects silent request/decision corruption."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip('tiktoken')
pytest.importorskip('matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_contextmap_controls import verify_call
from suite_transport import digest, save


def recorded(tmp_path):
    messages = [{'role': 'user', 'content': 'evidence'}]
    prepared = {'messages': messages, 'messages_sha256': digest(messages), 'step_ids': ['1', '2']}
    cfg = {'model': 'test', 'temperature': 0, 'max_tokens': 32768, 'enable_thinking': True}
    manifest = {'fingerprint': 'fixed', 'config': cfg}
    decision = {'ranked_step_ids': ['1', '2'], 'reason': 'cause', 'evidence_step_ids': ['1']}
    content = json.dumps(decision)
    save(tmp_path / 'state.json', {'signature': digest({'experiment': 'fixed', 'messages': digest(messages)}),
                                 'status': 'valid', 'decision': decision})
    call = tmp_path / 'calls/0001'
    payload = {**cfg, 'messages': messages, 'stream': True, 'stream_options': {'include_usage': True}}
    save(call / 'request.json', payload)
    save(call / 'state.json', {'request_sha256': digest(payload), 'status': 'received', 'response_file': 'response-1.json'})
    save(call / 'response-1.json', {'model': 'test', 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
    chunk = {'choices': [{'index': 0, 'delta': {'content': content}}]}
    (call / 'response-1.sse').write_text('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n')
    return prepared, manifest


def test_valid_raw_response_and_mismatched_decision(tmp_path):
    prepared, manifest = recorded(tmp_path)
    assert verify_call(tmp_path, prepared, manifest)[2] == {'request': 1, 'sse': 1, 'parsed': 1}
    state = json.loads((tmp_path / 'state.json').read_text())
    state['decision']['ranked_step_ids'].reverse()
    save(tmp_path / 'state.json', state)
    with pytest.raises(ValueError, match='Decision differs'):
        verify_call(tmp_path, prepared, manifest)


def test_consistently_rehashed_but_wrong_actual_prompt_rejected(tmp_path):
    prepared, manifest = recorded(tmp_path)
    call = tmp_path / 'calls/0001'
    payload = json.loads((call / 'request.json').read_text())
    payload['messages'][0]['content'] = 'altered evidence'
    save(call / 'request.json', payload)
    state = json.loads((call / 'state.json').read_text())
    state['request_sha256'] = digest(payload)
    save(call / 'state.json', state)
    with pytest.raises(ValueError, match='payload mismatch'):
        verify_call(tmp_path, prepared, manifest)
