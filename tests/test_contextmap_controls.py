"""Guard factorial evidence isolation and resumable execution without network calls."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('tiktoken')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_contextmap as base
import run_contextmap_controls as controls
from suite_transport import digest, save
from loghound_agent_debug.context_map import build_views, messages, tokens
from test_context_map import fixture_case


def inputs(method):
    _, safe, mapping = fixture_case()
    view = build_views(safe, mapping, 3000)['attention']
    original = {'case_id': 'test', 'method': 'attention', 'view': view,
                'messages': messages(safe, view, 'rules'), 'step_ids': [s['id'] for s in safe['steps']]}
    prepared = copy.deepcopy(original)
    order = sorted(view['selected_ids'], key=int) if method == controls.METHODS[0] else view['presentation_order']
    content = controls.render(view, mapping, order, method == controls.METHODS[0])
    prepared['method'] = method
    prepared['view'].update(content=content, presentation_order=order, context_tokens=tokens(content))
    prepared['messages'][1]['content'] = prepared['messages'][1]['content'].replace(view['content'], content)
    prepared['messages_sha256'] = digest(prepared['messages'])
    prepared['message_tokens_proxy'] = sum(tokens(m['content']) for m in prepared['messages'])
    return prepared, original, mapping


@pytest.mark.parametrize('method', controls.METHODS)
def test_only_presentation_may_change(method):
    prepared, original, mapping = inputs(method)
    controls.validate(prepared, original, mapping, method)
    tampered = copy.deepcopy(prepared)
    tampered['messages'][0]['content'] += ' new instruction'
    tampered['messages_sha256'] = digest(tampered['messages'])
    with pytest.raises(ValueError):
        controls.validate(tampered, original, mapping, method)
    prepared['view']['fragments'][0]['text'] += ' fabricated evidence'
    with pytest.raises(ValueError):
        controls.validate(prepared, original, mapping, method)


def test_completed_result_is_reused_and_signature_drift_rejected(tmp_path, monkeypatch):
    prepared, _, _ = inputs(controls.METHODS[0])
    save(tmp_path / 'prepared' / prepared['method'] / 'test.json', prepared)
    calls = []
    class Client:
        def __init__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
                '{"ranked_step_ids":["2","1","3","4","5"],"reason":"evidence","evidence_step_ids":["2"]}'))])
    monkeypatch.setattr(base, 'RecordedClient', Client)
    manifest = {'fingerprint': 'frozen', 'config': {'model': 'test'}}
    first = base.run_one(tmp_path, manifest, 'test', prepared['method'], 'unused')
    assert first['status'] == 'valid'
    assert base.run_one(tmp_path, manifest, 'test', prepared['method'], 'unused') == first
    assert len(calls) == 1
    with pytest.raises(ValueError):
        base.run_one(tmp_path, {**manifest, 'fingerprint': 'changed'}, 'test', prepared['method'], 'unused')
    assert len(calls) == 1
