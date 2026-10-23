"""New migration protocol tests use mocked transport and synthetic trajectories."""
import json
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import deepseek_migration_core as mc
from deepseek_suite_transport import RecordedClient
from loghound_agent_debug.schema import Step,TrajectoryInput,DiagnosticTest

CONFIG=json.loads((mc.ROOT/'experiments/rootse_migration/config_deepseek_official_flash.json').read_text())
# Nonstream fixtures isolate target validation; streaming is exercised below.
CONFIG={**CONFIG,"stream":False}

def item():
    return TrajectoryInput('0.1.0','rootse','fixture','fix','coding','failure',[],[
        Step(0,'0',['0'],'agent','assistant','',action_text='inspect',observation_text='x'*4000+'source evidence'),
        Step(1,'1',['1'],'agent','assistant','',decision_text='use source evidence',action_text='edit',observation_text='FUTURE_CANARY')])

def answer(content,finish='stop'):
    return Mock(status_code=200,text=json.dumps({'model':CONFIG['model'],'choices':[{'finish_reason':finish,'message':{'content':content}}],
                       'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}))

def factory(post):
    return lambda out,conf,key:RecordedClient(out,conf,key,post=post,sleep=lambda _:None)


def test_full_fields_and_future_evidence_isolation():
    i=item();req,meta=mc.build_target(i,1,CONFIG)
    assert 'source evidence' in req['user_payload']['candidates'][0]['fields']['observation']
    assert 'FUTURE_CANARY' not in json.dumps(req)
    post=replace(i,protocol='postmortem',diagnostic_tests=[DiagnosticTest('t','FINAL_TEST_CANARY','FINAL_ERROR_CANARY')])
    assert mc.build_target(post,1,CONFIG)==(req,meta)
    assert not any(v['truncated'] for f in meta['truncation'].values() for v in f.values())


def test_input_limit_refuses_candidate_drop():
    with pytest.raises(ValueError,match='truncate'):
        mc.build_target(item(),1,{**CONFIG,'max_input_chars':1500})


def test_exact_evidence_retained_and_invalid_quote_rejected(tmp_path):
    relation={'relation':'information_use','source_step_id':'0','source_evidence':{'field':'observation','quote':'source evidence'},
              'target_evidence':{'field':'thought','quote':'use source evidence'},'reason':'uses preceding fact'}
    invalid={**relation,'source_evidence':{'field':'observation','quote':'invented'}}
    post=Mock(return_value=answer(json.dumps({'dependencies':[relation,invalid]})))
    state=mc.run_target(item(),1,tmp_path,CONFIG,'synthetic-key',factory(post))
    assert state['status']=='complete' and len(state['validated_dependencies'])==1 and len(state['rejected_dependencies'])==1
    assert mc.run_target(item(),1,tmp_path,CONFIG,'synthetic-key',factory(post))==state and post.call_count==1
    assert all(b'synthetic-key' not in p.read_bytes() for p in tmp_path.rglob('*.json'))


@pytest.mark.parametrize('response,status',[(answer('{"dependencies":[]}',finish='length'),'output_truncated'),(answer('not json'),'format_error'),(answer(None),'empty_response')])
def test_terminal_failures_not_retried(tmp_path,response,status):
    post=Mock(return_value=response)
    assert mc.run_target(item(),1,tmp_path,CONFIG,'key',factory(post))['status']==status
    assert mc.run_target(item(),1,tmp_path,CONFIG,'key',factory(post))['status']==status
    assert post.call_count==1


def test_interrupted_request_preserves_unknown_attempt(tmp_path):
    post=Mock(return_value=answer('{"dependencies":[]}'))
    # Simulate process death during the first request, then recovery.
    def crash(*args,**kw):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        mc.run_target(item(),1,tmp_path,CONFIG,'key',factory(crash))
    result=mc.run_target(item(),1,tmp_path,CONFIG,'key',factory(post))
    assert result['status']=='complete'
    state=json.loads((tmp_path/'calls/0001/state.json').read_text())
    assert len(state['attempts'])==2 and state['attempts'][0]['usage_unknown']


def test_target_hash_mismatch_refuses_replay(tmp_path):
    post=Mock(return_value=answer('{"dependencies":[]}'))
    mc.run_target(item(),1,tmp_path,CONFIG,'key',factory(post))
    changed=replace(item(),steps=[item().steps[0],replace(item().steps[1],action_text='different')])
    with pytest.raises(ValueError,match='mismatch'):mc.run_target(changed,1,tmp_path,CONFIG,'key',factory(post))


def test_official_stream_and_target_hash_match(tmp_path):
    import run_rootse_deepseek_migration as runner
    config = {**CONFIG, 'stream': True}
    chunk = {'model': config['model'], 'choices': [{'delta': {'content': '{"dependencies":[]}'}, 'finish_reason': 'stop'}], 'usage': {'total_tokens': 20}}
    response = Mock(status_code=200)
    response.iter_lines.return_value = iter([('data: ' + json.dumps(chunk)).encode(), b'data: [DONE]'])
    post = Mock(return_value=response)
    result = mc.run_target(item(), 1, tmp_path, config, 'synthetic-key', factory(post))
    assert result['status'] == 'complete'
    payload = post.call_args.kwargs['json']
    assert post.call_args.args[0] == 'https://api.deepseek.com/chat/completions'
    assert payload['thinking'] == {'type': 'enabled'} and payload['reasoning_effort'] == 'high'
    assert 'enable_thinking' not in payload
    assert runner.run_target is mc.run_target
    req, _ = mc.build_target(item(), 1, config)
    prepared_hash = mc.digest({'model': config['model'], 'messages': req['model_messages'], 'temperature': config['temperature'], 'max_tokens': config['max_tokens']})
    assert result['request_sha256'] == prepared_hash
    assert mc.run_target(item(), 1, tmp_path, config, 'synthetic-key', factory(post)) == result
    assert post.call_count == 1
