"""Tests use local scripted responses only: no network or benchmark labels."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import suite_transport as tr
import run_rootse_suite_worker as worker
from run_qwen_baseline import messages_for

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'TrajAudit'
CONFIG=json.loads((ROOT/'experiments/rootse_suite/config.json').read_text())


def answer(content,finish='stop'):
    return Mock(status_code=200,text=json.dumps({'model':CONFIG['model'],'choices':[{'finish_reason':finish,'message':{'content':content}}],
                                                'usage':{'prompt_tokens':11,'completion_tokens':7,'total_tokens':18}}))


def test_replay_and_transport_retry(tmp_path):
    post=Mock(side_effect=[Mock(status_code=502,text='temporary'),answer('first half')])
    client=tr.RecordedClient(tmp_path,CONFIG,'secret-for-test',post=post,sleep=lambda _:None)
    args={'model':CONFIG['model'],'messages':[{'role':'user','content':'test'}],'max_tokens':1024,'extra_body':{'enable_thinking':False}}
    assert client.create(**args).choices[0].message.content=='first half'
    state=json.loads((tmp_path/'calls/0001/state.json').read_text())
    assert len(state['attempts'])==2 and state['attempts'][0]['usage_unknown']
    request=json.loads((tmp_path/'calls/0001/request.json').read_text())
    assert request['max_tokens']==32768 and 'extra_body' not in request
    replay=tr.RecordedClient(tmp_path,CONFIG,'secret-for-test',post=post)
    assert replay.create(**args).choices[0].message.content=='first half'
    assert post.call_count==2
    assert all(b'secret-for-test' not in p.read_bytes() for p in tmp_path.rglob('*.json'))


def test_unfinished_response_aborts_without_salvage(tmp_path):
    post=Mock(return_value=answer('{"failure_id":"0"}',finish='length'))
    client=tr.RecordedClient(tmp_path,CONFIG,'secret',post=post)
    with pytest.raises(tr.ExperimentAbort,match='Unfinished'):
        client.create(model=CONFIG['model'],messages=[])
    assert post.call_count==1


def test_replay_rejects_changed_input(tmp_path):
    post=Mock(return_value=answer('first half'))
    tr.RecordedClient(tmp_path,CONFIG,'secret',post=post).create(model=CONFIG['model'],messages=[])
    with pytest.raises(tr.ExperimentAbort,match='changed'):
        tr.RecordedClient(tmp_path,CONFIG,'secret',post=post).create(model=CONFIG['model'],messages=[{'role':'user','content':'changed'}])


@pytest.mark.parametrize('method,contents,expected',[
 ('all_at_once',['{"failure_id":"1","failure_reason":"fault"}'],'1'),
 ('step_by_step',['{"current_step_failure":"No","failure_reason":"ok"}','{"current_step_failure":"Yes","failure_reason":"fault"}'],'1'),
 ('binary_search',['second half'],'1'),
 ('TRAIL',['{"errors":[{"step_index":1,"category":"Goal Deviation","description":"fault"}]}','{"failure_id":1,"reason":"fault"}'],'1'),
 ('FAMAS',['{"T0S0":{"action":"view:a","thought":null},"T0S1":{"action":"edit:a","thought":null}}','{"failure_id":"1","failure_reason":"fault"}'],'1'),
 ('agent_locate',['{"stage":"Code Implementation","reason":"fault"}','{"thought":"inspect","action":"view_observation 1"}','{"thought":"done","action":"submit_result 1 fault"}'],'1'),
])
def test_official_algorithm_flow_and_gold_withholding(tmp_path,monkeypatch,method,contents,expected):
    raw={'agent':'synthetic','model':'synthetic','instance_id':'no_reference_fixture',
         'original_mission_prompt':'fix issue','failure_id':'CANARY_GOLD_ID','gold_patch':'CANARY_GOLD_PATCH',
         'failed_tests':{'t':{'code':'assert x','error_description':'x bad','expected':'CANARY_EXPECTATION'}},
         'original_traj':[{'index':0,'action':'cat a','thought':'look','observation':'a'},
                          {'index':1,'action':'edit a','thought':'fix','observation':'changed'}]}
    # Use an isolated source root with the actual reviewed modules/templates.
    source=tmp_path/'source';source.mkdir()
    for name in ['baseline','TrajAudit','utils.py']:(source/name).symlink_to(SOURCE/name)
    (source/'fixture.json').write_text(json.dumps(raw))
    monkeypatch.chdir(source)
    post=Mock(side_effect=[answer(c) for c in contents])
    monkeypatch.setattr(worker,'RecordedClient',lambda out,config,key:tr.RecordedClient(out,config,key,post=post))
    out=tmp_path/'out'
    state=worker.execute_case(source,out,CONFIG,method,'fixture.json','secret')
    assert state['status']=='valid',state
    assert state['failure_id']==expected
    assert post.call_count==len(contents)
    for p in out.glob('calls/*/request.json'):
        text=p.read_text();assert 'CANARY_' not in text
    if method=='all_at_once':
        payload=json.loads((out/'calls/0001/request.json').read_text())
        assert payload['messages']==messages_for(raw,SOURCE)
