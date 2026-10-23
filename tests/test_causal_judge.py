"""Protocol, evidence, eligibility and cached transport checks on synthetic data."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loghound_agent_debug.schema import Step, TrajectoryInput, DiagnosticTest
from loghound_agent_debug.rankers.causal_judge import build_messages, parse_and_validate, promote_decision
from run_rootse_qwen_ours import judge_case, require_nothinking, CONFIG
from suite_transport import RecordedClient

CONF=json.loads(CONFIG.read_text())

def item():
    return TrajectoryInput('0.1.0','rootse','fixture','return error when limit reached','coding','failure',[],[
        Step(i,str(i),[str(i)],'agent','assistant','LABEL_CANARY', decision_text='inspect' if i!=12 else 'equal limit is success',
             action_text='read code' if i!=12 else 'expect error = nil', observation_text='x'*4000+'VISIBLE_TAIL') for i in range(14)],
        protocol='postmortem', diagnostic_tests=[DiagnosticTest('test_limit','assert error == ErrLimitReached','actual error is nil')])

def decision():
    return {'step_id':'12','reason':'wrong boundary expectation','earlier_candidate_check':'earlier reads are correct',
            'recovery_check':'not corrected','evidence':[{'step_id':'12','field':'action_text','quote':'expect error = nil'}],
            'failure_evidence':{'test_index':0,'field':'code','quote':'assert error == ErrLimitReached'}}

def ranking():
    return {'variant':'static','ranking':[{'source_step_id':str(i),'rank':i+1,'score_raw':1/(i+1)} for i in range(14)]}

def test_all_steps_eligible_full_evidence_no_raw_annotation():
    payload=build_messages(item(),[],{'baseline':ranking()})
    text=json.dumps(payload)
    assert 'LABEL_CANARY' not in text and 'VISIBLE_TAIL' in text
    assert len(json.loads(payload[1]['content'])['trajectory'])==14
    assert '12' not in json.loads(payload[1]['content'])['retrieved_candidates']['baseline']
    assert parse_and_validate(json.dumps(decision()),item())['step_id']=='12'

@pytest.mark.parametrize('change', ['id','quote','test_quote','selected_quote','recovery','index_bool'])
def test_reject_unsupported_output(change):
    d=decision()
    if change=='id':d['step_id']='100'
    if change=='quote':d['evidence'][0]['quote']='invented'
    if change=='test_quote':d['failure_evidence']['quote']='not present'
    if change=='selected_quote':d['evidence'][0]={'step_id':'0','field':'action_text','quote':'read code'}
    if change=='recovery':d['recovery_check']=''
    if change=='index_bool':d['failure_evidence']['test_index']=False
    with pytest.raises(ValueError):parse_and_validate(json.dumps(d),item())

def test_promote_and_fallback_preserve_existing():
    original=ranking();copy=deepcopy(original)
    promoted=promote_decision(original,decision())
    assert promoted['ranking'][0]['source_step_id']=='12'
    assert len({r['source_step_id'] for r in promoted['ranking']})==14
    assert original==copy
    assert promote_decision(original,None)['decision_fallback']

@pytest.mark.parametrize('value',[True,None,0])
def test_thinking_must_be_explicitly_false(value):
    with pytest.raises(ValueError):require_nothinking({**CONF,'enable_thinking':value})

def test_nothinking_actual_payload_and_cached_replay(tmp_path):
    chunk={'model':CONF['model'],'choices':[{'delta':{'content':json.dumps(decision())},'finish_reason':'stop'}],
           'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}
    response=Mock(status_code=200)
    response.iter_lines.return_value=iter([('data: '+json.dumps(chunk)).encode(),b'data: [DONE]'])
    post=Mock(return_value=response)
    factory=lambda out,cfg,key:RecordedClient(out,cfg,key,post=post)
    a=judge_case(item(),{'dependencies':[]},{'baseline':ranking()},tmp_path,CONF,'secret_fixture',factory)
    assert a['status']=='valid'
    assert post.call_args.kwargs['json']['enable_thinking'] is False
    assert 'thinking' not in post.call_args.kwargs['json']
    assert judge_case(item(),{'dependencies':[]},{'baseline':ranking()},tmp_path,CONF,'secret_fixture',factory)==a
    assert post.call_count==1
    assert all(b'secret_fixture' not in p.read_bytes() for p in tmp_path.rglob('*.json'))

def test_malformed_terminal_is_not_silently_retried(tmp_path):
    client=Mock();client.create.return_value=Mock(choices=[Mock(message=Mock(content='not JSON'))])
    factory=Mock(return_value=client)
    a=judge_case(item(),{'dependencies':[]},{'baseline':ranking()},tmp_path,CONF,'key',factory)
    assert a['status']=='format_error' and a['decision'] is None
    assert judge_case(item(),{'dependencies':[]},{'baseline':ranking()},tmp_path,CONF,'key',factory)==a
    assert client.create.call_count==1
