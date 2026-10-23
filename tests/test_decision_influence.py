from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loghound_agent_debug.schema import Step,TrajectoryInput,DiagnosticTest
from loghound_agent_debug.rankers.decision_influence import messages,parse,parse_partial,propagate,shuffled_weights,rank
from run_rootse_influence import case_call,CONFIG


def fixture():
    return TrajectoryInput('0.1.0','rootse','fixture','return error at limit','coding','failure',[],[
        Step(i,str(i),[str(i)],'agent','assistant','SECRET_LABEL_CANARY',decision_text='use nil at limit',action_text='apply patch',observation_text='done') for i in range(3)],
        protocol='postmortem',diagnostic_tests=[DiagnosticTest('test','FAILURE_TEST_CANARY','expected ErrLimitReached')])


def answer():
    return {'edges':[{'id':'0>1','strength':.75,'source_ref':'s0:decision_text','target_ref':'s1:action_text','reason':'adopts boundary choice'}]}


def test_inputs_isolate_annotations_and_weight_phase_tests():
    a=json.dumps(messages(fixture(),[(0,1)],'weights'));b=json.dumps(messages(fixture(),[(0,1)],'nodes'))
    assert 'SECRET_LABEL_CANARY' not in a+b
    assert 'FAILURE_TEST_CANARY' not in a and 'FAILURE_TEST_CANARY' in b


@pytest.mark.parametrize('mutation',['missing','duplicate','unknown','ref','bool','nan'])
def test_complete_edge_validation(mutation):
    d=answer()
    if mutation=='missing':d['edges']=[]
    if mutation=='duplicate':d['edges']*=2
    if mutation=='unknown':d['edges'][0]['id']='1>2'
    if mutation=='ref':d['edges'][0]['source_ref']='s2:decision_text'
    if mutation=='bool':d['edges'][0]['strength']=True
    if mutation=='nan':d['edges'][0]['strength']=float('nan')
    with pytest.raises(ValueError):parse(json.dumps(d),fixture(),[(0,1)],'weights')


def test_weak_edge_and_chain_attenuation():
    assert propagate(3,[(0,1),(1,2)],[.8,.1],[0,0,1])==pytest.approx([.08,.1,1])
    assert propagate(3,[(0,2),(1,2)],[.8,.8],[0,0,1])==pytest.approx([.5,.5,1])


def test_dag_propagation_matches_independent_linear_solve():
    pairs=[(0,1),(0,3),(1,2),(1,3),(2,3)];w=[.2,.5,.8,.25,.75];f=[.1,0,0,.9]
    matrix=np.zeros((4,4))
    for (i,j),v in zip(pairs,w):matrix[i,j]=v
    matrix/=np.maximum(1,matrix.sum(axis=0))
    assert propagate(4,pairs,w,f)==pytest.approx(np.linalg.solve(np.eye(4)-matrix,f))
    with pytest.raises(ValueError):propagate(2,[(1,0)],[.5],[0,1])


def test_shuffle_preserves_weights_and_is_reproducible():
    w=[0,.25,.5,.75,1]
    assert sorted(shuffled_weights(w,'fixture',123))==w
    assert shuffled_weights(w,'fixture',123)==shuffled_weights(w,'fixture',123)
    assert w==[0,.25,.5,.75,1]


def test_rank_fallback_and_immutability():
    base={'graph_stats':{},'ranking':[{'source_step_id':str(i),'rank':i+1,'components':{}} for i in [2,1,0]]}
    saved=deepcopy(base)
    assert rank(base,fixture(),[1,.5,.1],'test')['ranking'][0]['source_step_id']=='0'
    assert rank(base,fixture(),[0,0,0],'test','zero')['ranking'][0]['source_step_id']=='2'
    assert saved==base


def test_call_reuses_terminal_and_rejects_protocol_drift(tmp_path):
    cfg=json.loads(CONFIG.read_text());client=Mock();client.create.return_value=Mock(choices=[Mock(message=Mock(content=json.dumps(answer())))])
    factory=Mock(return_value=client)
    first=case_call(fixture(),[(0,1)],'weights',tmp_path,cfg,'fixture_key',factory)
    assert first['status']=='valid'
    assert case_call(fixture(),[(0,1)],'weights',tmp_path,cfg,'fixture_key',factory)==first
    assert client.create.call_count==1
    with pytest.raises(ValueError):case_call(fixture(),[(0,1)],'weights',tmp_path,{**cfg,'temperature':.1},'fixture_key',factory)


def test_node_refs_and_complete_ids():
    rows=[{'step_id':str(i),'score':.75,'kind':'introduced','recovered':False,'evidence_ref':f's{i}:decision_text','reference_ref':'task','reason':'boundary contradiction'} for i in range(3)]
    assert len(parse(json.dumps({'nodes':rows}),fixture(),[],'nodes'))==3
    rows[0]['reference_ref']='invented'
    with pytest.raises(ValueError):parse(json.dumps({'nodes':rows}),fixture(),[],'nodes')


def test_partial_rejects_duplicate_and_missing_without_inventing_scores():
    d=answer();d['edges'][0]['strength']=.91
    result=parse_partial(json.dumps(d),fixture(),[(0,1),(1,2)],'weights')
    assert result['rows'][0]['strength']==.91 and result['missing_ids']==['1>2']
    d['edges']*=2
    result=parse_partial(json.dumps(d),fixture(),[(0,1),(1,2)],'weights')
    assert result['rows']==[] and result['missing_ids']==['0>1','1>2']
    assert len(result['rejected_rows'])==2


def test_lossless_numeric_node_id_and_invalid_row_are_separate():
    rows=[{'step_id':0,'score':.9,'kind':'introduced','recovered':False,'evidence_ref':'s0:decision_text','reference_ref':'task','reason':'explicit conflict'},
          {'step_id':'1','score':.8,'kind':'introduced','recovered':False,'evidence_ref':'invented','reference_ref':'task','reason':'conflict'}]
    result=parse_partial(json.dumps({'nodes':rows}),fixture(),[],'nodes')
    assert result['rows'][0]['step_id']=='0'
    assert result['missing_ids']==['1','2'] and len(result['rejected_rows'])==1
