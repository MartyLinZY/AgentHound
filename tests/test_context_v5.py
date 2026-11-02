from copy import deepcopy

import numpy as np

from loghound_agent_debug.context_map import FIELDS, tokens
from loghound_agent_debug.context_evidence import build_evidence_view, review_messages
from loghound_agent_debug.context_coverage import build_coverage_view, handoff_episodes, coverage_kernel, coverage_weights


def fixture(n=8):
    steps = [{'id': str(i), **{f: '' for f in FIELDS},
              'thought': 'Plan the next operation. '*12,
              'action': 'replace wrong_value with corrected_value',
              'observation': 'The edit completed. '*30,
              'state': f'name=agent{i//2}; role=assistant'} for i in range(n)]
    case = {'steps': steps, 'task': 'corrected_value', 'test_code': '', 'error': ''}
    mapping = {'priority': [1/(i+1) for i in range(n)], 'attention': np.eye(n).tolist(),
               'relations': [], 'original_relations': []}
    view = {'content': 'Overview\n<expanded_records>\nold\n</expanded_records>', 'budget': 2400}
    return case, mapping, view


def test_candidate_order_is_hidden_and_actions_are_original():
    case, mapping, old = fixture()
    a = build_evidence_view(case, mapping, old, ['5','1'])
    assert a == build_evidence_view(case, mapping, old, ['1','5'])
    assert a['context_tokens'] <= 2400
    for i in ('1','5'):
        assert any(f['step_id']==i and f['field']=='action' and
                   f['text']==case['steps'][int(i)]['action'] for f in a['fragments'])
    contaminated = deepcopy(case)
    contaminated.update(mistake_step='SECRET', ground_truth='SECRET')
    assert build_evidence_view(contaminated, mapping, old, ['5','1']) == a


def test_review_prompt_same_hypotheses_and_all_steps_remain_eligible():
    original = [{'role':'system','content':'diagnose'}, {'role':'user','content':'native rules\nOLD'}]
    a = review_messages(original, {'content':'OLD'}, {'content':'NEW'}, ['5','1'])
    assert a == review_messages(original, {'content':'OLD'}, {'content':'NEW'}, ['1','5'])
    assert '1, 5' in a[1]['content'] and 'any original step IDs' in a[1]['content']
    assert original[1]['content'].endswith('OLD')


def test_handoffs_keep_tool_output_with_its_actor_and_normalize_thought():
    case, _, _ = fixture(4)
    for step, state in zip(case['steps'], ['name=; role=Planner (thought)',
            'name=Planner; role=assistant', 'name=Computer_terminal; role=user', 'name=Worker; role=assistant']):
        step['state'] = state
    case['steps'][2]['thought'] = ''
    assert handoff_episodes(case['steps']) == [[0,1,2],[3]]


def test_coverage_preserves_budget_and_label_independence():
    case, mapping, old = fixture(16)
    view = build_coverage_view(case, mapping, old)
    assert view['context_tokens'] == tokens(view['content']) <= 2400
    assert len(view['episodes']) == 8
    assert np.isclose(sum(view['coverage_weights']),1)
    assert len({int(i)//2 for i in view['breadth_ids']}) >= 3
    for f in view['fragments']:
        assert f['text'] == case['steps'][int(f['step_id'])][f['field']][f['start']:f['end']]
    contaminated = deepcopy(case)
    contaminated['mistake_reason'] = 'SECRET'
    assert build_coverage_view(contaminated, mapping, old) == view


def test_coverage_marginal_gain_diminishes_for_fixed_kernel():
    mapping={'attention':[[0,.8,.2],[.5,0,.5],[.4,.6,0]]}
    k=coverage_kernel(mapping); w=coverage_weights([[0,1],[2]],3)
    small=np.zeros(3); large=k[:,0]
    assert float(w@np.maximum(k[:,2]-small,0)) >= float(w@np.maximum(k[:,2]-large,0))
    assert np.all(np.diag(k)==1)


def test_oversized_atomic_action_is_not_sliced_to_force_a_fit():
    case,mapping,old=fixture()
    case['steps'][1]['action']='{"payload":"'+'abc '*6000+'"}'
    view=build_evidence_view(case,mapping,old,['1','5'])
    assert view['context_tokens']<=2400
    assert not any(f['field']=='action' and f['step_id']=='1' for f in view['fragments'])
