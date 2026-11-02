from copy import deepcopy

from loghound_agent_debug.adapters.whowhen_context import safe_whowhen, native_messages


def test_labels_and_reference_answers_cannot_enter_adapter():
    raw = {'question': 'TASK', 'history': [{'name': 'Expert', 'role': 'assistant', 'content': 'plan'},
           {'name': 'Computer_terminal', 'role': 'user', 'content': 'output'},
           {'role': 'Orchestrator (thought)', 'content': 'decision'}],
           'mistake_step': '1', 'mistake_reason': 'SECRET', 'ground_truth': 'SECRET', 'groundtruth': 'SECRET'}
    safe = safe_whowhen(raw)
    other = deepcopy(raw)
    other.update(mistake_step='2', mistake_reason='OTHER', ground_truth='OTHER', groundtruth='OTHER')
    assert safe_whowhen(other) == safe
    assert [s['id'] for s in safe['steps']] == ['0', '1', '2']
    assert safe['steps'][1]['observation'] == 'output'
    assert safe['steps'][2]['thought'] == 'decision'
    assert 'SECRET' not in str(native_messages(safe, {'content': 'VIEW'}))
