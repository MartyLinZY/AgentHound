from copy import deepcopy

from loghound_agent_debug.context_balanced import build_balanced_view, aligned_rootse_messages
from loghound_agent_debug.context_map import FIELDS, tokens, messages
from loghound_agent_debug.context_units import build_unit_views


def test_breadth_increases_decision_coverage_without_cutting_fields():
    steps = [{'id': str(i), **{f: '' for f in FIELDS}, 'thought': 'A complete decision. ' * 12,
              'observation': 'Long evidence. ' * 200} for i in range(12)]
    case = {'steps': steps, 'task': 'evidence', 'test_code': '', 'error': ''}
    mapping = {'priority': [1/(i+1) for i in range(12)], 'relations': [], 'attention': [[0]*12 for _ in range(12)]}
    old = {'content': 'overview\n<expanded_records>\nold\n</expanded_records>', 'budget': 3000}
    before = build_unit_views(case, mapping, old)['units']
    after = build_balanced_view(case, mapping, old)
    assert len(after['selected_ids']) > len(before['selected_ids'])
    assert tokens(after['content']) <= 3000
    for f in after['fragments']:
        assert f['text'] == steps[int(f['step_id'])][f['field']][f['start']:f['end']]
        if f['field'] == 'thought':
            assert f['text'] == steps[int(f['step_id'])]['thought']
    contaminated = deepcopy(case)
    contaminated.update(mistake_step='SECRET', mistake_reason='SECRET', ground_truth='SECRET')
    assert build_balanced_view(contaminated, mapping, old) == after


def test_aligned_prompt_preserves_guideline_and_changes_only_defined_suffix_and_view():
    case = {'task': 'TASK', 'test_code': 'TEST', 'error': 'ERROR'}
    old = {'content': 'OLD_VIEW'}
    original = messages(case, old, 'NATIVE_GUIDELINE')
    changed = aligned_rootse_messages(original, old, {'content': 'NEW_VIEW'})
    assert original[0] == changed[0]
    assert 'NATIVE_GUIDELINE' in changed[1]['content']
    assert 'earliest uncorrected' not in changed[1]['content']
    assert 'OLD_VIEW' in original[1]['content'] and 'NEW_VIEW' in changed[1]['content']
