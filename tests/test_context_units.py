"""Integrity and controlled-comparison tests; no network."""
import json
import pytest

pytest.importorskip('tiktoken')

from loghound_agent_debug.context_map import FIELDS, tokens
from loghound_agent_debug.context_units import structural_units, unit_fragments, windows, build_unit_views


def test_structural_partition_keeps_code_fence_and_hunk():
    source = 'Plan.\n\n```python\nprint(1)\n\nprint(2)\n```\n\n@@ -1,2 +1,2 @@\n-a\n+b\n\n@@ -8 +8 @@\n-x\n+y\n'
    units = structural_units(source, ceiling=1)
    assert ''.join(source[a:b] for a, b, _ in units) == source
    fences = [source[a:b] for a, b, kind in units if kind == 'fence']
    assert fences == ['```python\nprint(1)\n\nprint(2)\n```\n']
    assert len([u for u in units if u[2] == 'diff_hunk']) == 2


def test_json_tool_call_is_atomic_and_oversize_is_explicit():
    source = json.dumps({'command': 'echo hello\n' * 4000})
    assert structural_units(source) == [(0, len(source), 'json')]
    step = {'id': '0', **{f: '' for f in FIELDS}, 'thought': 'Use the required protocol.', 'action': source}
    spans, audit = unit_fragments(step, 'hello')
    assert [f['field'] for f in spans] == ['thought']
    assert audit['omitted_units'][0]['reason'] == 'atomic_unit_exceeds_node_budget'


def test_long_decision_is_preserved_exactly_and_budget_reallocated():
    decision = 'Retain the user constraint and check the return value. ' * 90
    step = {'id': '0', **{f: '' for f in FIELDS}, 'response': decision,
            'action': 'apply change', 'observation': 'test output\n\n' * 3000}
    spans, audit = unit_fragments(step, 'return value')
    assert any(f['field'] == 'response' and f['text'] == decision for f in spans)
    assert sum(tokens(f['text']) for f in spans) <= 2736
    assert all(f['text'] == step[f['field']][f['start']:f['end']] for f in spans)


def fixture():
    steps = [{'id': str(i), **{f: '' for f in FIELDS}, 'thought': f'original plan {i}'} for i in range(5)]
    mapping = {'priority': [.2, .3, .4, .1, .05], 'attention': [[.2]*5 for _ in range(5)],
               'relations': [{'i': i, 'j': j, 'types': ['shared_path']} for i in range(5) for j in range(5) if i != j]}
    return {'task': 'debug', 'test_code': '', 'error': '', 'steps': steps}, mapping


def test_window_is_disjoint_complete_and_locally_chronological():
    _, mapping = fixture()
    groups = windows([2, 1, 0, 3, 4], mapping)
    assert groups[0] == [0, 2, 3]
    assert sorted(i for g in groups for i in g) == list(range(5))
    assert all(g == sorted(g) and len(g) <= 3 for g in groups)


def test_views_hold_exact_facts_and_links_under_same_budget():
    case, mapping = fixture()
    original = {'content': 'overview\n<expanded_records>\nold\n</expanded_records>', 'budget': 1000}
    result = build_unit_views(case, mapping, original)
    a, b = result['units'], result['units_window']
    assert a['fragments'] == b['fragments']
    assert a['selected_ids'] == b['selected_ids']
    assert sorted(a['presentation_order']) == sorted(b['presentation_order'])
    assert sorted(line for line in a['content'].splitlines() if line.startswith('Reading links')) == sorted(
        line for line in b['content'].splitlines() if line.startswith('Reading links'))
    assert max(a['context_tokens'], b['context_tokens']) <= 1000
    # Extraneous answer-bearing fields cannot enter the context.
    case['failure_id'] = 'SENTINEL'
    assert build_unit_views(case, mapping, original) == result
