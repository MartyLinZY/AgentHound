import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_contextmap_balanced import complete, family_correction, paired


def test_complete_requires_gap_free_exact_field_coverage():
    step = {'id': '2', 'thought': 'abcdef'}
    fragments = [{'step_id': 2, 'field': 'thought', 'start': 0, 'end': 2},
                 {'step_id': '2', 'field': 'thought', 'start': 3, 'end': 6}]
    assert not complete({'fragments': fragments}, step, 'thought')
    fragments[1]['start'] = 2
    assert complete({'fragments': fragments}, step, 'thought')


def test_primary_family_holm_is_monotonic_and_capped():
    results = {'a': {'contrasts': {'x': {'p': .01}, 'y': {'p': .03}}},
               'b': {'contrasts': {'z': {'p': .9}}}}
    result = family_correction(results)
    assert result['a/x']['p_holm'] == .03
    assert result['a/y']['p_holm'] == .06
    assert result['b/z']['p_holm'] == .9


def test_pair_keeps_all_cases_and_direction():
    cases = {str(i): {'group': str(i)} for i in range(4)}
    rows = {'a': {str(i): {'top5': v} for i, v in enumerate([1, 1, 1, 0])},
            'b': {str(i): {'top5': v} for i, v in enumerate([1, 0, 0, 1])}}
    result = paired(rows, 'a', 'b', 'top5', cases)
    assert result['mean'] == .25
    assert result['gained'] == ['1', '2'] and result['lost'] == ['3']
