"""Cover audit definitions and paired units, independent of inference."""
import sys
from pathlib import Path
import pytest

pytest.importorskip('tiktoken')
pytest.importorskip('matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_contextmap_units import complete_field, pair


def test_decision_completeness_requires_contiguous_full_coverage():
    step = {'id': '2', 'thought': '0123456789'}
    span = lambda a, b: {'step_id': '2', 'field': 'thought', 'start': a, 'end': b}
    assert complete_field({'fragments': [span(0, 5), span(5, 10)]}, step, 'thought')
    assert not complete_field({'fragments': [span(0, 4), span(5, 10)]}, step, 'thought')
    assert not complete_field({'fragments': []}, step, 'thought')
    assert complete_field({'fragments': []}, {'id': '2', 'thought': ''}, 'thought')


def test_paired_wins_losses_include_all_cases_and_group_dependence():
    cases = {str(i): {'group': str(i // 2)} for i in range(6)}
    left = {str(i): {'top1': int(i in (0, 2, 4))} for i in range(6)}
    right = {str(i): {'top1': int(i in (0, 1))} for i in range(6)}
    result = pair(left, right, cases)
    assert result['gained'] == 2 and result['lost'] == 1
    assert result['mean'] == pytest.approx(1/6)
    assert result['ci95'][0] <= result['mean'] <= result['ci95'][1]
