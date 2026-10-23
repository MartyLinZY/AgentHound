import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_rootse_models import paired_counts


def test_exact_paired_comparison_has_correct_direction_and_tail():
    result = paired_counts([False] * 10, [True] * 10)
    assert result['left_only'] == 0 and result['right_only'] == 10
    assert result['exact_mcnemar_two_sided_p'] == pytest.approx(2 / 1024)
    reverse = paired_counts([True] * 10, [False] * 10)
    assert reverse['exact_mcnemar_two_sided_p'] == result['exact_mcnemar_two_sided_p']
    assert paired_counts([True, False], [True, False])['exact_mcnemar_two_sided_p'] == 1
    with pytest.raises(ValueError):
        paired_counts([True], [])
