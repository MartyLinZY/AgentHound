import pytest

from loghound_agent_debug.context_map import tokens
from loghound_agent_debug.context_budget_route import choose_context_method


def view(text):
    return {'content': text, 'context_tokens': tokens(text)}


def test_full_at_boundary_and_bounded_above_boundary():
    full, bounded = view('complete evidence ' * 10), view('bounded evidence')
    size = full['context_tokens']
    assert choose_context_method(full, bounded, size) == 'full'
    assert choose_context_method(full, bounded, size-1) == 'balanced'
    assert choose_context_method({**full, 'mistake_step': 'SECRET'}, bounded, size) == 'full'


def test_rejects_invalid_budget_and_false_size_metadata():
    full, bounded = view('all information'), view('some information')
    with pytest.raises(ValueError):
        choose_context_method(full, bounded, 0)
    with pytest.raises(ValueError):
        choose_context_method(full, bounded, 1)
    with pytest.raises(ValueError):
        choose_context_method({**full, 'context_tokens': 1}, bounded)
