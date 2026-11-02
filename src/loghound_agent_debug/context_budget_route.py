"""Use complete evidence when it fits; otherwise use the bounded context map."""
from __future__ import annotations

from .context_map import tokens


def choose_context_method(full_view: dict, bounded_view: dict, budget: int = 16384) -> str:
    if budget <= 0:
        raise ValueError('Budget must be positive')
    full_size, bounded_size = (tokens(v['content']) for v in (full_view, bounded_view))
    if full_size != full_view['context_tokens'] or bounded_size != bounded_view['context_tokens']:
        raise ValueError('Reported context size differs from actual content')
    if bounded_size > budget:
        raise ValueError('Bounded fallback exceeds the budget')
    return 'full' if full_size <= budget else 'balanced'
