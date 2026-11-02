"""Exact structural units and label-free, relation-supported reading windows."""
from __future__ import annotations

import json
import re

from .context_map import FIELDS, render_packet, tokens, words

NODE_BUDGET = 2736  # Same sum as v1's seven field ceilings.


def structural_units(source: str, ceiling: int = NODE_BUDGET) -> list[tuple[int, int, str]]:
    """Partition at explicit boundaries, never cut a fence, JSON call, or diff hunk."""
    if not source:
        return []
    if tokens(source) <= ceiling:
        return [(0, len(source), 'field')]
    try:
        value = json.loads(source)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, (dict, list)):
        return [(0, len(source), 'json')]
    units: list[tuple[int, int, str]] = []
    start = offset = 0
    fence: tuple[str, int] | None = None
    kind = 'paragraph'

    def emit(end: int) -> None:
        nonlocal start
        if end > start:
            units.append((start, end, kind))
        start = end

    for line in source.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            offset += len(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                emit(offset)
                fence = None
                kind = 'paragraph'
            continue
        if marker:
            emit(offset)
            kind, fence = 'fence', (marker[1][0], len(marker[1]))
        elif line.startswith(('@@ ', 'diff --git ')):
            emit(offset)
            kind = 'diff_hunk' if line.startswith('@@ ') else 'diff_header'
        offset += len(line)
        if not line.strip() and fence is None and kind not in ('diff_hunk', 'diff_header'):
            emit(offset)
    emit(len(source))
    return units


def unit_fragments(step: dict, query: str) -> tuple[list[dict], dict]:
    """Reallocate a fixed node budget to whole decision/action/observation units."""
    remaining = NODE_BUDGET
    q = set(words(query))
    chosen: list[dict] = []
    omitted: list[dict] = []
    for field in FIELDS:
        source = step[field]
        candidates = structural_units(source)
        candidates.sort(key=lambda u: (-len(q & set(words(source[u[0]:u[1]]))), u[0]))
        selected = []
        for start, end, kind in candidates:
            size = tokens(source[start:end])
            if size <= remaining:
                selected.append({'step_id': step['id'], 'field': field, 'start': start,
                                 'end': end, 'text': source[start:end]})
                remaining -= size
            else:
                omitted.append({'field': field, 'start': start, 'end': end, 'kind': kind,
                                'tokens': size, 'reason': 'atomic_unit_exceeds_node_budget'
                                if size > NODE_BUDGET else 'remaining_node_budget'})
        chosen.extend(sorted(selected, key=lambda f: f['start']))
    return chosen, {'omitted_units': omitted, 'content_tokens': NODE_BUDGET - remaining}


def windows(selected: list[int], mapping: dict) -> list[list[int]]:
    """Greedy disjoint relation neighborhoods; chronological inside each window."""
    remaining = set(selected)
    result = []
    for anchor in selected:
        if anchor not in remaining:
            continue
        remaining.remove(anchor)
        neighbors = [r for r in mapping['relations'] if r['i'] == anchor and r['j'] in remaining]
        window = [anchor]
        for earlier in (True, False):
            side = [r for r in neighbors if (r['j'] < anchor) == earlier]
            if side:
                best = min(side, key=lambda r: (all(t == 'adjacent_context' for t in r['types']),
                                               -mapping['attention'][anchor][r['j']], r['j']))
                window.append(best['j'])
                remaining.remove(best['j'])
        result.append(sorted(window))
    return result


def render(prefix: str, spans: dict[int, list[dict]], selected: list[int], order: list[int], mapping: dict) -> str:
    sections = []
    for i in order:
        nearby = sorted((r for r in mapping['relations'] if r['i'] == i and r['j'] in selected),
                        key=lambda r: (-mapping['attention'][i][r['j']], r['j']))[:4]
        if nearby:
            sections.append('Reading links from step ' + str(i) + ': ' + '; '.join(
                f"step {r['j']} ({', '.join(r['types'])}; attention={mapping['attention'][i][r['j']]:.3f})"
                for r in nearby))
        sections.append(render_packet(str(i), spans[i]))
    return prefix + '<expanded_records>\n' + '\n\n'.join(sections) + '\n</expanded_records>'


def build_unit_views(case: dict, mapping: dict, original: dict) -> dict:
    query = '\n'.join(case[k] for k in ('task', 'test_code', 'error'))
    spans, audits = {}, {}
    for i, step in enumerate(case['steps']):
        spans[i], audits[str(i)] = unit_fragments(step, query)
    prefix, marker, _ = original['content'].partition('<expanded_records>\n')
    if not marker:
        raise ValueError('Missing frozen overview boundary')
    budget = original['budget']
    priority = sorted(spans, key=lambda i: (-mapping['priority'][i], i))
    selected: list[int] = []
    for i in priority:
        proposed = selected + [i]
        if spans[i] and tokens(render(prefix, spans, proposed, proposed, mapping)) <= budget:
            selected = proposed
    while True:
        groups = windows(selected, mapping)
        chronological = [i for group in groups for i in group]
        contents = {'units': render(prefix, spans, selected, selected, mapping),
                    'units_window': render(prefix, spans, selected, chronological, mapping)}
        if max(tokens(v) for v in contents.values()) <= budget:
            break
        selected.pop()
    shared = [f for i in sorted(selected) for f in spans[i]]
    for f in shared:
        if f['text'] != case['steps'][int(f['step_id'])][f['field']][f['start']:f['end']]:
            raise ValueError('Span differs from source')
    return {method: {'content': content, 'context_tokens': tokens(content), 'budget': budget,
                     'selected_ids': [str(i) for i in selected], 'fragments': shared,
                     'presentation_order': [str(i) for i in (selected if method == 'units' else chronological)],
                     'presentation_windows': [[str(i) for i in group] for group in groups] if method == 'units_window' else [],
                     'unit_audit': audits}
            for method, content in contents.items()}
