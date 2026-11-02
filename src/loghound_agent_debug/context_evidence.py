"""Expand recorded candidate evidence without importing attribution labels."""
from __future__ import annotations

from copy import deepcopy

from .context_map import FIELDS, tokens, words
from .context_units import render, structural_units, unit_fragments

REVIEW_INSTRUCTION = """Re-examine the following unranked hypotheses from an earlier reader: {candidates}.
They are listed in chronological order, not confidence order; the earlier ranking and explanation are unavailable. Treat each as a hypothesis to check, not an answer to preserve. Compare the recorded decision, actual action and observed outcome. Distinguish a proposed change, an executed change, an unsuccessful operation, and a later correction or reintroduction. Trace whether later behavior adopts an earlier choice or introduces a new error. Use the original attribution guidelines to resolve these distinctions.
You may reject all proposed hypotheses and nominate any original step IDs, including those outside the hypothesis set. Return the same JSON schema and exactly min(5, number of original steps) distinct ranked IDs."""


def review_messages(original: list[dict], old_view: dict, new_view: dict,
                    candidates: list[str]) -> list[dict]:
    result = deepcopy(original)
    if result[1]['content'].count(old_view['content']) != 1:
        raise ValueError('Expected one original evidence view')
    ordered = sorted(set(candidates), key=int)
    result[1]['content'] = result[1]['content'].replace(old_view['content'], new_view['content'], 1)
    result[1]['content'] += '\n\n' + REVIEW_INSTRUCTION.format(candidates=', '.join(ordered))
    return result


def evidence_queue(step: dict, query: str) -> list[dict]:
    """Round-robin complete action, decision and output units, retaining exact spans."""
    q = set(words(query))
    fields = ('action', 'thought', 'response', 'observation', 'state', 'extra_info', 'execution_time')
    by_field = {}
    for field in fields:
        source = step[field]
        units = structural_units(source)
        units.sort(key=lambda u: (-len(q & set(words(source[u[0]:u[1]]))), u[0]))
        by_field[field] = [{'step_id': step['id'], 'field': field, 'start': a, 'end': b,
                            'text': source[a:b]} for a, b, _ in units]
    return [by_field[f][round_id] for round_id in range(max(map(len, by_field.values()), default=0))
            for f in fields if round_id < len(by_field[f])]


def supporting_steps(candidates: list[int], mapping: dict, n: int) -> list[int]:
    """Read original plan/state links, then other context links and immediate neighbors."""
    result = []
    def add(i: int) -> None:
        if 0 <= i < n and i not in candidates and i not in result:
            result.append(i)
    for earlier in (True, False):
        for i in candidates:
            links = [r for r in mapping['relations'] if r['i'] == i and (r['j'] < i) == earlier]
            links.sort(key=lambda r: (
                not any('plan_adoption' in t or 'state_dependency' in t for t in r['types']),
                not any(t.startswith(('earlier:', 'later:')) for t in r['types']),
                -mapping['attention'][i][r['j']], abs(r['j']-i), r['j']))
            for r in links[:2]:
                add(r['j'])
            add(i-1 if earlier else i+1)
    # A second recorded plan/state link can expose an origin behind a late execution.
    for i in list(result):
        links = [r for r in mapping['original_relations'] if r['i'] == i and
                 any('plan_adoption' in t or 'state_dependency' in t for t in r['types'])]
        for r in sorted(links, key=lambda r: (-mapping['attention'][i][r['j']], r['j']))[:1]:
            add(r['j'])
    return result


def build_evidence_view(case: dict, mapping: dict, original: dict,
                        candidate_ids: list[str]) -> dict:
    candidates = sorted({int(i) for i in candidate_ids})
    if not candidates or any(i < 0 or i >= len(case['steps']) for i in candidates):
        raise ValueError('Invalid candidate set')
    prefix, marker, _ = original['content'].partition('<expanded_records>\n')
    if not marker:
        raise ValueError('Missing original overview')
    budget = original['budget']
    query = '\n'.join(case[k] for k in ('task', 'test_code', 'error'))
    queues = {i: evidence_queue(s, query) for i, s in enumerate(case['steps'])}
    neighbors = supporting_steps(candidates, mapping, len(case['steps']))
    spans: dict[int, list[dict]] = {}
    phases: dict[str, str] = {}

    def attempt(i: int, fs: list[dict], ceiling: int, phase: str) -> bool:
        old = {(f['field'], f['start'], f['end']) for f in spans.get(i, [])}
        extra = [f for f in fs if (f['field'], f['start'], f['end']) not in old]
        if not extra:
            return False
        trial = {**spans, i: sorted(spans.get(i, [])+extra,
                 key=lambda f: (FIELDS.index(f['field']), f['start']))}
        order = sorted(trial, key=lambda j: (-mapping['priority'][j], j))
        if tokens(render(prefix, trial, order, order, mapping)) > ceiling:
            return False
        spans.update(trial)
        phases[str(i)] = phase
        return True

    # All anchors get a turn before any anchor receives its next evidence unit.
    for round_id in range(max(len(queues[i]) for i in candidates)):
        for i in candidates:
            if round_id < len(queues[i]):
                attempt(i, [queues[i][round_id]], int(.8*budget), 'candidate_evidence')
    for i in neighbors:
        packet, _ = unit_fragments(case['steps'][i], query)
        core = [f for f in packet if f['field'] in ('thought', 'response')]
        attempt(i, core or packet[:1], budget, 'supporting_decision')
    for round_id in range(max(len(q) for q in queues.values())):
        for i in candidates+neighbors:
            if round_id < len(queues[i]):
                attempt(i, [queues[i][round_id]], budget, 'remaining_evidence')
    order = sorted(spans, key=lambda i: (-mapping['priority'][i], i))
    content = render(prefix, spans, order, order, mapping)
    return {'content': content, 'context_tokens': tokens(content), 'budget': budget,
            'selected_ids': [str(i) for i in order], 'presentation_order': [str(i) for i in order],
            'fragments': [f for i in sorted(spans) for f in spans[i]],
            'candidate_ids': [str(i) for i in candidates], 'supporting_ids': [str(i) for i in neighbors],
            'allocation_phases': phases, 'candidate_budget_fraction': .8}
