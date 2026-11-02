"""Select evidence by marginal coverage of observed handoff episodes and map neighborhoods."""
from __future__ import annotations

import re

import numpy as np

from .context_map import render_packet, tokens
from .context_units import render, unit_fragments


def handoff_episodes(steps: list[dict]) -> list[list[int]]:
    episodes: list[list[int]] = []
    previous = ''
    for i, step in enumerate(steps):
        match = re.fullmatch(r'name=(.*); role=(.*)', step['state'])
        actor = (match[1] or match[2]) if match else 'agent'
        actor = re.sub(r'\s*\(thought\)\s*', '', actor, flags=re.I).strip()
        if 'terminal' in actor.lower() or (not step['thought'] and not step['response']):
            actor = previous or actor
        if actor != previous or not episodes:
            episodes.append([])
        episodes[-1].append(i)
        previous = actor
    return episodes


def coverage_kernel(mapping: dict) -> np.ndarray:
    attention = np.array(mapping['attention'], dtype=float)
    kernel = attention / np.maximum(attention.max(axis=1, keepdims=True), 1e-12)
    np.fill_diagonal(kernel, 1.)
    return kernel


def coverage_weights(episodes: list[list[int]], n: int) -> np.ndarray:
    weights = np.zeros(n)
    for episode in episodes:
        weights[episode] = 1 / (len(episodes)*len(episode))
    return weights


def build_coverage_view(case: dict, mapping: dict, original: dict) -> dict:
    query = '\n'.join(case[k] for k in ('task', 'test_code', 'error'))
    full, cores, audits = {}, {}, {}
    for i, step in enumerate(case['steps']):
        full[i], audits[str(i)] = unit_fragments(step, query)
        decision = [f for f in full[i] if f['field'] in ('thought', 'response')]
        cores[i] = decision or full[i][:1]
    prefix, marker, _ = original['content'].partition('<expanded_records>\n')
    if not marker:
        raise ValueError('Missing overview')
    budget = original['budget']
    priority = sorted(full, key=lambda i: (-mapping['priority'][i], i))
    episodes = handoff_episodes(case['steps'])
    kernel = coverage_kernel(mapping)
    weights = coverage_weights(episodes, len(full))
    costs = {i: max(1, tokens(render_packet(str(i), cores[i]))+96) for i in full}
    spans: dict[int, list[dict]] = {}
    phases = {}

    def attempt(i: int, fs: list[dict], ceiling: int, phase: str) -> bool:
        if not fs:
            return False
        trial = {**spans, i: fs}
        order = sorted(trial, key=lambda j: (-mapping['priority'][j], j))
        if tokens(render(prefix, trial, order, order, mapping)) > ceiling:
            return False
        spans.update(trial)
        phases[str(i)] = phase
        return True

    def cover(ceiling: int, phase: str) -> None:
        available = {i for i in full if i not in spans and cores[i]}
        covered = kernel[:, list(spans)].max(axis=1) if spans else np.zeros(len(full))
        while available:
            gains = {i: float(weights @ np.maximum(kernel[:, i]-covered, 0)) for i in available}
            i = min(available, key=lambda j: (-gains[j]/costs[j], -mapping['priority'][j], j))
            available.remove(i)
            if attempt(i, cores[i], ceiling, phase):
                covered = np.maximum(covered, kernel[:, i])

    # Match v4's breadth/depth fractions; change only breadth selection.
    cover(int(.65*budget), 'episode_graph_coverage')
    breadth_ids = list(spans)
    for i in priority:
        attempt(i, full[i], budget, 'evidence_depth')
    cover(budget, 'residual_coverage')
    order = sorted(spans, key=lambda i: (-mapping['priority'][i], i))
    content = render(prefix, spans, order, order, mapping)
    return {'content': content, 'context_tokens': tokens(content), 'budget': budget,
            'selected_ids': [str(i) for i in order], 'presentation_order': [str(i) for i in order],
            'fragments': [f for i in sorted(spans) for f in spans[i]], 'unit_audit': audits,
            'breadth_ids': [str(i) for i in breadth_ids], 'allocation_phases': phases,
            'episodes': episodes, 'coverage_weights': weights.tolist(),
            'breadth_fraction': .65, 'selection_cost_link_allowance': 96}
