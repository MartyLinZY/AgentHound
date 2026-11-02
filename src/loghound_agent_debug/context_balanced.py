"""Breadth before depth, with whole original decision units and a fixed map."""
from __future__ import annotations

from copy import deepcopy

from .context_map import tokens
from .context_units import render, unit_fragments

BREADTH_FRACTION = 0.65

ROOTSE_INSTRUCTION = """Identify the decisive error step according to the annotation guidelines above. Trace each plausible failure explanation through its initial decision, subsequent adoption, implementation, verification, and any correction or reintroduction. A committed erroneous plan may precede the code that implements it; a tentative alternative or an unrelated earlier mistake is not sufficient. Determine whether later steps continue the same error or introduce a different one, using the recorded evidence. Apply the dataset's attribution definition when an error is corrected and later reintroduced.
Compare the strongest competing causal explanations before ranking. The five candidates should reflect uncertainty over individual causal steps; do not fill the list merely with chronological neighbors of the first choice. All original step IDs remain eligible. Reading order and attention weights indicate where to inspect, not root-cause probabilities.
Return exactly min(5, number of original steps) distinct candidate step IDs ranked by likelihood, strongest first. Explain the first choice and cite supporting original step IDs. Return only a JSON object: {"ranked_step_ids": ["ID", "ID"], "reason": "explanation", "evidence_step_ids": ["ID"]}."""


def aligned_rootse_messages(original: list[dict], old_view: dict, new_view: dict) -> list[dict]:
    result = deepcopy(original)
    content = result[1]['content']
    marker = '\nIdentify the earliest uncorrected decisive error responsible for this final test failure.'
    if content.count(marker) != 1 or content.count(old_view['content']) != 1:
        raise ValueError('Unexpected frozen message structure')
    content = content.split(marker, 1)[0]
    result[1]['content'] = content.replace(old_view['content'], new_view['content'], 1) + '\n' + ROOTSE_INSTRUCTION + '\n'
    return result


def build_balanced_view(case: dict, mapping: dict, original: dict) -> dict:
    query = '\n'.join(case[k] for k in ('task', 'test_code', 'error'))
    full, cores, audits = {}, {}, {}
    for i, step in enumerate(case['steps']):
        full[i], audits[str(i)] = unit_fragments(step, query)
        decision = [f for f in full[i] if f['field'] in ('thought', 'response')]
        # For a tool-only record, its first complete action/output unit is its core.
        cores[i] = decision or full[i][:1]
    prefix, marker, _ = original['content'].partition('<expanded_records>\n')
    if not marker:
        raise ValueError('Missing overview boundary')
    budget = original['budget']
    priority = sorted(full, key=lambda i: (-mapping['priority'][i], i))
    selected, spans = [], {}
    phases = {}

    def attempt(i: int, proposed_spans: list[dict], ceiling: int, phase: str) -> bool:
        if not proposed_spans:
            return False
        proposed = sorted(set(selected + [i]), key=lambda j: (-mapping['priority'][j], j))
        trial = {**spans, i: proposed_spans}
        if tokens(render(prefix, trial, proposed, proposed, mapping)) > ceiling:
            return False
        selected[:] = proposed
        spans[i] = proposed_spans
        phases[str(i)] = phase
        return True

    # First spend 65% of total context on distinct decisions, including overview/links.
    for i in priority:
        attempt(i, cores[i], int(budget * BREADTH_FRACTION), 'decision_breadth')
    breadth_ids = list(selected)
    # Then provide full action/output packets at high-priority nodes.
    for i in priority:
        attempt(i, full[i], budget, 'evidence_depth')
    # Use a small residual for another whole decision rather than padding/truncation.
    for i in priority:
        if i not in spans:
            attempt(i, cores[i], budget, 'residual_breadth')
    content = render(prefix, spans, selected, selected, mapping)
    fragments = [f for i in sorted(selected) for f in spans[i]]
    for f in fragments:
        if f['text'] != case['steps'][int(f['step_id'])][f['field']][f['start']:f['end']]:
            raise ValueError('Span differs from original')
    return {'content': content, 'context_tokens': tokens(content), 'budget': budget,
            'selected_ids': [str(i) for i in selected], 'presentation_order': [str(i) for i in selected],
            'fragments': fragments, 'breadth_ids': [str(i) for i in breadth_ids],
            'allocation_phases': phases, 'breadth_fraction': BREADTH_FRACTION, 'unit_audit': audits}
