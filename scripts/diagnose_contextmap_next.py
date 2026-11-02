#!/usr/bin/env python3
"""Offline evidence audit to prioritize context-map changes; no model calls."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/attentionmap-context-v1-20261028'
OUT = ROOT / 'outputs/attentionmap-context-v2-planning-20261028'
METHODS = ('full', 'window', 'graph', 'attention', 'attention_flat')
hashes = {}


def read(path: Path) -> dict:
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def coverage(original: dict, spans: list[dict], fields: tuple[str, ...]) -> dict:
    total = sum(len(original.get(k, '')) for k in fields)
    shown = sum(len(s['text']) for s in spans if s['field'] in fields)
    tail_omitted = [k for k in fields if original.get(k) and
                    max((s['end'] for s in spans if s['field'] == k), default=0) < len(original[k])]
    return {'total_chars': total, 'shown_chars': shown, 'complete': shown == total,
            'fraction': shown / total if total else 1., 'tail_omitted_fields': tail_omitted}


manifest = read(SOURCE / 'manifest.json')
predictions = read(SOURCE / 'analysis/predictions.json')
summary = {}
for method in METHODS:
    valid_wrong = [r for r in predictions[method].values() if not r['top1'] and r['prediction'] is not None]
    summary[method] = {'valid_wrong': len(valid_wrong),
        'later': sum(int(r['prediction']) > int(r['gold'][0]) for r in valid_wrong),
        'earlier': sum(int(r['prediction']) < int(r['gold'][0]) for r in valid_wrong)}

cases = []
for case in manifest['cases']:
    cid = case['case_id']
    views = {m: read(SOURCE / 'prepared' / m / f'{cid}.json')['view'] for m in METHODS}
    root = predictions['attention'][cid]['gold'][0]
    original = {f['field']: f['text'] for f in views['full']['fragments'] if f['step_id'] == root}
    fragments = [f for f in views['attention']['fragments'] if f['step_id'] == root]
    order = views['attention']['presentation_order']
    prediction = predictions['attention'][cid]['prediction']
    root_pos = order.index(root) if root in order else None
    pred_pos = order.index(prediction) if prediction in order else None
    a = set(views['attention']['selected_ids'])
    g = set(views['graph']['selected_ids'])
    record = {'case_id': cid, 'trajectory_id': case['trajectory_id'], 'root': root,
              'prediction': prediction, 'root_expanded': root in order,
              'correct': predictions['attention'][cid]['top1'], 'root_position': root_pos,
              'predicted_position': pred_pos, 'root_fields': {k: coverage(original, fragments, fields)
              for k, fields in {'decision': ('thought', 'response'), 'action': ('action',),
                                'observation': ('observation',)}.items()},
              'graph_map_selection_jaccard': len(a & g) / max(1, len(a | g)),
              'method_predictions': {m: predictions[m][cid]['prediction'] for m in METHODS},
              'method_correct': {m: predictions[m][cid]['top1'] for m in METHODS},
              'presentation_order': order}
    cases.append(record)

expanded = [r for r in cases if r['root_expanded']]
wrong_expanded = [r for r in expanded if not r['correct']]
details = {'n': len(cases), 'expanded': len(expanded), 'wrong_expanded': len(wrong_expanded),
    'decision_incomplete_when_expanded': sum(not r['root_fields']['decision']['complete'] for r in expanded),
    'action_incomplete_when_expanded': sum(not r['root_fields']['action']['complete'] for r in expanded),
    'observation_incomplete_when_expanded': sum(not r['root_fields']['observation']['complete'] for r in expanded),
    'decision_incomplete_wrong_expanded': sum(not r['root_fields']['decision']['complete'] for r in wrong_expanded),
    'decision_tail_missing_when_expanded': sum(bool(r['root_fields']['decision']['tail_omitted_fields']) for r in expanded),
    'both_root_and_prediction_expanded_wrong': sum(r['predicted_position'] is not None for r in wrong_expanded),
    'wrong_prediction_read_before_root': sum(r['predicted_position'] is not None and r['predicted_position'] < r['root_position'] for r in wrong_expanded),
    'graph_map_selection_jaccard_mean': float(np.mean([r['graph_map_selection_jaccard'] for r in cases])),
    'graph_map_identical_selected_sets': sum(r['graph_map_selection_jaccard'] == 1 for r in cases)}
comparisons = {}
for other in ('full', 'graph', 'attention_flat'):
    lost = [r for r in cases if not r['correct'] and r['method_correct'][other]]
    gained = [r for r in cases if r['correct'] and not r['method_correct'][other]]
    comparisons[other] = {'gained': [r['case_id'] for r in gained], 'lost': [r['case_id'] for r in lost],
                         'lost_later': sum(int(r['prediction']) > int(r['root']) for r in lost)}
save(OUT / 'diagnosis.json', {'method_error_direction': summary, 'context_audit': details,
                            'comparisons': comparisons, 'cases': cases})
for path, expected in hashes.items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('Source changed during audit: ' + path)
save(OUT / 'verification.json', {'source_files_unchanged': len(hashes), 'source_hashes': hashes,
                               'new_api_calls': 0, 'new_training': 0})
print(json.dumps({'error_direction': summary, 'context': details}, ensure_ascii=False, indent=2))
