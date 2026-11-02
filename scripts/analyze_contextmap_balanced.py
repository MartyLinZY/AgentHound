#!/usr/bin/env python3
"""Full-population v4 evaluation, request audit, paired uncertainty and transfer strata."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from run_contextmap_balanced import OUT as ROOTSE, V1, V3, check_manifest
from run_whowhen_context import OUT as WHOWHEN
from run_contextmap import read, sha
from analyze_contextmap import account, grouped
from analyze_contextmap_controls import verify_call
from suite_transport import save, digest, now

METRICS = ('top1', 'top5', 'mrr_at5')


def complete(view: dict, step: dict, field: str) -> bool:
    end = 0
    for a, b in sorted((f['start'], f['end']) for f in view['fragments']
                       if str(f['step_id']) == str(step['id']) and f['field'] == field):
        if a > end:
            return False
        end = max(end, b)
    return end == len(step[field])


def paired(rows: dict, a: str, b: str, metric: str, cases: dict) -> dict:
    values = {cid: rows[a][cid][metric] - rows[b][cid][metric] for cid in cases}
    return {**grouped(values, cases, paired=True), 'a': a, 'b': b, 'metric': metric,
            'gained': [c for c, v in values.items() if v > 0], 'lost': [c for c, v in values.items() if v < 0]}


def describe(values: list[dict], cases: dict) -> dict:
    population = {r['case_id']: cases[r['case_id']] for r in values}
    return {'n': len(values), 'groups': len({c['group'] for c in population.values()}),
            'status_counts': dict(Counter(r['status'] for r in values)),
            **{k: sum(r[k] for r in values) for k in ('top1', 'top5', 'root_expanded', 'root_complete_decision', 'late', 'early')},
            'metrics': {k: grouped({r['case_id']: r[k] for r in values}, population) for k in METRICS},
            'mean_selected_nodes': float(np.mean([r['selected_nodes'] for r in values])),
            'mean_context_tokens_proxy': float(np.mean([r['context_tokens_proxy'] for r in values])),
            'median_seconds': float(np.median([r['seconds'] for r in values])),
            'uniform_random_top1': float(np.mean([1 / r['n_steps'] for r in values])),
            'uniform_random_top5': float(np.mean([min(5, r['n_steps']) / r['n_steps'] for r in values]))}


def analyze(out: Path) -> dict:
    manifest = check_manifest(out)
    methods = manifest['config']['methods']
    finished = read(out / 'run_status.json')
    if finished['fingerprint'] != manifest['fingerprint'] or finished['completed'] != len(methods)*manifest['n']:
        raise ValueError('Experiment is incomplete')
    cases = {r['case_id']: r for r in manifest['cases']}
    labels = read(out / 'labels.json')
    for p, expected in read(out / 'historical_hashes.json').items():
        if sha(Path(p)) != expected:
            raise ValueError('Historical run changed: ' + p)
    predictions, checks, raw_hashes = {}, Counter(), {}
    for method in methods:
        predictions[method] = {}
        for cid, metadata in cases.items():
            packet = read(out / 'prepared' / method / f'{cid}.json')
            directory = out / 'runs' / method / cid
            state, call, verified = verify_call(directory, packet, manifest)
            checks.update(verified)
            ranking = state['decision']['ranked_step_ids'] if state['status'] == 'valid' else []
            gold = labels[cid][0]
            rank = ranking.index(gold)+1 if gold in ranking else None
            case = read(out / 'cases' / f'{cid}.json')
            step, view = case['steps'][int(gold)], packet['view']
            offset = int(ranking[0])-int(gold) if ranking else None
            row = {'case_id': cid, 'method': method, 'dataset': manifest['dataset'],
                   'group': metadata['group'], 'subset': metadata.get('subset', 'rootse'),
                   'n_steps': metadata['n_steps'], 'status': state['status'], 'gold': gold,
                   'ranked_step_ids': ranking, 'rank': rank, 'top1': int(rank == 1), 'top5': int(rank is not None),
                   'mrr_at5': 1/rank if rank else 0., 'offset': offset,
                   'late': int(offset is not None and offset > 0), 'early': int(offset is not None and offset < 0),
                   'root_expanded': int(gold in view['selected_ids']),
                   'root_has_decision': int(bool(step['thought'] or step['response'])),
                   'root_complete_decision': int(bool(step['thought'] or step['response']) and
                                                all(complete(view, step, f) for f in ('thought', 'response'))),
                   'root_complete_action': int(bool(step['action']) and complete(view, step, 'action')),
                   'root_complete_observation': int(bool(step['observation']) and complete(view, step, 'observation')),
                   'selected_nodes': len(view['selected_ids']), 'context_tokens_proxy': view['context_tokens'],
                   'seconds': sum(a.get('elapsed_seconds', 0.) for a in call['attempts']),
                   'requests': len(call['attempts']), 'reason': state.get('decision', {}).get('reason')}
            predictions[method][cid] = row
            for path in directory.rglob('*'):
                if path.is_file():
                    raw_hashes[str(path.relative_to(out))] = sha(path)
    rows = [r for method in methods for r in predictions[method].values()]
    if dict(Counter(r['status'] for r in rows)) != finished['status_counts']:
        raise ValueError('Run status counts mismatch')
    summaries = {m: {**describe(list(predictions[m].values()), cases), 'cost': account(out / 'runs' / m)} for m in methods}
    contrasts = (('units_aligned', 'units_repeat'), ('balanced', 'units_aligned')) if manifest['dataset'] == 'rootse' else (
                 ('balanced', 'units_aligned'), ('balanced', 'full'))
    comparisons = {f'{a}_minus_{b}_{k}': paired(predictions, a, b, k, cases)
                   for a, b in contrasts for k in ('top1', 'top5')}
    complete_cases = {cid: c for cid, c in cases.items() if all(predictions[m][cid]['status'] == 'valid' for m in methods)}
    sensitivity = {'n': len(complete_cases), 'excluded_ids': sorted(set(cases)-set(complete_cases)),
                   'methods': {m: describe([predictions[m][cid] for cid in complete_cases], cases) for m in methods} if complete_cases else {}}
    strata = {}
    subsets = sorted({r['subset'] for r in rows})
    selections = {subset: [c for c in cases if cases[c].get('subset', 'rootse') == subset] for subset in subsets}
    selections.update({'steps_le5': [c for c in cases if cases[c]['n_steps'] <= 5],
                       'steps_6_to20': [c for c in cases if 5 < cases[c]['n_steps'] <= 20],
                       'steps_gt20': [c for c in cases if cases[c]['n_steps'] > 20]})
    if manifest['dataset'] == 'whowhen':
        flagged = {c for pair in read(out / 'dataset_audit.json')['identical_history_different_question_groups'] for c in pair}
        selections['quality_without_reused_histories'] = [c for c in cases if c not in flagged]
        input_rows = read(out / 'input_contrast_audit.json')['rows']
        selections['balanced_changed_request'] = [r['case_id'] for r in input_rows if not r['same_messages']]
        selections['balanced_identical_request'] = [r['case_id'] for r in input_rows if r['same_messages']]
    for name, ids in selections.items():
        if ids:
            strata[name] = {m: describe([predictions[m][c] for c in ids], cases) for m in methods}
    mechanisms = {}
    for a, b in contrasts:
        changes = {}
        for field in ('root_expanded', 'root_complete_decision', 'root_complete_action', 'root_complete_observation'):
            changes[field] = {}
            for label, predicate in (('gained', lambda d: d > 0), ('lost', lambda d: d < 0), ('unchanged', lambda d: d == 0)):
                ids = [c for c in cases if predicate(predictions[a][c][field]-predictions[b][c][field])]
                changes[field][label] = {'n': len(ids), 'ids': ids,
                    'top1_a': sum(predictions[a][c]['top1'] for c in ids), 'top1_b': sum(predictions[b][c]['top1'] for c in ids),
                    'top5_a': sum(predictions[a][c]['top5'] for c in ids), 'top5_b': sum(predictions[b][c]['top5'] for c in ids)}
        mechanisms[f'{a}_minus_{b}'] = changes
    result = {'dataset': manifest['dataset'], 'n': len(cases), 'fingerprint': manifest['fingerprint'],
              'methods': summaries, 'contrasts': comparisons, 'strata': strata, 'complete_case_sensitivity': sensitivity,
              'mechanism_diagnostics_exploratory': mechanisms, 'verification': dict(checks),
              'cost': account(out / 'runs'), 'finished_at': now()}
    if manifest['dataset'] == 'whowhen':
        same = selections['balanced_identical_request']
        result['identical_request_variation'] = {'n': len(same),
            'top1_disagreements': sum(predictions['balanced'][c]['ranked_step_ids'][:1] != predictions['units_aligned'][c]['ranked_step_ids'][:1] for c in same),
            'ranking_disagreements': sum(predictions['balanced'][c]['ranked_step_ids'] != predictions['units_aligned'][c]['ranked_step_ids'] for c in same),
            'note': 'Identical prompts are separate provider calls; differences here do not measure a context change.'}
    analysis = out / 'analysis'
    save(analysis / 'summary.json', result)
    save(analysis / 'predictions.json', predictions)
    save(analysis / 'raw_hashes.json', raw_hashes)
    with (analysis / 'cases.csv').open('w', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def family_correction(results: dict[str, dict]) -> dict:
    entries = [(d + '/' + name, c) for d, r in results.items() for name, c in r['contrasts'].items()]
    adjusted, previous = {}, 0.
    for i, (name, c) in enumerate(sorted(entries, key=lambda x: x[1]['p'])):
        previous = max(previous, min(1., (len(entries)-i)*c['p']))
        adjusted[name] = {**c, 'p_holm': previous}
    return adjusted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('rootse', 'whowhen', 'both'), default='both')
    args = p.parse_args()
    outputs = {'rootse': ROOTSE, 'whowhen': WHOWHEN}
    results = {name: analyze(out) for name, out in outputs.items() if args.dataset in ('both', name)}
    if len(results) == 2:
        adjusted = family_correction(results)
        for out in outputs.values():
            save(out / 'analysis/primary_family_holm.json', adjusted)
    for name, r in results.items():
        print(name, {m: {k: s[k] for k in ('n', 'top1', 'top5', 'root_expanded', 'root_complete_decision')}
                     for m, s in r['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
