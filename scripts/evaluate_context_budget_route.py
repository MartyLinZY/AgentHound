#!/usr/bin/env python3
"""Freeze an input-only routing rule, then replay matching recorded responses."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from run_whowhen_context import OUT
from run_contextmap_balanced import ROOT, check_manifest
from run_contextmap import read, sha
from suite_transport import digest, save, now
from loghound_agent_debug.context_budget_route import choose_context_method
from analyze_contextmap_balanced import describe, paired


def freeze_policy(out: Path = OUT) -> dict:
    manifest = check_manifest(out)
    path = out / 'budget_route_manifest.json'
    if path.exists():
        policy = read(path)
        if digest({k: v for k, v in policy.items() if k != 'fingerprint'}) != policy['fingerprint']:
            raise ValueError('Policy changed')
        for p, expected in policy['code_hashes'].items():
            if sha(Path(p)) != expected:
                raise ValueError('Policy code changed')
        return policy
    choices = {}
    for c in manifest['cases']:
        cid = c['case_id']
        packets = {m: read(out / 'prepared' / m / f'{cid}.json') for m in ('full', 'balanced')}
        method = choose_context_method(packets['full']['view'], packets['balanced']['view'])
        chosen = packets[method]
        choices[cid] = {'method': method, 'messages_sha256': chosen['messages_sha256'],
                        'context_tokens': chosen['view']['context_tokens']}
    sources = [Path(__file__).resolve(), ROOT / 'src/loghound_agent_debug/context_budget_route.py',
               ROOT / 'experiments/attentionmap/BUDGET_ROUTE_PROTOCOL.md']
    policy = {'created_at': now(), 'parent_fingerprint': manifest['fingerprint'], 'budget': 16384,
              'choices': choices, 'route_counts': dict(Counter(v['method'] for v in choices.values())),
              'code_hashes': {str(p): sha(p) for p in sources},
              'external_model_outcomes_read_before_freeze': 0, 'additional_api_calls': 0,
              'scope': 'additional input-only routing evaluation; primary contrasts unchanged'}
    policy['fingerprint'] = digest(policy)
    save(path, policy)
    return policy


def evaluate(out: Path = OUT) -> dict:
    policy = freeze_policy(out)
    manifest = check_manifest(out)
    if policy['parent_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Parent experiment changed')
    summary = read(out / 'analysis/summary.json')
    if summary['verification']['request'] != 552:
        raise ValueError('Parent request audit incomplete')
    rows = read(out / 'analysis/predictions.json')
    cases = {c['case_id']: c for c in manifest['cases']}
    selected, cost = {}, Counter()
    for cid, choice in policy['choices'].items():
        method = choice['method']
        packet = read(out / 'prepared' / method / f'{cid}.json')
        if packet['messages_sha256'] != choice['messages_sha256']:
            raise ValueError('Selected request changed')
        row = deepcopy(rows[method][cid])
        row.update(method='budget_route', source_method=method)
        selected[cid] = row
        call = read(out / 'runs' / method / cid / 'calls/0001/state.json')
        cost['logical_calls'] += 1
        for a in call['attempts']:
            cost['requests'] += 1
            if a.get('usage') is None:
                cost['unknown_usage_requests'] += 1
            else:
                cost.update({k: a['usage'].get(k, 0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
    extended = {**rows, 'budget_route': selected}
    groups = {'all': list(cases), **{m+'_route': [c for c,v in policy['choices'].items() if v['method']==m] for m in ('full','balanced')}}
    result = {'fingerprint': policy['fingerprint'], 'parent_fingerprint': manifest['fingerprint'],
              'route_counts': policy['route_counts'], 'summary': {**describe(list(selected.values()), cases), 'cost_of_selected_calls': dict(cost)},
              'contrasts_descriptive': {f'route_minus_{m}_{k}': paired(extended,'budget_route',m,k,cases)
                                       for m in ('balanced','full') for k in ('top1','top5')},
              'strata': {name:{m:describe([extended[m][c] for c in ids],cases) for m in ('full','balanced','budget_route')}
                        for name,ids in groups.items() if ids}, 'additional_api_calls': 0, 'evaluated_at': now()}
    save(out / 'analysis/budget_route_summary.json', result)
    save(out / 'analysis/budget_route_predictions.json', selected)
    print({'route_counts': result['route_counts'], 'top1':result['summary']['top1'],'top5':result['summary']['top5']}, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('freeze','evaluate'), required=True)
    args=p.parse_args()
    if args.stage=='freeze':
        policy=freeze_policy();print({'fingerprint':policy['fingerprint'],'route_counts':policy['route_counts']})
    else:
        evaluate()


if __name__=='__main__':
    main()
