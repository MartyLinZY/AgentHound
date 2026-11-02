#!/usr/bin/env python3
"""Independently recount saved results without importing the metric implementation."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parents[2] / 'output/attentionmap_v4_generalization_2026-10-29'


def read(path):
    return json.loads(path.read_text())


def recount(out):
    manifest, labels = read(out / 'manifest.json'), read(out / 'labels.json')
    summary = read(out / 'analysis/summary.json')
    metrics, states = {}, {}
    for method in manifest['config']['methods']:
        top1 = top5 = 0
        reciprocal = []
        statuses, usage = Counter(), Counter()
        for metadata in manifest['cases']:
            cid = metadata['case_id']
            state = read(out / 'runs' / method / cid / 'state.json')
            call = read(out / 'runs' / method / cid / 'calls/0001/state.json')
            statuses[state['status']] += 1
            assert state['status'] != 'running'
            ranking = state['decision']['ranked_step_ids'] if state['status'] == 'valid' else []
            assert len(ranking) == len(set(ranking)) <= 5
            assert all(0 <= int(step) < metadata['n_steps'] for step in ranking)
            assert len(labels[cid]) == 1
            gold = labels[cid][0]
            rank = ranking.index(gold) + 1 if gold in ranking else None
            top1 += rank == 1
            top5 += rank is not None
            reciprocal.append(1 / rank if rank else 0)
            states[method, cid] = state
            for attempt in call['attempts']:
                usage['requests'] += 1
                if attempt.get('usage') is None:
                    usage['unknown_usage_requests'] += 1
                else:
                    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        usage[field] += attempt['usage'].get(field, 0)
        n = len(manifest['cases'])
        mrr = sum(reciprocal) / n
        expected = summary['methods'][method]
        assert (top1, top5, n) == (expected['top1'], expected['top5'], expected['n'])
        assert math.isclose(mrr, expected['metrics']['mrr_at5']['mean'], abs_tol=1e-12)
        assert dict(statuses) == expected['status_counts']
        for field, value in usage.items():
            assert value == expected['cost'][field]
        metrics[method] = dict(n=n, top1=top1, top5=top5, mrr_at5=mrr,
                               status_counts=dict(statuses), usage=dict(usage))
    return manifest, labels, metrics, states


def main():
    outputs = {
        'rootse': ROOT / 'outputs/attentionmap-context-v4-balanced-20261029',
        'whowhen': ROOT / 'outputs/attentionmap-whowhen-v4-20261029',
    }
    verified = {name: recount(out) for name, out in outputs.items()}
    manifest, labels, _, states = verified['whowhen']
    out = outputs['whowhen']
    for c in manifest['cases']:
        raw = read(DEST / 'sources' / c['trajectory_id'])
        assert labels[c['case_id']] == [str(raw['mistake_step'])]
        assert c['n_steps'] == len(raw['history'])
    policy = read(out / 'budget_route_manifest.json')
    reported = read(out / 'analysis/budget_route_predictions.json')
    expected = read(out / 'analysis/budget_route_summary.json')
    encoder = tiktoken.get_encoding('cl100k_base')
    top1 = top5 = 0
    reciprocal, lengths, routes = [], [], Counter()
    for cid, choice in policy['choices'].items():
        full = read(out / 'prepared/full' / f'{cid}.json')
        full_size = len(encoder.encode(full['view']['content'], disallowed_special=()))
        method = 'full' if full_size <= 16384 else 'balanced'
        assert choice['method'] == method
        selected = read(out / 'prepared' / method / f'{cid}.json')
        size = len(encoder.encode(selected['view']['content'], disallowed_special=()))
        assert size == choice['context_tokens'] <= 16384
        assert choice['messages_sha256'] == selected['messages_sha256']
        state = states[method, cid]
        ranking = state['decision']['ranked_step_ids'] if state['status'] == 'valid' else []
        assert ranking == reported[cid]['ranked_step_ids']
        rank = ranking.index(labels[cid][0]) + 1 if labels[cid][0] in ranking else None
        top1 += rank == 1
        top5 += rank is not None
        reciprocal.append(1 / rank if rank else 0)
        lengths.append(size)
        routes[method] += 1
    assert (top1, top5) == (expected['summary']['top1'], expected['summary']['top5'])
    assert math.isclose(sum(reciprocal)/184, expected['summary']['metrics']['mrr_at5']['mean'], abs_tol=1e-12)
    assert dict(routes) == policy['route_counts']
    result = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'methods': {name: values[2] for name, values in verified.items()},
        'native_labels_preserved': 184,
        'route': dict(n=184, top1=top1, top5=top5, max_context_tokens=max(lengths),
                      route_counts=dict(routes), additional_api_calls=0),
        'network_calls_in_check': 0,
    }
    (DEST / 'final_independent_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
