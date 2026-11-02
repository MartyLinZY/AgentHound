#!/usr/bin/env python3
"""Freeze and run matched v5 evidence-review / coverage diagnostics."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from pathlib import Path

from run_contextmap import ROOT, read, sha
from run_contextmap_balanced import freeze, check_manifest, code_sources, run, V1, OUT as PARENT_R
from run_whowhen_context import OUT as PARENT_W
from suite_transport import save, digest, now
from loghound_agent_debug.context_map import tokens
from loghound_agent_debug.context_evidence import build_evidence_view, review_messages
from loghound_agent_debug.context_coverage import build_coverage_view

OUTPUTS = {name: ROOT / ('outputs/attentionmap-v5-'+name+'-20261030') for name in ('rootse', 'whowhen')}
PARENTS = {'rootse': PARENT_R, 'whowhen': PARENT_W}
METHODS = {'rootse': ['review_same', 'review_evidence'], 'whowhen': ['balanced_repeat', 'coverage']}


def sources() -> list[Path]:
    return code_sources()+[Path(__file__).resolve(), ROOT/'scripts/run_whowhen_context.py',
        ROOT/'src/loghound_agent_debug/context_evidence.py', ROOT/'src/loghound_agent_debug/context_coverage.py',
        ROOT/'experiments/attentionmap/CONTEXT_EVIDENCE_V5_PROTOCOL.md']


def prepare(dataset: str) -> dict:
    out, parent = OUTPUTS[dataset], PARENTS[dataset]
    if (out/'manifest.json').exists():
        return check_manifest(out)
    baseline = check_manifest(parent)
    source_files = sources()+[parent/'manifest.json', parent/'labels.json']
    checks = {'prepared_requests': 0, 'identical_messages': 0, 'exact_spans': 0, 'time': now()}
    for c in baseline['cases']:
        cid = c['case_id']
        paths = [parent/'cases'/f'{cid}.json', parent/'prepared/balanced'/f'{cid}.json',
                 (V1 if dataset=='rootse' else parent)/'maps'/f'{cid}.json']
        source_files.extend(paths)
        case, packet, mapping = map(read, paths)
        candidates = None
        if dataset == 'rootse':
            state_file = parent/'runs/balanced'/cid/'state.json'
            state = read(state_file)
            if state['status'] != 'valid':
                raise ValueError('First-round candidate list unavailable')
            source_files.append(state_file)
            candidates = sorted(state['decision']['ranked_step_ids'], key=int)
            new_view = build_evidence_view(case, mapping, packet['view'], candidates)
        else:
            new_view = build_coverage_view(case, mapping, packet['view'])
        save(out/'cases'/f'{cid}.json', case)
        save(out/'maps'/f'{cid}.json', mapping)
        packets = []
        for method, view in zip(METHODS[dataset], (packet['view'], new_view)):
            p = deepcopy(packet)
            p.update(method=method, view=view)
            if dataset == 'rootse':
                p['messages'] = review_messages(packet['messages'], packet['view'], view, candidates)
                p['initial_candidate_ids'] = candidates
            elif method == 'coverage':
                text = p['messages'][1]['content']
                if text.count(packet['view']['content']) != 1:
                    raise ValueError('Evidence block is not unique')
                p['messages'][1]['content'] = text.replace(packet['view']['content'], view['content'], 1)
            p['messages_sha256'] = digest(p['messages'])
            p['message_tokens_proxy'] = sum(tokens(m['content']) for m in p['messages'])
            if tokens(view['content']) != view['context_tokens'] or view['context_tokens'] > 16384:
                raise ValueError('Context budget mismatch')
            for f in view['fragments']:
                if f['text'] != case['steps'][int(f['step_id'])][f['field']][f['start']:f['end']]:
                    raise ValueError('Non-original evidence')
                checks['exact_spans'] += 1
            save(out/'prepared'/method/f'{cid}.json', p)
            packets.append(p)
            checks['prepared_requests'] += 1
        checks['identical_messages'] += packets[0]['messages'] == packets[1]['messages']
        print({'prepared': cid, 'dataset': dataset}, flush=True)
    save(out/'labels.json', read(parent/'labels.json'))
    protected = {str(p): sha(p) for p in (parent/'runs').rglob('*') if p.is_file()}
    save(out/'parent_run_hashes.json', protected)
    source_files.append(out/'parent_run_hashes.json')
    save(out/'input_audit.json', checks)
    cfg = {**baseline['config'], 'methods': METHODS[dataset], 'version': 'context_evidence_v5'}
    return freeze(out, cfg, baseline['cases'], source_files,
                  {'dataset': dataset, 'parent_fingerprint': baseline['fingerprint'],
                   'scope': 'development diagnostic', 'stages': 2 if dataset=='rootse' else 1})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(OUTPUTS), required=True)
    parser.add_argument('--stage', choices=('prepare','pilot','run','all'), default='prepare')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    out = OUTPUTS[args.dataset]
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = prepare(args.dataset)
        if args.stage == 'prepare':
            print({'n':manifest['n'], 'fingerprint':manifest['fingerprint']}); return
        if not args.api_key_file:
            parser.error('--api-key-file required')
        key = args.api_key_file.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid credential')
        if args.stage in ('pilot','all') and not run(out, manifest, key, True)['technical_gate_passed']:
            raise SystemExit(2)
        if args.stage in ('run','all'):
            run(out, manifest, key, False)


if __name__ == '__main__':
    main()
