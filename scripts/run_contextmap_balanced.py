#!/usr/bin/env python3
"""Frozen v4 RootSE controls; reusable single-call runner for external data."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
from pathlib import Path
import random
import shutil

from run_contextmap import ROOT, read, sha, run_one
from run_contextmap_units import reconstruct
from suite_transport import digest, save, now
from loghound_agent_debug.context_map import tokens
from loghound_agent_debug.context_balanced import build_balanced_view, aligned_rootse_messages

V1 = ROOT / 'outputs/attentionmap-context-v1-20261028'
V3 = ROOT / 'outputs/attentionmap-context-v3-units-20261029'
OUT = ROOT / 'outputs/attentionmap-context-v4-balanced-20261029'
METHODS = ('units_repeat', 'units_aligned', 'balanced')


def check_manifest(out: Path) -> dict:
    m = read(out / 'manifest.json')
    if digest({k: v for k, v in m.items() if k != 'fingerprint'}) != m['fingerprint']:
        raise ValueError('Manifest changed')
    for path, expected in m['source_hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Source changed: ' + path)
    for path, expected in m['prepared_hashes'].items():
        if sha(out / path) != expected:
            raise ValueError('Input changed: ' + path)
    return m


def freeze(out: Path, cfg: dict, cases: list, sources: list[Path], extra: dict) -> dict:
    hashes = {}
    for p in sources:
        hashes[str(p)] = sha(p)
        if p.is_relative_to(ROOT):
            target = out / 'snapshot' / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    save(out / 'config.json', cfg)
    protected = {str(p): sha(p) for directory in (V1, V3, ROOT / 'outputs/attentionmap-context-v2-presentation-20261028')
                 for p in (directory / 'runs').rglob('*') if p.is_file()}
    save(out / 'historical_hashes.json', protected)
    paths = [p for folder in ('prepared', 'cases', 'maps') for p in (out / folder).rglob('*.json')]
    paths += [out / name for name in ('config.json', 'historical_hashes.json', 'labels.json') if (out / name).exists()]
    manifest = {'config': cfg, 'cases': cases, 'n': len(cases), 'created_at': now(),
                'source_hashes': hashes, 'prepared_hashes': {str(p.relative_to(out)): sha(p) for p in paths},
                'seed': 20261029, **extra}
    manifest['fingerprint'] = digest(manifest)
    save(out / 'manifest.json', manifest)
    return manifest


def code_sources() -> list[Path]:
    return [Path(__file__).resolve(), ROOT / 'scripts/run_contextmap.py', ROOT / 'scripts/run_contextmap_units.py',
            ROOT / 'scripts/suite_transport.py', ROOT / 'src/loghound_agent_debug/context_map.py',
            ROOT / 'src/loghound_agent_debug/context_units.py', ROOT / 'src/loghound_agent_debug/context_balanced.py',
            ROOT / 'experiments/attentionmap/CONTEXT_BALANCED_PROTOCOL.md']


def prepare(out: Path = OUT) -> dict:
    if (out / 'manifest.json').exists():
        return check_manifest(out)
    baseline, previous = read(V1 / 'manifest.json'), read(V3 / 'manifest.json')
    sources = code_sources() + [V1 / 'manifest.json', V3 / 'manifest.json', V1 / 'labels.json']
    for c in baseline['cases']:
        cid = c['case_id']
        paths = [V1 / 'prepared/full' / f'{cid}.json', V1 / 'prepared/attention' / f'{cid}.json',
                 V1 / 'maps' / f'{cid}.json', V3 / 'prepared/units' / f'{cid}.json']
        for p in paths:
            origin, manifest = (V3, previous) if p.is_relative_to(V3) else (V1, baseline)
            if sha(p) != manifest['prepared_hashes'][str(p.relative_to(origin))]:
                raise ValueError('Frozen source changed')
        sources.extend(paths)
        full, original, mapping, unit = map(read, paths)
        case = reconstruct(full)
        balanced = build_balanced_view(case, mapping, original['view'])
        save(out / 'cases' / f'{cid}.json', case)
        for method in METHODS:
            p = deepcopy(unit)
            p['method'] = method
            if method != 'units_repeat':
                p['view'] = balanced if method == 'balanced' else unit['view']
                p['messages'] = aligned_rootse_messages(unit['messages'], unit['view'], p['view'])
                p['messages_sha256'] = digest(p['messages'])
                p['message_tokens_proxy'] = sum(tokens(m['content']) for m in p['messages'])
            save(out / 'prepared' / method / f'{cid}.json', p)
        print('Prepared RootSE ' + cid, flush=True)
    save(out / 'labels.json', read(V1 / 'labels.json'))
    cfg = {**baseline['config'], 'methods': list(METHODS), 'version': 'context_balanced_v4'}
    return freeze(out, cfg, baseline['cases'], sources, {'dataset': 'rootse', 'scope': 'development'})


def run(out: Path, manifest: dict, key: str, pilot: bool) -> dict:
    methods = manifest['config']['methods']
    cases = manifest['cases']
    if pilot:
        ordered = sorted(cases, key=lambda c: read(out / 'prepared' / methods[-1] / f"{c['case_id']}.json")['message_tokens_proxy'])
        cases = [ordered[len(ordered)//2], ordered[-1]]
    else:
        gate = read(out / 'pilot_status.json')
        if gate['fingerprint'] != manifest['fingerprint'] or not gate['technical_gate_passed']:
            raise ValueError('Technical pilot failed')
    tasks = [(c['case_id'], m) for c in cases for m in methods]
    random.Random(manifest['seed']).shuffle(tasks)
    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=manifest['config']['workers']) as pool:
        futures = [pool.submit(run_one, out, manifest, cid, method, key) for cid, method in tasks]
        for f in as_completed(futures):
            counts[f.result()['status']] += 1
            progress = {'completed': sum(counts.values()), 'planned': len(tasks), 'status_counts': dict(counts), 'updated_at': now()}
            save(out / 'progress.json', progress)
            print(progress, flush=True)
    result = {**progress, 'fingerprint': manifest['fingerprint'], 'technical_gate_passed': counts['valid'] == len(tasks),
              'case_ids': [c['case_id'] for c in cases], 'finished_at': now()}
    save(out / ('pilot_status.json' if pilot else 'run_status.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--stage', choices=('prepare', 'pilot', 'run', 'all'), default='prepare')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = prepare(args.out)
        if args.stage == 'prepare':
            print({'n': m['n'], 'fingerprint': m['fingerprint']}); return
        if not args.api_key_file:
            parser.error('--api-key-file required')
        key = args.api_key_file.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid credential')
        if args.stage in ('pilot', 'all') and not run(args.out, m, key, True)['technical_gate_passed']:
            raise SystemExit(2)
        if args.stage in ('run', 'all'):
            run(args.out, m, key, False)


if __name__ == '__main__':
    main()
