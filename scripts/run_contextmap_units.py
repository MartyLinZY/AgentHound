#!/usr/bin/env python3
"""Freeze and execute simultaneous v1 / atomic-unit / relation-window controls."""
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
from suite_transport import digest, save, now
from loghound_agent_debug.context_map import FIELDS, tokens
from loghound_agent_debug.context_units import build_unit_views

SOURCE = ROOT / 'outputs/attentionmap-context-v1-20261028'
OUT = ROOT / 'outputs/attentionmap-context-v3-units-20261029'
METHODS = ('v1_repeat', 'units', 'units_window')


def reconstruct(full: dict) -> dict:
    steps = [{'id': i, **{f: '' for f in FIELDS}} for i in full['step_ids']]
    for span in full['view']['fragments']:
        if span['start'] != 0 or span['end'] != len(span['text']):
            raise ValueError('Full source is not a complete original field')
        steps[int(span['step_id'])][span['field']] = span['text']
    # Exact admitted task/test/error prefix. No annotation labels or model answers.
    query = full['messages'][1]['content'].split('\nAnnotation guidelines:\n', 1)[0]
    return {'steps': steps, 'task': query, 'test_code': '', 'error': ''}


def prepare(out: Path = OUT, source: Path = SOURCE) -> dict:
    if out.resolve() == source.resolve():
        raise ValueError('Output must differ from frozen baseline')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
            raise ValueError('Manifest changed')
        for p, expected in manifest['source_hashes'].items():
            if sha(Path(p)) != expected:
                raise ValueError('Source changed: ' + p)
        for p, expected in manifest['prepared_hashes'].items():
            if sha(out / p) != expected:
                raise ValueError('Prepared data changed: ' + p)
        return manifest
    baseline = read(source / 'manifest.json')
    hashes = {str(source / 'manifest.json'): sha(source / 'manifest.json')}
    cfg = {**baseline['config'], 'methods': list(METHODS), 'version': 'context_units_v3'}
    for record in baseline['cases']:
        cid = record['case_id']
        paths = [source / 'prepared/attention' / f'{cid}.json',
                 source / 'prepared/full' / f'{cid}.json', source / 'maps' / f'{cid}.json']
        for path in paths:
            if sha(path) != baseline['prepared_hashes'][str(path.relative_to(source))]:
                raise ValueError('Frozen baseline changed')
            hashes[str(path)] = sha(path)
        original, full, mapping = map(read, paths)
        views = build_unit_views(reconstruct(full), mapping, original['view'])
        for method in METHODS:
            prepared = deepcopy(original)
            prepared['method'] = method
            if method != 'v1_repeat':
                prepared['view'] = views[method]
                if prepared['messages'][1]['content'].count(original['view']['content']) != 1:
                    raise ValueError('Ambiguous context replacement')
                prepared['messages'][1]['content'] = prepared['messages'][1]['content'].replace(
                    original['view']['content'], prepared['view']['content'], 1)
                prepared['messages_sha256'] = digest(prepared['messages'])
                prepared['message_tokens_proxy'] = sum(tokens(m['content']) for m in prepared['messages'])
            save(out / 'prepared' / method / f'{cid}.json', prepared)
        print(f'Prepared {cid}', flush=True)
    own = [Path(__file__).resolve(), ROOT / 'src/loghound_agent_debug/context_units.py',
           ROOT / 'scripts/run_contextmap.py', ROOT / 'scripts/suite_transport.py',
           ROOT / 'src/loghound_agent_debug/context_map.py',
           ROOT / 'experiments/attentionmap/CONTEXT_UNITS_PROTOCOL.md']
    for path in own:
        hashes[str(path)] = sha(path)
        target = out / 'snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(out / 'config.json', cfg)
    protected = {}
    for previous in (source, ROOT / 'outputs/attentionmap-context-v2-presentation-20261028'):
        for p in (previous / 'runs').rglob('*'):
            if p.is_file():
                protected[str(p)] = sha(p)
    save(out / 'historical_hashes.json', protected)
    prepared_hashes = {str(p.relative_to(out)): sha(p) for p in (out / 'prepared').rglob('*.json')}
    for name in ('config.json', 'historical_hashes.json'):
        prepared_hashes[name] = sha(out / name)
    manifest = {'config': cfg, 'cases': baseline['cases'], 'n': baseline['n'],
                'source_hashes': hashes, 'prepared_hashes': prepared_hashes, 'created_at': now(),
                'comparison_scope': 'development set; three conditions scheduled together',
                'seed': 20261029}
    manifest['fingerprint'] = digest(manifest)
    save(manifest_path, manifest)
    return manifest


def run(out: Path, manifest: dict, key: str, pilot: bool) -> dict:
    cases = manifest['cases']
    if pilot:
        ordered = sorted(cases, key=lambda c: read(out / 'prepared/units' / f"{c['case_id']}.json")['message_tokens_proxy'])
        cases = [ordered[len(ordered)//2], ordered[-1]]
    else:
        gate = read(out / 'pilot_status.json')
        if gate['fingerprint'] != manifest['fingerprint'] or not gate['technical_gate_passed']:
            raise ValueError('Technical pilot not passed')
    tasks = [(c['case_id'], m) for c in cases for m in METHODS]
    random.Random(manifest['seed']).shuffle(tasks)
    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=manifest['config']['workers']) as pool:
        futures = [pool.submit(run_one, out, manifest, cid, method, key) for cid, method in tasks]
        for future in as_completed(futures):
            counts[future.result()['status']] += 1
            progress = {'completed': sum(counts.values()), 'planned': len(tasks),
                        'status_counts': dict(counts), 'updated_at': now()}
            save(out / 'progress.json', progress)
            print(progress, flush=True)
    result = {**progress, 'fingerprint': manifest['fingerprint'],
              'technical_gate_passed': counts['valid'] == len(tasks),
              'case_ids': [c['case_id'] for c in cases], 'finished_at': now()}
    save(out / ('pilot_status.json' if pilot else 'run_status.json'), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--stage', choices=('prepare', 'pilot', 'run', 'all'), default='prepare')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.out)
        if args.stage == 'prepare':
            print({'prepared_requests': 3 * manifest['n'], 'new_api_calls': 0})
            return
        if not args.api_key_file:
            parser.error('--api-key-file required')
        key = args.api_key_file.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid credential file')
        if args.stage in ('pilot', 'all'):
            if not run(args.out, manifest, key, True)['technical_gate_passed']:
                raise SystemExit(2)
        if args.stage in ('run', 'all'):
            run(args.out, manifest, key, False)


if __name__ == '__main__':
    main()
