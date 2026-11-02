#!/usr/bin/env python3
"""Freeze and run the two missing presentation cells against unchanged v1 facts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from pathlib import Path
import random
import shutil

from prepare_contextmap_presentation_controls import render
from run_contextmap import ROOT, read, sha, run_one
from suite_transport import digest, save, now
from loghound_agent_debug.context_map import tokens

SOURCE = ROOT / 'outputs/attentionmap-context-v1-20261028'
CONTROLS = ROOT / 'outputs/attentionmap-context-v2-planning-20261028/presentation_controls'
OUT = ROOT / 'outputs/attentionmap-context-v2-presentation-20261028'
METHODS = ('chronological_links', 'priority_without_links')


def validate(prepared: dict, original: dict, mapping: dict, method: str) -> None:
    """Reconstruct the sole permitted change, including the exact prompt."""
    v = original['view']
    order = sorted(v['selected_ids'], key=int) if method == METHODS[0] else v['presentation_order']
    content = render(v, mapping, order, method == METHODS[0])
    expected = {**v, 'content': content, 'presentation_order': order, 'context_tokens': tokens(content)}
    if prepared['view'] != expected:
        raise ValueError('Evidence, ordering, or view metadata changed')
    messages = [dict(m) for m in original['messages']]
    if messages[1]['content'].count(v['content']) != 1:
        raise ValueError('Ambiguous context replacement')
    messages[1]['content'] = messages[1]['content'].replace(v['content'], content, 1)
    if (prepared['messages'] != messages or prepared['messages_sha256'] != digest(messages)
            or prepared['step_ids'] != original['step_ids']
            or prepared['case_id'] != original['case_id'] or prepared['method'] != method
            or prepared['message_tokens_proxy'] != sum(tokens(m['content']) for m in messages)
            or tokens(content) > v['budget']):
        raise ValueError('Frozen request or budget changed')


def prepare(out: Path, source: Path = SOURCE, controls: Path = CONTROLS) -> dict:
    if out.resolve() in (source.resolve(), controls.resolve()):
        raise ValueError('Output must be separate from existing inputs')
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        unsigned = {k: v for k, v in manifest.items() if k != 'fingerprint'}
        if digest(unsigned) != manifest['fingerprint']:
            raise ValueError('Manifest changed')
        for path, expected in manifest['source_hashes'].items():
            if sha(Path(path)) != expected:
                raise ValueError('Source changed: ' + path)
        for path, expected in manifest['prepared_hashes'].items():
            if sha(out / path) != expected:
                raise ValueError('Prepared input changed: ' + path)
        return manifest
    baseline = read(source / 'manifest.json')
    inventory = read(controls / 'manifest.json')
    expected = {(c['case_id'], m) for c in baseline['cases'] for m in METHODS}
    records = inventory['records']
    if len(records) != len(expected) or {(r['case_id'], r['method']) for r in records} != expected:
        raise ValueError('Missing or duplicated control inputs')
    hashes = {}
    def tracked(path: Path) -> dict:
        hashes[str(path.resolve())] = sha(path)
        return read(path)
    tracked(source / 'manifest.json')
    tracked(controls / 'manifest.json')
    cfg = tracked(source / 'config.json')
    if cfg != baseline['config']:
        raise ValueError('Baseline configuration drift')
    cfg = {**cfg, 'methods': list(METHODS)}
    for record in records:
        cid, method = record['case_id'], record['method']
        original_path = source / 'prepared/attention' / f'{cid}.json'
        map_path = source / 'maps' / f'{cid}.json'
        for path in (original_path, map_path):
            if sha(path) != baseline['prepared_hashes'][str(path.relative_to(source))]:
                raise ValueError('Frozen baseline artifact changed')
        original, mapping = tracked(original_path), tracked(map_path)
        prepared = tracked(controls / method / f'{cid}.json')
        validate(prepared, original, mapping, method)
        if record['messages_sha256'] != prepared['messages_sha256']:
            raise ValueError('Control inventory hash mismatch')
        save(out / 'prepared' / method / f'{cid}.json', prepared)
    for path in (Path(__file__).resolve(), ROOT / 'scripts/run_contextmap.py',
                 ROOT / 'scripts/prepare_contextmap_presentation_controls.py',
                 ROOT / 'scripts/suite_transport.py', ROOT / 'src/loghound_agent_debug/context_map.py'):
        hashes[str(path)] = sha(path)
        target = out / 'snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(out / 'config.json', cfg)
    prepared_hashes = {str(p.relative_to(out)): sha(p) for p in (out / 'prepared').rglob('*.json')}
    prepared_hashes['config.json'] = sha(out / 'config.json')
    manifest = {'config': cfg, 'cases': baseline['cases'], 'n': baseline['n'],
                'source_hashes': hashes, 'prepared_hashes': prepared_hashes, 'created_at': now(),
                'comparison_scope': 'development diagnostic; v1 cells were run at a different time'}
    manifest['fingerprint'] = digest(manifest)
    save(manifest_path, manifest)
    return manifest


def run(out: Path, manifest: dict, key: str, pilot: bool) -> dict:
    cases = manifest['cases']
    if pilot:
        ordered = sorted(cases, key=lambda c: read(out / 'prepared' / METHODS[0] / f"{c['case_id']}.json")['message_tokens_proxy'])
        cases = list({c['case_id']: c for c in (ordered[len(ordered)//2], ordered[-1])}.values())
    else:
        gate = read(out / 'pilot_status.json')
        if gate['fingerprint'] != manifest['fingerprint'] or not gate['technical_gate_passed']:
            raise ValueError('Technical pilot must pass for this experiment')
    tasks = [(c['case_id'], m) for c in cases for m in METHODS]
    random.Random(20261028).shuffle(tasks)
    counts = Counter()
    with ThreadPoolExecutor(max_workers=manifest['config']['workers']) as pool:
        futures = [pool.submit(run_one, out, manifest, cid, method, key) for cid, method in tasks]
        for future in as_completed(futures):
            counts[future.result()['status']] += 1
            save(out / 'progress.json', {'completed': sum(counts.values()), 'planned': len(tasks),
                                       'status_counts': dict(counts), 'updated_at': now()})
    result = {'fingerprint': manifest['fingerprint'], 'planned': len(tasks), 'status_counts': dict(counts),
              'technical_gate_passed': counts['valid'] == len(tasks), 'finished_at': now()}
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
            print({'prepared_requests': len(manifest['cases']) * len(METHODS), 'new_api_calls': 0})
            return
        if not args.api_key_file:
            parser.error('--api-key-file required for inference')
        key = args.api_key_file.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid credential file')
        if args.stage in ('pilot', 'all'):
            result = run(args.out, manifest, key, True)
            print(result)
            if not result['technical_gate_passed']:
                raise SystemExit(2)
        if args.stage in ('run', 'all'):
            print(run(args.out, manifest, key, False))


if __name__ == '__main__':
    main()
