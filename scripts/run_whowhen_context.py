#!/usr/bin/env python3
"""Freeze Who&When transfer with label-free inputs and the unchanged v4 method."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import re

import pyarrow.parquet as pq
import torch

from run_contextmap_balanced import ROOT, V1, freeze, check_manifest, code_sources, run
from run_contextmap import read, sha
from suite_transport import save, digest
from loghound_agent_debug.adapters.whowhen_context import safe_whowhen, native_messages
from loghound_agent_debug.context_map import FIELDS, build_map, build_views, tokens
from loghound_agent_debug.context_units import build_unit_views
from loghound_agent_debug.context_balanced import build_balanced_view
from loghound_agent_debug.features.frozen_semantic import FrozenTextEncoder

OUT = ROOT / 'outputs/attentionmap-whowhen-v4-20261029'
DATA = ROOT.parents[2] / 'output/attentionmap_v4_generalization_2026-10-29/sources'
METHODS = ('full', 'units_aligned', 'balanced')


def normalize(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip().lower()


def prepare(out: Path = OUT, data: Path = DATA) -> dict:
    if (out / 'manifest.json').exists():
        return check_manifest(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = code_sources() + [Path(__file__).resolve(), ROOT / 'src/loghound_agent_debug/adapters/whowhen_context.py',
              ROOT / 'src/loghound_agent_debug/features/frozen_semantic.py',
              ROOT / 'experiments/attentionmap/WHOWHEN_TRANSFER_PROTOCOL.md', data / 'who_when.json',
              data / 'utils.py', data / 'whowhen_repo_tree.json']
    records, cases, labels, annotations, texts, differences = [], {}, {}, {}, [], []
    metadata = read(data / 'who_when.json')
    if metadata['sha'] != '59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c':
        raise ValueError('Dataset revision changed')
    for subset, count in (('Algorithm-Generated', 126), ('Hand-Crafted', 58)):
        files = sorted((data / 'Who&When' / subset).glob('*.json'), key=lambda p: int(p.stem))
        if len(files) != count:
            raise ValueError('Incomplete release')
        parquet = data / (subset + '.parquet')
        sources.append(parquet)
        lookup = {r['question_ID']: r for r in pq.read_table(parquet).to_pylist()}
        for path in files:
            sources.append(path)
            raw = read(path)
            case = safe_whowhen(raw)
            cid = digest({'dataset': metadata['id'], 'subset': subset, 'file': path.name})[:16]
            gold = str(raw['mistake_step'])
            if not gold.isdigit() or not 0 <= int(gold) < len(case['steps']):
                raise ValueError('Invalid native step ID: ' + str(path))
            alternate = lookup[raw['question_ID']]
            if alternate['history'] != raw['history']:
                raise ValueError('Different histories between release formats')
            if str(alternate['mistake_step']) != gold:
                differences.append({'case_id': cid, 'file': str(path.relative_to(data)), 'question_ID': raw['question_ID'],
                                    'json_step': gold, 'parquet_step': str(alternate['mistake_step']), 'n_steps': len(case['steps'])})
            labels[cid] = [gold]
            annotations[cid] = {k: raw.get(k) for k in ('mistake_step', 'mistake_agent', 'mistake_reason')}
            cases[cid] = case
            records.append({'case_id': cid, 'trajectory_id': f'Who&When/{subset}/{path.name}',
                            'subset': subset, 'question_id': str(raw['question_ID']),
                            'n_steps': len(case['steps']), 'task_hash': digest(normalize(case['task'])),
                            'trajectory_hash': digest(case['steps'])})
            # Task-only query because this dataset does not supply an observed failed test.
            texts.append(case['task'])
            for s in case['steps']:
                texts.extend(['\n'.join(s[f] for f in FIELDS if f != 'observation'), s['observation']])

    # Union repeated question identifiers or identical task text, across subsets.
    parent = {c['case_id']: c['case_id'] for c in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for index, a in enumerate(records):
        for b in records[:index]:
            if (a['question_id'] == b['question_id'] or a['task_hash'] == b['task_hash']
                    or a['trajectory_hash'] == b['trajectory_hash']):
                parent[find(a['case_id'])] = find(b['case_id'])
    for c in records:
        c['group'] = find(c['case_id'])
    rootse = read(V1 / 'manifest.json')
    overlap = []
    for r in rootse['cases']:
        full = read(V1 / 'prepared/full' / f"{r['case_id']}.json")
        task = full['messages'][1]['content'].split('<problem_description>\n', 1)[1].split('\n</problem_description>', 1)[0]
        for c in records:
            if digest(normalize(task)) == c['task_hash']:
                overlap.append([r['case_id'], c['case_id']])
    if overlap:
        raise ValueError('Exact task overlap with development set')
    duplicate_conflicts = [sorted([r['case_id'] for r in records if r['trajectory_hash'] == h])
                          for h in sorted({r['trajectory_hash'] for r in records})
                          if len({r['task_hash'] for r in records if r['trajectory_hash'] == h}) > 1]
    audit = {'n': len(records), 'groups': len({r['group'] for r in records}), 'release_revision': metadata['sha'],
             'invalid_json_labels': 0, 'parquet_json_differences': differences, 'rootse_exact_task_overlap': overlap,
             'unique_trajectory_hashes': len({r['trajectory_hash'] for r in records}),
             'identical_history_different_question_groups': duplicate_conflicts,
             'answer_free': True, 'external_outcomes_read': 0, 'query_fallback': 'task_embedding'}
    save(out / 'dataset_audit.json', audit)
    save(out / 'labels.json', labels)
    save(out / 'annotations.json', annotations)
    torch.set_num_threads(4)
    encoder = FrozenTextEncoder(ROOT / 'outputs/models/all-MiniLM-L6-v2-1110a243', device='cpu')
    vectors, stats = encoder.encode(texts, out / 'embedding_cache', batch_size=32,
                                   progress=lambda p: print(p, flush=True))
    save(out / 'embedding_stats.json', stats)
    cursor = 0
    for c in records:
        cid = c['case_id']; case = cases[cid]; n = len(case['steps'])
        task = vectors[cursor]; decision = vectors[cursor+1:cursor+1+2*n:2]; observation = vectors[cursor+2:cursor+1+2*n:2]
        cursor += 1 + 2*n
        mapping = build_map(case, [], decision, observation, task, task)
        base = build_views(case, mapping)
        units = build_unit_views(case, mapping, base['attention'])['units']
        balanced = build_balanced_view(case, mapping, base['attention'])
        save(out / 'cases' / f'{cid}.json', case)
        save(out / 'maps' / f'{cid}.json', mapping)
        for method, view in zip(METHODS, (base['full'], units, balanced)):
            msgs = native_messages(case, view)
            prepared = {'case_id': cid, 'method': method, 'step_ids': [s['id'] for s in case['steps']],
                        'view': view, 'messages': msgs, 'messages_sha256': digest(msgs),
                        'message_tokens_proxy': sum(tokens(m['content']) for m in msgs)}
            save(out / 'prepared' / method / f'{cid}.json', prepared)
        print('Prepared Who&When ' + cid, flush=True)
    if cursor != len(vectors):
        raise ValueError('Embedding index mismatch')
    cfg = {**read(V1 / 'config.json'), 'methods': list(METHODS), 'version': 'whowhen_context_transfer_v4'}
    sources += [out / 'dataset_audit.json', out / 'embedding_stats.json', out / 'annotations.json']
    sources += [ROOT / 'outputs/models/all-MiniLM-L6-v2-1110a243' / name for name in encoder.identity['files']]
    return freeze(out, cfg, records, sources, {'dataset': 'whowhen', 'scope': 'frozen-method external transfer; no task reference answer',
                                             'query_fallback': 'task', 'graph_source': 'rule/semantic only; no LLM dependency cache'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--stage', choices=('prepare', 'pilot', 'run', 'all'), default='prepare')
    p.add_argument('--api-key-file', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = prepare(args.out, args.data)
        if args.stage == 'prepare':
            print({'n': m['n'], 'fingerprint': m['fingerprint']}); return
        if not args.api_key_file:
            p.error('--api-key-file required')
        key = args.api_key_file.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid credential')
        if args.stage in ('pilot', 'all') and not run(args.out, m, key, True)['technical_gate_passed']:
            raise SystemExit(2)
        if args.stage in ('run', 'all'):
            run(args.out, m, key, False)


if __name__ == '__main__':
    main()
