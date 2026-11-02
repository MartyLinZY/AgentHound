#!/usr/bin/env python3
"""Reconstruct frozen evidence and verify cached replay without network access."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

import run_contextmap as transport_base
from run_contextmap import read, sha
from run_contextmap_balanced import OUT as ROOTSE, V1, V3, check_manifest
from run_contextmap_units import reconstruct
from run_whowhen_context import OUT as WHOWHEN, DATA
from suite_transport import save, digest, now
from loghound_agent_debug.context_map import FIELDS, build_map, build_views, tokens
from loghound_agent_debug.context_units import build_unit_views, structural_units
from loghound_agent_debug.context_balanced import build_balanced_view, aligned_rootse_messages
from loghound_agent_debug.adapters.whowhen_context import safe_whowhen, native_messages
from loghound_agent_debug.features.frozen_semantic import text_key


def audit_inputs(out: Path) -> dict:
    manifest = check_manifest(out)
    counts = {'cases': 0, 'prepared_requests': 0, 'exact_spans': 0, 'reconstructed_maps': 0,
              'max_map_numeric_error': 0.0}
    for c in manifest['cases']:
        cid = c['case_id']
        case = read(out / 'cases' / f'{cid}.json')
        packets = {m: read(out / 'prepared' / m / f'{cid}.json') for m in manifest['config']['methods']}
        if manifest['dataset'] == 'rootse':
            original = read(V1 / 'prepared/attention' / f'{cid}.json')
            mapping = read(V1 / 'maps' / f'{cid}.json')
            unit = read(V3 / 'prepared/units' / f'{cid}.json')
            assert reconstruct(read(V1 / 'prepared/full' / f'{cid}.json')) == case
            expected = {'units_repeat': unit['view'], 'units_aligned': unit['view'],
                        'balanced': build_balanced_view(case, mapping, original['view'])}
            for m, p in packets.items():
                msgs = unit['messages'] if m == 'units_repeat' else aligned_rootse_messages(unit['messages'], unit['view'], expected[m])
                assert p['messages'] == msgs
        else:
            raw = read(DATA / c['trajectory_id'])
            assert safe_whowhen(raw) == case
            poisoned = deepcopy(raw)
            poisoned.update(mistake_step='SENTINEL', mistake_agent='SENTINEL', mistake_reason='SENTINEL',
                            ground_truth='SENTINEL', groundtruth='SENTINEL')
            assert safe_whowhen(poisoned) == case
            texts = [case['task']] + [text for s in case['steps'] for text in (
                '\n'.join(s[f] for f in FIELDS if f != 'observation'), s['observation'])]
            vectors = np.stack([np.load(out / 'embedding_cache' / (text_key(t)+'.npz'), allow_pickle=False)['vector'] for t in texts])
            mapping = build_map(case, [], vectors[1::2], vectors[2::2], vectors[0], vectors[0])
            frozen_map = read(out / 'maps' / f'{cid}.json')
            # v1 BM25 sums over Python sets; process hash order can change the last
            # float64 bit. Check topology exactly and numerical arrays to 1e-12.
            for key in ('relevance', 'attention', 'priority', 'graph_attention', 'graph_priority'):
                delta = float(np.max(np.abs(np.array(mapping[key])-np.array(frozen_map[key]))))
                counts['max_map_numeric_error'] = max(counts['max_map_numeric_error'], delta)
                assert delta <= 1e-12
            assert mapping['original_relations'] == frozen_map['original_relations']
            assert mapping['relations'] == frozen_map['relations']
            # Rendering must reproduce the exact frozen graph used by the requests.
            mapping = frozen_map
            counts['reconstructed_maps'] += 1
            base = build_views(case, mapping)
            expected = {'full': base['full'], 'units_aligned': build_unit_views(case, mapping, base['attention'])['units'],
                        'balanced': build_balanced_view(case, mapping, base['attention'])}
            for m, p in packets.items():
                assert p['messages'] == native_messages(case, expected[m])
        for method, p in packets.items():
            view = p['view']
            assert view == expected[method]
            assert digest(p['messages']) == p['messages_sha256']
            assert p['message_tokens_proxy'] == sum(tokens(m['content']) for m in p['messages'])
            assert tokens(view['content']) == view['context_tokens']
            assert view['budget'] is None or view['context_tokens'] <= view['budget']
            assert sorted(view['selected_ids']) == sorted(view['presentation_order'])
            for f in view['fragments']:
                source = case['steps'][int(f['step_id'])][f['field']]
                assert f['text'] == source[f['start']:f['end']]
                if method != 'full':
                    assert (f['start'], f['end']) in {(a, b) for a, b, _ in structural_units(source)}
                counts['exact_spans'] += 1
            counts['prepared_requests'] += 1
        counts['cases'] += 1
        if counts['cases'] % 25 == 0:
            print(manifest['dataset'], counts, flush=True)
    result = {**counts, 'fingerprint': manifest['fingerprint'], 'attribution_fields_used_for_context': 0,
              'network_calls': 0, 'time': now()}
    save(out / 'verification/input_reconstruction.json', result)
    return result


def replay(out: Path) -> dict:
    manifest = check_manifest(out)
    expected = manifest['n'] * len(manifest['config']['methods'])
    if read(out / 'run_status.json')['completed'] != expected:
        raise ValueError('Wait for all calls to finish before replay')
    paths = [p for p in (out / 'runs').rglob('*') if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    def forbidden(*args, **kwargs):
        raise RuntimeError('Replay unexpectedly attempted a network client')
    saved = transport_base.RecordedClient
    transport_base.RecordedClient = forbidden
    count = 0
    try:
        for case in manifest['cases']:
            for method in manifest['config']['methods']:
                result = transport_base.run_one(out, manifest, case['case_id'], method, 'NOT_A_CREDENTIAL')
                assert result['status'] != 'running'
                count += 1
    finally:
        transport_base.RecordedClient = saved
    for path, checksum in before.items():
        assert sha(Path(path)) == checksum
    protected = read(out / 'historical_hashes.json')
    for path, checksum in protected.items():
        assert sha(Path(path)) == checksum
    result = {'cached_cases': count, 'network_calls': 0, 'unchanged_new_files': len(before),
              'unchanged_historical_files': len(protected), 'fingerprint': manifest['fingerprint'], 'time': now()}
    save(out / 'verification/replay.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('inputs', 'replay'), required=True)
    p.add_argument('--dataset', choices=('rootse', 'whowhen', 'both'), default='both')
    args = p.parse_args()
    for name, out in (('rootse', ROOTSE), ('whowhen', WHOWHEN)):
        if args.dataset in (name, 'both'):
            print(name, (audit_inputs if args.stage == 'inputs' else replay)(out), flush=True)


if __name__ == '__main__':
    main()
