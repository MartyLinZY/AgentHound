#!/usr/bin/env python3
"""Prepare the two missing presentation controls, reusing exact v1 evidence.

No labels, response text, credentials, or inference APIs are read.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from loghound_agent_debug.context_map import render_packet, tokens

SOURCE = ROOT / 'outputs/attentionmap-context-v1-20261028'
OUT = ROOT / 'outputs/attentionmap-context-v2-planning-20261028/presentation_controls'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def render(view: dict, mapping: dict, order: list[str], with_links: bool) -> str:
    marker = '<expanded_records>\n'
    prefix, marker_found, _ = view['content'].partition(marker)
    if not marker_found:
        raise ValueError('Missing frozen context delimiter')
    spans = {i: [f for f in view['fragments'] if f['step_id'] == i] for i in view['selected_ids']}
    selected = {int(x) for x in view['selected_ids']}
    a = mapping['attention']
    parts = []
    for step in order:
        i = int(step)
        if with_links:
            nearby = sorted((r for r in mapping['relations'] if r['i'] == i and r['j'] in selected),
                            key=lambda r: (-a[i][r['j']], r['j']))[:4]
            if nearby:
                parts.append('Reading links from step ' + str(i) + ': ' + '; '.join(
                    f"step {r['j']} ({', '.join(r['types'])}; attention={a[i][r['j']]:.3f})" for r in nearby))
        parts.append(render_packet(step, spans[step]))
    return prefix + marker + '\n\n'.join(parts) + '\n</expanded_records>'


def main() -> None:
    records = []
    for case in read(SOURCE / 'manifest.json')['cases']:
        cid = case['case_id']
        original = read(SOURCE / 'prepared/attention' / f'{cid}.json')
        flat = read(SOURCE / 'prepared/attention_flat' / f'{cid}.json')
        mapping = read(SOURCE / 'maps' / f'{cid}.json')
        v = original['view']
        chronological = sorted(v['selected_ids'], key=int)
        if render(v, mapping, v['presentation_order'], True) != v['content']:
            raise ValueError('Cannot reconstruct frozen map view')
        if render(v, mapping, chronological, False) != flat['view']['content']:
            raise ValueError('Cannot reconstruct frozen flat view')
        for name, order, links in [('chronological_links', chronological, True),
                                   ('priority_without_links', v['presentation_order'], False)]:
            content = render(v, mapping, order, links)
            prepared = deepcopy(original)
            prepared['method'] = name
            prepared['view'].update(content=content, presentation_order=order, context_tokens=tokens(content))
            user = prepared['messages'][1]['content']
            if user.count(v['content']) != 1:
                raise ValueError('Ambiguous context replacement')
            prepared['messages'][1]['content'] = user.replace(v['content'], content, 1)
            prepared['messages_sha256'] = hashlib.sha256(json.dumps(prepared['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            prepared['message_tokens_proxy'] = sum(tokens(m['content']) for m in prepared['messages'])
            prepared['derived_from'] = {'experiment': 'attentionmap-context-v1-20261028',
                                        'method': 'attention', 'case_id': cid}
            if prepared['view']['fragments'] != flat['view']['fragments'] or tokens(content) > v['budget']:
                raise ValueError('Same facts or common budget violated')
            save(OUT / name / f'{cid}.json', prepared)
            records.append({'case_id': cid, 'method': name, 'tokens': tokens(content),
                            'messages_sha256': prepared['messages_sha256']})
    result = {'cases': len(records) // 2, 'prepared_requests': len(records), 'new_api_calls': 0,
              'original_map_and_flat_exactly_reconstructed': True, 'same_fragments_verified': True,
              'context_budget': 16384, 'max_context_tokens': max(r['tokens'] for r in records),
              'conditions': ['chronological_links', 'priority_without_links'], 'records': records}
    save(OUT / 'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == "__main__":
    main()
