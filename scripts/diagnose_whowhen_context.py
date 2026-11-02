#!/usr/bin/env python3
"""Post-result coverage diagnosis; never changes inputs, labels, or rankings."""
from run_whowhen_context import OUT
from run_contextmap import read
from suite_transport import save
from loghound_agent_debug.context_units import unit_fragments


def main():
    rows = read(OUT / 'analysis/predictions.json')['balanced']
    cases = {c['case_id']: c for c in read(OUT / 'manifest.json')['cases']}
    choices = read(OUT / 'budget_route_manifest.json')['choices']
    result = {}
    for name, method in (('over_budget', 'balanced'), ('fits_budget', 'full')):
        ids = [cid for cid, choice in choices.items() if choice['method'] == method]
        misses = []
        for cid in ids:
            row = rows[cid]
            if row['top5']:
                continue
            case = read(OUT / 'cases' / f'{cid}.json')
            view = read(OUT / 'prepared/balanced' / f'{cid}.json')['view']
            mapping = read(OUT / 'maps' / f'{cid}.json')
            gold = row['gold']
            fragments, audit = unit_fragments(case['steps'][int(gold)],
                '\n'.join(case[k] for k in ('task', 'test_code', 'error')))
            priority = sorted(range(len(case['steps'])), key=lambda i: (-mapping['priority'][i], i))
            misses.append({'case_id': cid, 'trajectory_id': cases[cid]['trajectory_id'],
                'gold': gold, 'n_steps': row['n_steps'], 'root_expanded': row['root_expanded'],
                'root_complete_decision': row['root_complete_decision'],
                'priority_rank': priority.index(int(gold))+1, 'root_packet_nonempty': bool(fragments),
                'root_packet_tokens': audit['content_tokens'], 'omitted_units': audit['omitted_units'],
                'context_tokens': view['context_tokens'], 'selected_nodes': row['selected_nodes']})
        result[name] = {'n': len(ids), 'top5': sum(rows[c]['top5'] for c in ids),
            'root_expanded': sum(rows[c]['root_expanded'] for c in ids),
            'root_complete_decision': sum(rows[c]['root_complete_decision'] for c in ids),
            'misses': misses}
    save(OUT / 'analysis/budget_coverage_diagnostic.json', result)
    print({name: {k: v for k, v in s.items() if k != 'misses'} for name, s in result.items()})


if __name__ == '__main__':
    main()
