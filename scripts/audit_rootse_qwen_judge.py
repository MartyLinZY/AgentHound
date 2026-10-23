#!/usr/bin/env python3
"""Read-only output diagnostics; never changes decisions or relaxes validation."""
import csv
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rootse-ours-qwen38-nothinking-causal-v1'


def leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, dict)):
        for child in value.values() if isinstance(value, dict) else value:
            yield from leaves(child)


def match_kind(quote, original):
    if not isinstance(quote, str) or not quote.strip():
        return 'unmatched'
    if quote in original:
        return 'exact'
    normalized = lambda s: re.sub(r'\s+', ' ', s).strip()
    if normalized(quote) in normalized(original):
        return 'whitespace'
    try:
        obj = json.loads(original)
    except (ValueError, TypeError):
        return 'unmatched'
    if isinstance(obj, (dict, list)):
        for leaf in leaves(obj):
            if quote in leaf:
                return 'json_leaf'
            if normalized(quote) in normalized(leaf):
                return 'json_leaf_whitespace'
    return 'unmatched'


def main():
    counts = Counter()
    details = []
    reasons = Counter()
    for path in sorted((OUT / 'judge').glob('*/state.json')):
        state = json.loads(path.read_text())
        if state['status'] != 'valid':
            reasons[state['failure_reason'].split(':', 1)[0] if state['failure_reason'].startswith('JSONDecodeError') else state['failure_reason']] += 1
        base = path.parent / 'calls/0001'
        request = json.loads((base / 'request.json').read_text())
        supplied = json.loads(request['messages'][1]['content'])
        responses = list(base.glob('response-*.json'))
        assert len(responses) == 1, 'Audit expects this run to have one response per judge'
        response = json.loads(responses[0].read_text())['choices'][0]['message']['content']
        try:
            decision = json.loads(response.strip().removeprefix('```json').removesuffix('```').strip())
        except ValueError:
            counts['invalid_json'] += 1
            continue
        by_id = {s['step_id']: s for s in supplied['trajectory']}
        matches = []
        try:
            for e in decision['evidence']:
                matches.append(match_kind(e['quote'], by_id[e['step_id']][e['field']]))
            e = decision['failure_evidence']
            matches.append(match_kind(e['quote'], supplied['failed_tests'][e['test_index']][e['field']]))
        except (KeyError, TypeError, IndexError):
            counts['invalid_schema'] += 1
            continue
        counts['quotes_match_after_format_normalization' if 'unmatched' not in matches else 'quotes_still_unmatched'] += 1
        counts.update('quote_' + m for m in matches)
        details.append({'trajectory_id': state['trajectory_id'], 'official_status': state['status'], 'quote_matches': matches})
    rows = list(csv.DictReader((OUT / 'analysis/paired_cases.csv').open()))
    valid_wrong = [r for r in rows if r['judge_status'] == 'valid' and r['causal-judge_hit'] == 'False']
    # This dataset's step IDs are chronological integers. No semantic cause labels are inferred here.
    offsets = Counter('later' if int(r['causal-judge_step']) > int(r['gold']) else 'earlier' for r in valid_wrong)
    data = {'official_failures': dict(reasons), 'diagnostic_quote_checks': dict(counts),
            'note': 'Format normalization is diagnostic only; all frozen decisions and fallback rules remain unchanged.',
            'valid_wrong': len(valid_wrong), 'valid_wrong_temporal_offsets': dict(offsets),
            'status_and_hits': dict(Counter(r['judge_status'] + ':' + r['causal-judge_hit'] for r in rows)),
            'per_case_quotes': details}
    (OUT / 'analysis/output_diagnostics.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in data.items() if k != 'per_case_quotes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
