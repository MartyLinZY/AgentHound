#!/usr/bin/env python3
"""Audit all v5 results and keep primary matched contrasts separate from historical comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import csv

from run_contextmap_v5 import OUTPUTS, PARENTS, METHODS
from run_contextmap import read, sha
from run_contextmap_balanced import check_manifest
from analyze_contextmap_controls import verify_call
from analyze_contextmap_balanced import complete, describe, paired, family_correction
from analyze_contextmap import account
from suite_transport import save, now


def analyze(dataset: str) -> dict:
    out, parent = OUTPUTS[dataset], PARENTS[dataset]
    manifest, previous = check_manifest(out), check_manifest(parent)
    finished = read(out/'run_status.json')
    if finished['completed'] != manifest['n']*2 or finished['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Incomplete experiment')
    for filename in ('historical_hashes.json', 'parent_run_hashes.json'):
        for path, checksum in read(out/filename).items():
            from pathlib import Path
            if sha(Path(path)) != checksum:
                raise ValueError('Historical run changed')
    labels = read(out/'labels.json')
    if labels != read(parent/'labels.json'):
        raise ValueError('Labels differ from parent')
    cases = {c['case_id']: c for c in manifest['cases']}
    rows, checked, hashes = {}, Counter(), {}
    for method in ['initial']+METHODS[dataset]:
        rows[method] = {}
        origin = parent if method=='initial' else out
        source_method = 'balanced' if method=='initial' else method
        frozen = previous if method=='initial' else manifest
        for cid, meta in cases.items():
            packet = read(origin/'prepared'/source_method/f'{cid}.json')
            directory = origin/'runs'/source_method/cid
            state, call, audit = verify_call(directory, packet, frozen)
            if method != 'initial':
                checked.update(audit)
                for path in directory.rglob('*'):
                    if path.is_file():
                        hashes[str(path.relative_to(out))] = sha(path)
            ranking = state['decision']['ranked_step_ids'] if state['status']=='valid' else []
            gold = labels[cid][0]
            rank = ranking.index(gold)+1 if gold in ranking else None
            offset = int(ranking[0])-int(gold) if ranking else None
            case = read(out/'cases'/f'{cid}.json')
            step, view = case['steps'][int(gold)], packet['view']
            row = {'case_id':cid, 'method':method, 'dataset':dataset, 'group':meta['group'],
                'subset':meta.get('subset','rootse'), 'n_steps':meta['n_steps'], 'gold':gold,
                'status':state['status'], 'ranked_step_ids':ranking, 'rank':rank,
                'top1':int(rank==1), 'top5':int(rank is not None), 'mrr_at5':1/rank if rank else 0.,
                'late':int(offset is not None and offset>0), 'early':int(offset is not None and offset<0),
                'root_expanded':int(gold in view['selected_ids']),
                'root_has_decision':int(bool(step['thought'] or step['response'])),
                'root_complete_decision':int(bool(step['thought'] or step['response']) and
                    all(complete(view,step,f) for f in ('thought','response'))),
                'root_complete_action':int(bool(step['action']) and complete(view,step,'action')),
                'root_complete_observation':int(bool(step['observation']) and complete(view,step,'observation')),
                'selected_nodes':len(view['selected_ids']), 'context_tokens_proxy':view['context_tokens'],
                'seconds':sum(a.get('elapsed_seconds',0) for a in call['attempts']),
                'reason':state.get('decision',{}).get('reason')}
            rows[method][cid] = row
    actual = Counter(r['status'] for m in METHODS[dataset] for r in rows[m].values())
    if dict(actual) != finished['status_counts']:
        raise ValueError('Completion counts differ')
    summaries = {}
    first_cost = account(parent/'runs/balanced')
    for method in rows:
        cost = first_cost if method=='initial' else account(out/'runs'/method)
        summaries[method] = {**describe(list(rows[method].values()),cases), 'recorded_call_cost':cost,
            **{field:sum(r[field] for r in rows[method].values())
               for field in ('root_complete_action','root_complete_observation')}}
        if dataset=='rootse' and method!='initial':
            summaries[method]['deployment_two_stage_cost'] = dict(Counter(first_cost)+Counter(cost))
    control, treatment = METHODS[dataset]
    contrasts = {k:paired(rows,treatment,control,k,cases) for k in ('top1','top5')}
    historical = {m:{k:paired(rows,m,'initial',k,cases) for k in ('top1','top5')} for m in METHODS[dataset]}
    selections = {'all':list(cases), 'common_valid':[c for c in cases if all(rows[m][c]['status']=='valid' for m in METHODS[dataset])]}
    if dataset=='rootse':
        selections.update({'initial_top1':[c for c in cases if rows['initial'][c]['rank']==1],
            'initial_rank2_to5':[c for c in cases if rows['initial'][c]['top5'] and not rows['initial'][c]['top1']],
            'initial_miss':[c for c in cases if not rows['initial'][c]['top5']]})
    else:
        sizes={c:read(parent/'prepared/full'/f'{c}.json')['view']['context_tokens'] for c in cases}
        selections.update({'full_fits_16k':[c for c in cases if sizes[c]<=16384],
                           'full_exceeds_16k':[c for c in cases if sizes[c]>16384]})
        same = [c for c in cases if read(out/'prepared'/control/f'{c}.json')['messages_sha256']==
                                   read(out/'prepared'/treatment/f'{c}.json')['messages_sha256']]
        flagged = {c for pair in read(parent/'dataset_audit.json')['identical_history_different_question_groups'] for c in pair}
        selections.update({'same_request':same, 'changed_request':[c for c in cases if c not in same],
                           'quality_without_reused_histories':[c for c in cases if c not in flagged]})
        for subset in sorted({c['subset'] for c in cases.values()}):
            selections[subset]=[c for c in cases if cases[c]['subset']==subset]
    strata={name:{m:describe([rows[m][c] for c in ids],cases) for m in rows}
            for name,ids in selections.items() if ids}
    mechanism={}
    for field in ('root_expanded','root_complete_decision','root_complete_action','root_complete_observation'):
        mechanism[field]={}
        for direction,predicate in (('gained',lambda d:d>0),('lost',lambda d:d<0),('unchanged',lambda d:d==0)):
            ids=[c for c in cases if predicate(rows[treatment][c][field]-rows[control][c][field])]
            mechanism[field][direction]={'ids':ids,'n':len(ids),
                'top1_treatment':sum(rows[treatment][c]['top1'] for c in ids),
                'top1_control':sum(rows[control][c]['top1'] for c in ids),
                'top5_treatment':sum(rows[treatment][c]['top5'] for c in ids),
                'top5_control':sum(rows[control][c]['top5'] for c in ids)}
    result={'dataset':dataset,'n':len(cases),'fingerprint':manifest['fingerprint'],
        'methods':summaries,'contrasts':contrasts,'historical_descriptive':historical,
        'strata':strata,'mechanism_diagnostics':mechanism,'verification':dict(checked),
        'new_run_cost':account(out/'runs'),'finished_at':now()}
    if dataset=='whowhen':
        result['identical_request_variation']={'n':len(same),
            'top1_disagreements':sum(rows[control][c]['ranked_step_ids'][:1]!=rows[treatment][c]['ranked_step_ids'][:1] for c in same),
            'ranking_disagreements':sum(rows[control][c]['ranked_step_ids']!=rows[treatment][c]['ranked_step_ids'] for c in same)}
    save(out/'analysis/summary.json',result)
    save(out/'analysis/predictions.json',rows)
    save(out/'analysis/raw_hashes.json',hashes)
    flat=[r for method in rows.values() for r in method.values()]
    with (out/'analysis/cases.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(dataset,{m:{k:s[k] for k in ('n','top1','top5','root_expanded','root_complete_action')} for m,s in summaries.items()},flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('rootse','whowhen','both'),default='both')
    args=parser.parse_args()
    results={d:analyze(d) for d in OUTPUTS if args.dataset in (d,'both')}
    if len(results)==2:
        adjusted=family_correction(results)
        for out in OUTPUTS.values():
            save(out/'analysis/primary_family_holm.json',adjusted)


if __name__=='__main__':
    main()
