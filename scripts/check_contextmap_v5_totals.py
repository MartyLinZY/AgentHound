#!/usr/bin/env python3
"""Recount v5 saved decisions, cost, strata and frozen hashes independently."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parents[2]/'output/attentionmap_v5_evidence_2026-10-30'
OUTPUTS = {d:ROOT/f'outputs/attentionmap-v5-{d}-20261030' for d in ('rootse','whowhen')}
PARENTS = {'rootse':ROOT/'outputs/attentionmap-context-v4-balanced-20261029',
           'whowhen':ROOT/'outputs/attentionmap-whowhen-v4-20261029'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recount(dataset):
    out,parent=OUTPUTS[dataset],PARENTS[dataset]
    manifest,summary=read(out/'manifest.json'),read(out/'analysis/summary.json')
    labels=read(out/'labels.json')
    assert labels==read(parent/'labels.json')
    hashes={**manifest['source_hashes'],**manifest['prepared_hashes'],
            **read(out/'analysis_code_before_results.json')['code_hashes'],
            **read(out/'historical_hashes.json'),**read(out/'parent_run_hashes.json')}
    for name,expected in hashes.items():
        path=Path(name) if Path(name).is_absolute() else out/name
        assert sha(path)==expected, name
    encoder=tiktoken.get_encoding('cl100k_base')
    metrics,rows={},{}
    for method in ['initial']+manifest['config']['methods']:
        source=parent if method=='initial' else out
        source_method='balanced' if method=='initial' else method
        totals,statuses,usage=Counter(),Counter(),Counter()
        rows[method]={}
        reciprocal=[]
        for meta in manifest['cases']:
            cid=meta['case_id'];directory=source/'runs'/source_method/cid
            state=read(directory/'state.json')
            assert state['status']!='running'
            statuses[state['status']]+=1
            ranking=state['decision']['ranked_step_ids'] if state['status']=='valid' else []
            assert len(set(ranking))==len(ranking)<=5
            assert all(0<=int(s)<meta['n_steps'] for s in ranking)
            assert len(labels[cid])==1
            gold=labels[cid][0];rank=ranking.index(gold)+1 if gold in ranking else None
            row={'top1':int(rank==1),'top5':int(rank is not None),'rank':rank,
                 'mrr_at5':1/rank if rank else 0.,'ranking':ranking,'status':state['status']}
            rows[method][cid]=row
            totals.update({k:row[k] for k in ('top1','top5')});reciprocal.append(row['mrr_at5'])
            packet=read(source/'prepared'/source_method/f'{cid}.json')
            size=len(encoder.encode(packet['view']['content'],disallowed_special=()))
            assert size==packet['view']['context_tokens']<=16384
            call=read(directory/'calls/0001/state.json');usage['logical_calls']+=1
            for attempt in call['attempts']:
                usage['requests']+=1
                if attempt.get('usage') is None:
                    usage['unknown_usage_requests']+=1
                else:
                    for field in ('prompt_tokens','completion_tokens','total_tokens'):
                        usage[field]+=attempt['usage'].get(field,0)
                if attempt['status']=='received':
                    response=read(directory/'calls/0001'/f"response-{attempt['number']}.json")
                    assert response['usage']==attempt['usage']
                    sse=directory/'calls/0001'/f"response-{attempt['number']}.sse"
                    recorded=None
                    for line in sse.read_text().splitlines():
                        if line.startswith('data:') and line[5:].strip()!='[DONE]':
                            chunk=json.loads(line[5:].strip())
                            if chunk.get('usage') is not None:recorded=chunk['usage']
                    assert recorded==attempt['usage']
        expected=summary['methods'][method];n=len(rows[method])
        assert n==expected['n'] and dict(statuses)==expected['status_counts']
        assert all(totals[k]==expected[k] for k in ('top1','top5'))
        assert math.isclose(sum(reciprocal)/n,expected['metrics']['mrr_at5']['mean'],abs_tol=1e-12)
        assert dict(usage)==expected['recorded_call_cost']
        metrics[method]={'n':n,**dict(totals),'status_counts':dict(statuses),'usage':dict(usage)}
        if dataset=='rootse' and method!='initial':
            cost=Counter(metrics['initial']['usage'])+usage
            assert dict(cost)==expected['deployment_two_stage_cost']
    control,treatment=manifest['config']['methods']
    for metric,c in summary['contrasts'].items():
        gained=[cid for cid in rows[control] if rows[treatment][cid][metric]>rows[control][cid][metric]]
        lost=[cid for cid in rows[control] if rows[treatment][cid][metric]<rows[control][cid][metric]]
        assert gained==c['gained'] and lost==c['lost']
        assert math.isclose((len(gained)-len(lost))/manifest['n'],c['mean'],abs_tol=1e-12)
    strata={}
    if dataset=='rootse':
        strata={'initial_top1':[c for c,r in rows['initial'].items() if r['rank']==1],
                'initial_rank2_to5':[c for c,r in rows['initial'].items() if r['top5'] and not r['top1']],
                'initial_miss':[c for c,r in rows['initial'].items() if not r['top5']]}
    else:
        sizes={c:len(encoder.encode(read(parent/'prepared/full'/f'{c}.json')['view']['content'],disallowed_special=())) for c in labels}
        same=[c for c in labels if read(out/'prepared'/control/f'{c}.json')['messages']==
                                  read(out/'prepared'/treatment/f'{c}.json')['messages']]
        strata={'full_fits_16k':[c for c,s in sizes.items() if s<=16384],
                'full_exceeds_16k':[c for c,s in sizes.items() if s>16384],
                'same_request':same,'changed_request':[c for c in labels if c not in same]}
        variation=summary['identical_request_variation']
        assert variation=={'n':len(same),
            'top1_disagreements':sum(rows[control][c]['ranking'][:1]!=rows[treatment][c]['ranking'][:1] for c in same),
            'ranking_disagreements':sum(rows[control][c]['ranking']!=rows[treatment][c]['ranking'] for c in same)}
    for name,ids in strata.items():
        for method in rows:
            assert summary['strata'][name][method]['n']==len(ids)
            for metric in ('top1','top5'):
                assert sum(rows[method][c][metric] for c in ids)==summary['strata'][name][method][metric]
    return {'methods':metrics,'hashes_checked':len(hashes),'independent_strata_checked':list(strata)}


def main():
    results={d:recount(d) for d in OUTPUTS}
    adjusted=read(OUTPUTS['rootse']/'analysis/primary_family_holm.json')
    assert adjusted==read(OUTPUTS['whowhen']/'analysis/primary_family_holm.json')
    assert len(adjusted)==4
    previous=0.
    for i,(_,contrast) in enumerate(sorted(adjusted.items(),key=lambda v:v[1]['p'])):
        previous=max(previous,min(1.,(4-i)*contrast['p']))
        assert previous==contrast['p_holm']
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'methods':results,
            'source_sha256':sha(Path(__file__)),'primary_holm_tests':4,'network_calls':0}
    (DEST/'final_independent_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
