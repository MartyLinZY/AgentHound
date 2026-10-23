#!/usr/bin/env python3
"""Freeze, run and rank a same-graph Qwen nonthinking influence experiment."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.schema import input_to_dict,input_from_dict
from loghound_agent_debug.anchors import extract_anchors
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rankers.decision_influence import VERSION,messages,parse_partial,pairs_from,propagate,rank,shuffled_weights
from suite_transport import RecordedClient,ExperimentAbort,save,digest,now

SOURCE=ROOT/'outputs/rootse-ours-qwen38-nothinking-causal-v1'
DEFAULT=ROOT/'outputs/rootse-influence-qwen38-nothinking-v3'
CONFIG=ROOT/'experiments/rootse_migration/config_qwen38_influence.json'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def batches(pairs: list[tuple[int,int]]) -> list[list[tuple[int,int]]]:
    targets=sorted({j for _,j in pairs})
    return [[pair for pair in pairs if pair[1] in set(targets[k:k+8])] for k in range(0,len(targets),8)]


def prepare(out: Path) -> dict:
    cfg=read(CONFIG)
    if cfg.get('enable_thinking') is not False or cfg.get('stream') is not True or cfg['model']!='qwen3.8-flash':
        raise ValueError('Nonthinking model protocol mismatch')
    if 'thinking' in cfg or 'reasoning_effort' in cfg: raise ValueError('Conflicting thinking flags')
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/suite_transport.py',CONFIG,
             SOURCE/'snapshot/configs/clue_rules.yaml']
    code={str(p.relative_to(ROOT)):sha(p) for p in files}
    graphs={p.name:sha(p) for p in sorted((SOURCE/'graphs').glob('*.json'))}
    historical={str(p.relative_to(SOURCE)):sha(p) for folder in ['retrievals/anchors-backtrace','rankings']
                for p in sorted((SOURCE/folder).glob('*.json'))}
    dataset={str(p.relative_to(ROOT.parent/'TrajAudit')):sha(p) for p in sorted((ROOT.parent/'TrajAudit/RootSE').rglob('*.json'))}
    protocol={'version':VERSION,'config':cfg,'code':code,'graphs':graphs,'historical':historical,'dataset':dataset,
              'source_graph_fingerprint':read(SOURCE/'manifest.json')['fingerprint']}
    fingerprint=digest(protocol)
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json')
        if m['fingerprint']!=fingerprint: raise ValueError('Frozen experiment drift; use a new version')
        return m
    out.mkdir(parents=True,exist_ok=True)
    for p in files:
        dest=out/'snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    items,_=load_rootse(ROOT.parent/'TrajAudit/RootSE',protocol='postmortem')
    cases=[];lengths=[]
    for item in items:
        cid=hashlib.sha256(item.trajectory_id.encode()).hexdigest()[:16]
        graph=read(SOURCE/'graphs'/f'{cid}.json');pairs=pairs_from(graph['dependencies'],len(item.steps))
        data={'item':input_to_dict(item),'pairs':pairs,'graph_sha256':graphs[f'{cid}.json']}
        save(out/'inputs'/f'{cid}.json',data)
        chars={'weights':max([sum(len(x['content']) for x in messages(item,b,'weights')) for b in batches(pairs)] or [0]),
               'nodes':sum(len(x['content']) for x in messages(item,pairs,'nodes'))}
        if max(chars.values())>cfg['max_input_chars']: raise ValueError('Input too large')
        case={'case_id':cid,'trajectory_id':item.trajectory_id,'n_steps':len(item.steps),'n_edges':len(pairs),
              'input_sha256':digest(data),'message_chars':chars,'weight_batches':len(batches(pairs))}
        cases.append(case);lengths.append((max(chars.values()),cid))
    lengths.sort();pilot_ids=[lengths[k][1] for k in [0,len(lengths)//2,len(lengths)-1]]
    m={'status':'prepared','fingerprint':fingerprint,'protocol':protocol,'cases':cases,'pilot_ids':pilot_ids,
       'n':len(cases),'n_edges':sum(c['n_edges'] for c in cases),'created_at':now()}
    assert m['n']==102 and len(graphs)==102
    save(out/'manifest.json',m);return m


def case_call(item, pairs, phase: str, out: Path, cfg: dict, key: str, client_factory=RecordedClient) -> dict:
    msg=messages(item,pairs,phase);signature=digest({'messages':msg,'config':cfg,'version':VERSION})
    path=out/'state.json'
    if path.exists():
        old=read(path)
        if old['signature']!=signature: raise ValueError('Cached request drift')
        if old['status']!='running': return old
    s={'trajectory_id':item.trajectory_id,'phase':phase,'signature':signature,'status':'running','rows':None,'started_at':now()}
    save(path,s)
    if phase=='weights' and not pairs:
        s.update(status='valid',rows=[],no_call_empty_graph=True)
    else:
        try:
            answer=client_factory(out,cfg,key).create(model=cfg['model'],messages=msg)
            checked=parse_partial(answer.choices[0].message.content,item,pairs,phase)
            s.update(**checked,status='partial' if checked['missing_ids'] or checked['rejected_rows'] else 'valid')
        except ExperimentAbort as exc:
            if exc.status in {'model_mismatch','replay_mismatch','call_limit'}: raise RuntimeError(str(exc)) from exc
            s.update(status=exc.status,failure_reason=str(exc))
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            s.update(status='format_error',failure_reason=f'{type(exc).__name__}: {str(exc)[:300]}')
    s['finished_at']=now();save(path,s);return s


def run(m: dict,out: Path,key_path: Path,pilot: bool=False) -> dict:
    cfg=m['protocol']['config'];key=key_path.read_text().strip()
    if not key: raise ValueError('Empty key')
    cases=[c for c in m['cases'] if not pilot or c['case_id'] in m['pilot_ids']]
    jobs=[(c,'nodes',None) for c in cases]+[(c,'weights',b) for c in cases for b in range(c['weight_batches'])]
    counts=Counter();coverage=Counter();base=out/'pilot' if pilot else out/'inference'
    def job(c,p,b):
        d=read(out/'inputs'/f"{c['case_id']}.json")
        if digest(d)!=c['input_sha256']: raise ValueError('Frozen input drift')
        pairs=[tuple(x) for x in d['pairs']];path=base/p/c['case_id']
        if b is not None: pairs=batches(pairs)[b];path=path/f'b{b:04d}'
        return case_call(input_from_dict(d['item']),pairs,p,path,cfg,key)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures=[pool.submit(job,c,p,b) for c,p,b in jobs]
        for n,f in enumerate(as_completed(futures),1):
            s=f.result();counts[s['phase']+':'+s['status']]+=1
            coverage[s['phase']+':valid_rows']+=len(s.get('rows') or [])
            coverage[s['phase']+':expected']+=s.get('expected_count',0)
            progress={'stage':'pilot' if pilot else 'inference','seen':n,'planned':len(jobs),'counts':dict(counts),'updated_at':now()}
            save(base/'progress.json',progress);print(json.dumps(progress),flush=True)
    result={'status':'complete','n_jobs':len(jobs),'counts':dict(counts),'finished_at':now()}
    result['row_coverage']=dict(coverage)
    if pilot:
        result['gate_passed']=(all(k.endswith((':valid',':partial')) for k in counts)
            and all(coverage[p+':expected']>0 and coverage[p+':valid_rows']/coverage[p+':expected']>=.95 for p in ['weights','nodes']))
    save(base/'run_status.json',result);return result


def rank_all(m: dict,out: Path) -> None:
    if read(out/'inference/run_status.json')['n_jobs']!=sum(1+c['weight_batches'] for c in m['cases']): raise ValueError('Incomplete inference')
    rules=load_clue_rules(out/'snapshot'/str((SOURCE/'snapshot/configs/clue_rules.yaml').relative_to(ROOT)))
    for c in m['cases']:
        cid=c['case_id'];d=read(out/'inputs'/f'{cid}.json');item=input_from_dict(d['item']);pairs=[tuple(x) for x in d['pairs']]
        n=len(item.steps);base=read(SOURCE/'retrievals/anchors-backtrace'/f'{cid}.json')
        ns=read(out/'inference/nodes'/cid/'state.json');byedge={};batch_status=Counter()
        for b,bp in enumerate(batches(pairs)):
            state=read(out/'inference/weights'/cid/f'b{b:04d}'/'state.json');batch_status[state['status']]+=1
            byedge.update({f'{i}>{j}':.5 for i,j in bp})
            if state['status'] in {'valid','partial'}: byedge.update({r['id']:r['strength'] for r in state['rows']})
        ws={'status':'valid' if sum(v for k,v in batch_status.items() if k!='valid')==0 else 'partial','batch_status':dict(batch_status)}
        w=[byedge[f'{i}>{j}'] for i,j in pairs]
        if ns['status'] in {'valid','partial'}:
            byid={r['step_id']:r for r in ns['rows']}
            q=[(byid[s.source_step_id]['score'] if byid[s.source_step_id]['kind']=='introduced' and not byid[s.source_step_id]['recovered'] else 0.0)
               if s.source_step_id in byid else 1.0 for s in item.steps]
        else: q=[1.0]*n
        evidence=extract_anchors(item,rules,'structured');f=list(evidence.seeds)
        total=sum(f);f=[v/total for v in f] if total else f
        mean=sum(w)/len(w) if w else .5
        equal=propagate(n,pairs,[mean]*len(w),f)
        weighted=propagate(n,pairs,w,f)
        fixed=propagate(n,pairs,[.1]*len(w),f)
        shuffled=propagate(n,pairs,shuffled_weights(w,item.trajectory_id,20261013),f)
        scores={'equal':equal,'weighted':weighted,'fixed-point1':fixed,'shuffled':shuffled,
                'q-only':q,'q-equal':[a*b for a,b in zip(q,equal)],'q-weighted':[a*b for a,b in zip(q,weighted)],
                'q-shuffled':[a*b for a,b in zip(q,shuffled)]}
        for name,v in scores.items():
            fallback=None
            if name=='q-only' and ns['status'] not in {'valid','partial'}: fallback='node_assessment_failed_use_legacy'
            elif not any(v): fallback='zero_score_use_legacy'
            r=rank(base,item,v,name,fallback);r['influence_status']={'weights':ws['status'],'nodes':ns['status']}
            save(out/'rankings'/name/f'{cid}.json',r)
        save(out/'features'/f'{cid}.json',{'trajectory_id':item.trajectory_id,'pairs':pairs,'weights':w,'q':q,'seeds':f,
             'weights_status':ws['status'],'weight_batch_status':ws['batch_status'],'nodes_status':ns['status'],'uniform_mean':mean,'n_steps':n})
    m.update(status='complete',finished_at=now());save(out/'manifest.json',m)


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--stage',choices=['prepare','pilot','run','rank','all'],default='all');p.add_argument('--api-key-file',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=prepare(a.out)
        if a.stage=='prepare': print(json.dumps({k:m[k] for k in ['fingerprint','n','n_edges','pilot_ids']}));return 0
        if a.stage in {'pilot','run','all'} and a.api_key_file is None: p.error('--api-key-file required')
        if a.stage in {'pilot','all'}:
            if not run(m,a.out,a.api_key_file,True)['gate_passed']: return 2
        if a.stage in {'run','all'}:
            if not read(a.out/'pilot/run_status.json').get('gate_passed'): raise ValueError('Pilot must pass')
            run(m,a.out,a.api_key_file)
        if a.stage in {'rank','all'}: rank_all(m,a.out)
    return 0


if __name__=='__main__': raise SystemExit(main())
