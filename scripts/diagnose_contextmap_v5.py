#!/usr/bin/env python3
"""Export every gain/loss and reproducible case packets after full v5 completion."""
from collections import Counter

from run_contextmap_v5 import ROOT, OUTPUTS, PARENTS, METHODS
from run_contextmap import read
from suite_transport import save, now

DEST=ROOT.parents[2]/'output/attentionmap_v5_evidence_2026-10-30'


def analyze(dataset):
    out,parent=OUTPUTS[dataset],PARENTS[dataset]
    manifest=read(out/'manifest.json')
    assert read(out/'run_status.json')['completed']==manifest['n']*2
    predictions=read(out/'analysis/predictions.json')
    summary=read(out/'analysis/summary.json')
    control,treatment=METHODS[dataset]
    transitions={metric:{direction:summary['contrasts'][metric][direction]
                        for direction in ('gained','lost')} for metric in ('top1','top5')}
    selected={}
    for metric,directions in transitions.items():
        for direction,ids in directions.items():
            if dataset=='whowhen':
                ids=[c for c in ids if read(out/'prepared'/control/f'{c}.json')['messages_sha256']!=
                                       read(out/'prepared'/treatment/f'{c}.json')['messages_sha256']]
            if ids:selected.setdefault(sorted(ids)[0],[]).append(metric+'_'+direction)
    if dataset=='rootse':
        for cid in ('0e21497e7d7da5b4','eb0f47d2882627f1','51115bda0199fe42','8be468d283c47a23'):
            if cid in predictions['initial']:selected.setdefault(cid,[]).append('v4_named_case')
    destination=DEST/'case-studies'/dataset
    destination.mkdir(parents=True,exist_ok=True)
    details=[]
    for cid,tags in selected.items():
        case=read(out/'cases'/f'{cid}.json')
        gold=predictions[control][cid]['gold']
        packets={}
        for method in predictions:
            source=parent if method=='initial' else out
            sm='balanced' if method=='initial' else method
            view=read(source/'prepared'/sm/f'{cid}.json')['view']
            packets[method]={'result':predictions[method][cid],
                             'root_fragments':[f for f in view['fragments'] if f['step_id']==gold],
                             'selected_ids':view['selected_ids']}
        ids=sorted({gold,*[p['result']['ranked_step_ids'][0] for p in packets.values()
                          if p['result']['ranked_step_ids']]},key=int)
        packet={'case_id':cid,'selection_tags':tags,'task':case['task'],'test_code':case['test_code'],
                'error':case['error'],'gold':gold,'steps':{i:case['steps'][int(i)] for i in ids},'methods':packets}
        save(destination/f'{cid}.json',packet)
        lines=[f'# {dataset} / {cid}','','选择原因：'+', '.join(tags),'',f'标注根因：{gold}','',
               '| 条件 | Top-5原始步骤号 | 根因排名 | 完整决定/动作/输出 |','|---|---|---:|---|']
        for m,p in packets.items():
            r=p['result'];complete='/'.join(str(r[k]) for k in ('root_complete_decision','root_complete_action','root_complete_observation'))
            lines.append(f"| {m} | {r['ranked_step_ids']} | {r['rank']} | {complete} |")
        lines+=['','任务（展示前4000字符，全文与完整证据在同名JSON）：','',case['task'][:4000]]
        for i in ids:
            lines+=['',f'## 原始步骤 {i}']
            for field in ('thought','response','action','observation'):
                value=case['steps'][int(i)][field]
                if value:lines+=['',f'{field}（{len(value)}字符；展示前4000）：','', '```text',value[:4000],'```']
        for method,p in packets.items():lines+=['',f'## {method} 模型解释','',str(p['result'].get('reason') or '')]
        (destination/f'{cid}.md').write_text('\n'.join(lines)+'\n')
        details.append({'case_id':cid,'tags':tags,'ranks':{m:p['result']['rank'] for m,p in packets.items()}})
    directions={}
    for m,rs in predictions.items():
        errors=[r for r in rs.values() if not r['top1']]
        misses=[r for r in rs.values() if not r['top5']]
        directions[m]={'top1_errors':len(errors),'late_errors':sum(r['late'] for r in errors),
            'early_errors':sum(r['early'] for r in errors),'top5_misses':len(misses),
            'misses_root_expanded':sum(r['root_expanded'] for r in misses),
            'misses_complete_decision':sum(r['root_complete_decision'] for r in misses),
            'misses_complete_action':sum(r['root_complete_action'] for r in misses),
            'misses_complete_observation':sum(r['root_complete_observation'] for r in misses)}
    result={'time':now(),'dataset':dataset,'transitions':transitions,'error_directions':directions,
            'case_packets':details,'selection_rule':'lexicographic first gain and loss for each metric; Who&When requires changed input; RootSE adds four cases named in v4 analysis',
            'scope':'post-result descriptive diagnosis; no inference changes or further model calls'}
    save(DEST/dataset/'case_diagnostics.json',result)
    print(dataset,details,flush=True)


if __name__=='__main__':
    for dataset in OUTPUTS:analyze(dataset)
