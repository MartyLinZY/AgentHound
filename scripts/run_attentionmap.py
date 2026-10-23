#!/usr/bin/env python3
"""Freeze and run grouped out-of-fold attention experiments using existing caches."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.features.attention_inputs import build_inputs, grouped_folds
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker, VARIANTS, VERSION, listwise_loss
from loghound_agent_debug.rankers.loghound_static import load_config, rank_trajectory

SOURCE = ROOT/'outputs/rootse-influence-qwen38-nothinking-v3'
GRAPH = ROOT/'outputs/rootse-ours-qwen38-nothinking-causal-v1'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data: object) -> str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(out: Path, epochs: int, seeds: list[int], folds: int) -> dict:
    source_files=[Path(__file__),ROOT/'src/loghound_agent_debug/features/attention_inputs.py',
                  ROOT/'src/loghound_agent_debug/rankers/attention_map.py',
                  ROOT/'src/loghound_agent_debug/rankers/loghound_static.py',
                  ROOT/'src/loghound_agent_debug/adapters/rootse.py',
                  ROOT/'src/loghound_agent_debug/schema.py',ROOT/'src/loghound_agent_debug/graph.py',
                  ROOT/'src/loghound_agent_debug/evidence.py',
                  ROOT/'src/loghound_agent_debug/features/semantic.py',
                  ROOT/'src/loghound_agent_debug/features/anchor.py',
                  ROOT/'configs/default.yaml',ROOT/'configs/clue_rules.yaml']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    inputs,labels=load_rootse(ROOT.parent/'TrajAudit/RootSE',protocol='postmortem')
    fold_ids,groups=grouped_folds(inputs,folds)
    cfg={'version':VERSION,'epochs':epochs,'seeds':seeds,'folds':folds,'hidden':32,
         'lr':.001,'weight_decay':.001,'batch_size':8,'variants':list(VARIANTS),
         'source':str(SOURCE),'graph':str(GRAPH),'source_hashes':hashes,
         'device':'cpu','threads':1,'api_calls':0,'evaluation':'repository/task-grouped OOF development experiment'}
    records=[]
    for item,lab,fold,group in zip(inputs,labels,fold_ids,groups):
        cid=hashlib.sha256(item.trajectory_id.encode()).hexdigest()[:16]
        files={'raw':next(p for p in (ROOT.parent/'TrajAudit/RootSE').rglob('*.json')
                          if str(p.relative_to(ROOT.parent/'TrajAudit/RootSE')).removesuffix('.json')==item.trajectory_id.removeprefix('rootse/')),
               'features':SOURCE/'features'/f'{cid}.json',
               'nodes':SOURCE/'inference/nodes'/cid/'state.json',
               'graph':GRAPH/'graphs'/f'{cid}.json',
               'old_weighted':SOURCE/'rankings/weighted'/f'{cid}.json'}
        records.append({'case_id':cid,'trajectory_id':item.trajectory_id,'step_ids':[s.source_step_id for s in item.steps],
                        'fold':fold,'group':group,'task_id':item.task_id,'repository':item.repository,
                        'n_steps':len(item.steps),'sources':{k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()}})
    fingerprint=digest({'config':cfg,'cases':records})
    if (out/'manifest.json').exists():
        old=read(out/'manifest.json')
        if old['fingerprint']!=fingerprint:raise ValueError('Frozen experiment changed; use a new output directory')
        for r in old['cases']:
            if sha(out/'features'/f"{r['case_id']}.npz")!=r['feature_sha256']:raise ValueError('Feature drift')
        if sha(out/'labels.json')!=old['labels_sha256']:raise ValueError('Label drift')
        return old
    rules=load_clue_rules(ROOT/'configs/clue_rules.yaml');rank_cfg=load_config(ROOT/'configs/default.yaml')
    label_data={};baseline={}
    for item,label,record in zip(inputs,labels,records):
        cid=record['case_id'];sources=record['sources'];feature=read(Path(sources['features']['path']))
        graph=read(Path(sources['graph']['path']));nodes=read(Path(sources['nodes']['path']))
        if feature['trajectory_id']!=item.trajectory_id or len(feature['q'])!=len(item.steps):
            raise ValueError('Cache identity mismatch')
        ranked=rank_trajectory(item,rank_cfg,rules,graph['dependencies'])
        byid={r['source_step_id']:r for r in ranked['ranking']}
        valid_ids={r['step_id'] for r in nodes.get('rows') or []}
        numeric=np.array([[byid[s.source_step_id]['components']['semantic_normalized'],
                           byid[s.source_step_id]['components']['anchor']/(1+byid[s.source_step_id]['components']['anchor']),
                           byid[s.source_step_id]['components']['clue'],feature['q'][i],float(s.source_step_id not in valid_ids)]
                          for i,s in enumerate(item.steps)],dtype=np.float32)
        data=build_inputs(item,[tuple(p) for p in feature['pairs']],numeric)
        dest=out/'features'/f'{cid}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,x=data.x,failure=data.failure,prior=data.prior)
        record['feature_sha256']=sha(dest)
        gold=[record['step_ids'].index(s) for s in label.root_cause_steps]
        label_data[cid]=gold
        weighted=read(Path(sources['old_weighted']['path']))
        old_scores={r['source_step_id']:r['components']['decision_influence'] for r in weighted['ranking']}
        record['old_zero_root']=all(old_scores[label.root_cause_steps[i]]==0 for i in range(len(gold)))
        baseline[cid]={'rrf':[byid[s.source_step_id]['score_raw'] for s in item.steps],'q_only':feature['q'],
                       'old_weighted':[old_scores[s.source_step_id] for s in item.steps]}
    save(out/'labels.json',label_data);save(out/'baselines.json',baseline)
    for p in source_files:
        dest=out/'snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    manifest={'fingerprint':fingerprint,'config':cfg,'cases':records,'labels_sha256':sha(out/'labels.json'),
              'baselines_sha256':sha(out/'baselines.json'),'n':len(inputs),'groups':len(set(groups)),
              'n_steps':sum(len(x.steps) for x in inputs),'status':'prepared','created_at':time.time()}
    save(out/'manifest.json',manifest)
    return manifest


def order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)),key=lambda i:(-scores[i],i))


def metric(scores: list[float], gold: list[int]) -> dict:
    rank=min(order(scores).index(g)+1 for g in gold)
    return {'rank':rank,'top1':float(rank==1),'top3':float(rank<=3),'top5':float(rank<=5),'mrr':1/rank}


def train(manifest: dict, out: Path) -> None:
    cfg=manifest['config'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    labels=read(out/'labels.json');cases=manifest['cases'];data={}
    for c in cases:
        z=np.load(out/'features'/f"{c['case_id']}.npz",allow_pickle=False)
        data[c['case_id']]=tuple(torch.from_numpy(z[k]) for k in ('x','failure','prior'))
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            for fold in range(cfg['folds']):
                run=out/'runs'/variant/f'seed-{seed}'/f'fold-{fold}'
                if (run/'status.json').exists():
                    status=read(run/'status.json')
                    if status['fingerprint']!=manifest['fingerprint']:raise ValueError('Run fingerprint mismatch')
                    if sha(run/'predictions.json')!=status['predictions_sha256']:raise ValueError('Predictions drift')
                    continue
                start=time.monotonic();torch.manual_seed(seed*100+fold)
                sample=next(iter(data.values()));model=AttentionMapRanker(sample[0].shape[1],hidden=cfg['hidden'],variant=variant)
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                training=[c for c in cases if c['fold']!=fold];testing=[c for c in cases if c['fold']==fold]
                if {c['group'] for c in training}&{c['group'] for c in testing}:raise ValueError('Group leakage')
                rng=random.Random(seed*100+fold);losses=[]
                for epoch in range(cfg['epochs']):
                    model.train();indices=list(range(len(training)));rng.shuffle(indices);total=0.0
                    for offset in range(0,len(indices),cfg['batch_size']):
                        batch=indices[offset:offset+cfg['batch_size']];optimizer.zero_grad()
                        for idx in batch:
                            cid=training[idx]['case_id'];scores=model(*data[cid])['scores']
                            loss=listwise_loss(scores,labels[cid]);total+=float(loss.detach())
                            (loss/len(batch)).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                    losses.append(total/len(training))
                model.eval();predictions={};run.mkdir(parents=True,exist_ok=True)
                with torch.no_grad():
                    for c in testing:
                        cid=c['case_id'];x,f,e=data[cid];pred=model(x,f,e)
                        record={k:v.tolist() for k,v in pred.items() if k not in {'attention'}}
                        if 'attention' in pred:
                            maps=out/'maps'/variant/f'seed-{seed}';maps.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(maps/f'{cid}.npz',attention=pred['attention'].numpy(),
                                                seeds=pred['seeds'].numpy(),mass=pred['mass'].numpy())
                            record['shuffled_scores']=model(x,f,e,shuffle_seed=int(cid[:8],16)+seed)['scores'].tolist()
                            generator=torch.Generator().manual_seed(int(cid[:8],16))
                            half=e*(torch.rand(e.shape,generator=generator)>=.5)
                            record['drop50_scores']=model(x,f,half)['scores'].tolist()
                            record['drop100_scores']=model(x,f,torch.zeros_like(e))['scores'].tolist()
                        predictions[cid]=record
                save(run/'predictions.json',predictions)
                torch.save({'state_dict':model.state_dict(),'input_dim':sample[0].shape[1],
                            'variant':variant,'hidden':cfg['hidden'],'fingerprint':manifest['fingerprint']},run/'model.pt')
                save(run/'status.json',{'fingerprint':manifest['fingerprint'],'train_ids':[c['case_id'] for c in training],
                    'test_ids':[c['case_id'] for c in testing],'losses':losses,'seconds':time.monotonic()-start,
                    'parameters_total':sum(p.numel() for p in model.parameters()),
                    'parameters_with_gradient':sum(p.numel() for p in model.parameters() if p.grad is not None),
                    'model_sha256':sha(run/'model.pt'),'predictions_sha256':sha(run/'predictions.json'),'status':'complete'})
                print(json.dumps({'variant':variant,'seed':seed,'fold':fold,'seconds':round(time.monotonic()-start,1)}),flush=True)
    manifest['status']='complete';save(out/'manifest.json',manifest)


def report(manifest: dict, out: Path) -> dict:
    cfg=manifest['config'];labels=read(out/'labels.json');cases={c['case_id']:c for c in manifest['cases']}
    if sha(out/'baselines.json')!=manifest['baselines_sha256']:raise ValueError('Baseline drift')
    all_metrics={};individual=[]
    baseline=read(out/'baselines.json')
    for name in ('rrf','q_only','old_weighted'):
        all_metrics[name]=[{cid:metric(v[name],labels[cid]) for cid,v in baseline.items()}]
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            combined={}
            for fold in range(cfg['folds']):
                run=out/'runs'/variant/f'seed-{seed}'/f'fold-{fold}'
                status=read(run/'status.json');pred=read(run/'predictions.json')
                if sha(run/'predictions.json')!=status['predictions_sha256']:raise ValueError('Prediction drift')
                if set(combined)&set(pred):raise ValueError('Duplicate OOF prediction')
                combined.update(pred)
            if set(combined)!=set(cases):raise ValueError('Incomplete OOF predictions')
            for mode in ('scores','shuffled_scores','drop50_scores','drop100_scores'):
                if mode not in next(iter(combined.values())):continue
                name=variant if mode=='scores' else variant+'_'+mode.removesuffix('_scores')
                measured={cid:metric(row[mode],labels[cid]) for cid,row in combined.items()}
                all_metrics.setdefault(name,[]).append(measured)
                for cid,m in measured.items():individual.append({'method':name,'seed':seed,'case_id':cid,**m})
    summaries={}
    for name,replicates in all_metrics.items():
        per_seed=[{k:float(np.mean([row[k] for row in r.values()])) for k in ('top1','top3','top5','mrr')} for r in replicates]
        zero=[cid for cid,c in cases.items() if c['old_zero_root']]
        summaries[name]={'per_seed':per_seed,**{k:float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]},
                          'zero_root_n':len(zero),'zero_root_top1':float(np.mean([r[cid]['top1'] for r in replicates for cid in zero]))}
    groups=sorted({c['group'] for c in cases.values()});comparisons={}
    dense={cid:np.mean([r[cid]['top1'] for r in all_metrics['dense']]) for cid in cases}
    for other in ('mlp','fixed_graph','dense_unconditioned','dense_no_prior','dense_shuffled','rrf'):
        ref={cid:np.mean([r[cid]['top1'] for r in all_metrics[other]]) for cid in cases}
        delta={g:[dense[cid]-ref[cid] for cid,c in cases.items() if c['group']==g] for g in groups}
        rng=np.random.default_rng(20261023);samples=[]
        for _ in range(2000):
            selected=rng.choice(groups,len(groups),replace=True)
            samples.append(float(np.mean([v for g in selected for v in delta[g]])))
        comparisons[other]={'top1_delta':float(np.mean([dense[c]-ref[c] for c in cases])),
                            'group_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist(),
                            'mean_gained':float(np.mean([sum(r[c]['top1']>b[c]['top1'] for c in cases)
                                                         for r,b in zip(all_metrics['dense'],all_metrics[other]*len(all_metrics['dense']))])),
                            'mean_lost':float(np.mean([sum(r[c]['top1']<b[c]['top1'] for c in cases)
                                                       for r,b in zip(all_metrics['dense'],all_metrics[other]*len(all_metrics['dense']))]))}
    summary={'n':len(cases),'groups':len(groups),'n_steps':manifest['n_steps'],'seeds':cfg['seeds'],
             'methods':summaries,'dense_comparisons':comparisons,'new_api_calls':0,'fingerprint':manifest['fingerprint']}
    save(out/'summary.json',summary);save(out/'per_case_metrics.json',individual)
    lines=['# Attention map 首轮结果','',f"102条轨迹，{len(groups)}个仓库/任务组，{cfg['folds']}折，{len(cfg['seeds'])}个种子。分组交叉验证开发实验。",'',
           '| 方法 | Top-1 | Top-3 | Top-5 | MRR | 旧传播为零子集Top-1 |','|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['top1']:.2%} | {s['top3']:.2%} | {s['top5']:.2%} | {s['mrr']:.4f} | {s['zero_root_top1']:.2%} |")
    lines+=['','所有方法逐例OOF预测、种子结果、模型权重、输入/代码哈希、关系矩阵均保存在本目录。',
            'rrf为师弟融合实现接入本轮固定Qwen图和首条失败测试的结果；与历史汇报使用不同构图缓存。',
            '新API请求0；训练使用既有LLM构图和q缓存，历史特征生成成本另计。',
            '文本表示为128维固定signed hashing，训练32维小模型。全102条已参与历史方法开发。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/attentionmap-v1')
    p.add_argument('--stage',choices=('prepare','train','report','all'),default='all')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--folds',type=int,default=5);a=p.parse_args()
    if a.epochs<1 or len(set(a.seeds))!=len(a.seeds):p.error('Positive epochs and unique seeds required')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=prepare(a.out,a.epochs,a.seeds,a.folds)
        print(json.dumps({'status':m['status'],'n':m['n'],'groups':m['groups'],'fingerprint':m['fingerprint']}),flush=True)
        if a.stage in {'train','all'}:train(m,a.out)
        if a.stage in {'report','all'}:report(m,a.out)


if __name__=='__main__':main()
