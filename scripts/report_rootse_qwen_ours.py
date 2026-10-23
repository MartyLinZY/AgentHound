#!/usr/bin/env python3
"""Offline full-set evaluation of our frozen Qwen nonthinking experiment."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.rootse_evaluation import evaluate
from suite_transport import save, digest
from compare_rootse_models import paired_counts

read=lambda p:json.loads(p.read_text())
DEFAULT=ROOT/'outputs/rootse-ours-qwen38-nothinking-causal-v1'


def account(base: Path) -> dict:
    requests=0; logical=0; prompt=0; completion=0; total=0; unknown=0; reasoning=0; finishes=Counter()
    for p in base.rglob('calls/*/state.json'):
        s=read(p);req=read(p.parent/'request.json');logical+=1
        if digest(req)!=s['request_sha256'] or req.get('enable_thinking') is not False:
            raise ValueError('Request hash or thinking protocol mismatch')
        for a in s['attempts']:
            requests+=1;u=a.get('usage')
            if u is None:unknown+=1
            else:
                prompt+=u.get('prompt_tokens',0);completion+=u.get('completion_tokens',0);total+=u.get('total_tokens',0)
        for f in p.parent.glob('response-*.json'):
            d=read(f)
            for c in d.get('choices',[]):
                finishes[c.get('finish_reason')]+=1
                reasoning+=len(c.get('message',{}).get('reasoning_content') or '')
    return {'logical_calls':logical,'actual_requests':requests,'prompt_tokens':prompt,'completion_tokens':completion,
            'known_total_tokens':total,'unknown_usage_requests':unknown,'reasoning_characters':reasoning,'finish_reasons':dict(finishes)}


def interval(ids: list[str], values: dict[str,float], repos: dict[str,str]) -> list[float]:
    groups=defaultdict(list)
    for tid in ids:groups[repos[tid]].append(values[tid])
    names=sorted(groups);sums=np.array([sum(groups[k]) for k in names]);sizes=np.array([len(groups[k]) for k in names])
    samples=np.random.default_rng(20261012).integers(0,len(names),(5000,len(names)))
    v=sums[samples].sum(axis=1)/sizes[samples].sum(axis=1)*100
    return np.quantile(v,[.025,.975]).tolist()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);a=p.parse_args();out=a.out
    m=read(out/'manifest.json')
    if m['status']!='complete' or read(out/'judge/run_status.json')['n']!=102:
        raise ValueError('Require completed 102-case inference')
    for name,h in m['protocol']['code'].items():
        if hashlib.sha256((out/'snapshot'/name).read_bytes()).hexdigest()!=h:raise ValueError('Snapshot drift: '+name)
    for name,h in m['protocol']['config']['driver_hashes'].items():
        if hashlib.sha256((out/'snapshot'/name).read_bytes()).hexdigest()!=h:raise ValueError('Driver snapshot drift: '+name)
    for name,h in m['protocol']['dataset'].items():
        if hashlib.sha256((ROOT.parent/'TrajAudit'/name).read_bytes()).hexdigest()!=h:raise ValueError('Dataset drift: '+name)
    inputs,labels=load_rootse(ROOT.parent/'TrajAudit/RootSE',protocol='postmortem')
    ids=sorted(i.trajectory_id for i in inputs);repos={i.trajectory_id:i.repository or i.task_id for i in inputs}
    metrics={};records={};gold={l.trajectory_id:set(l.root_cause_steps) for l in labels}
    for method in ['baseline','anchors-global','anchors-backtrace','causal-judge']:
        folder=out/'rankings' if method=='causal-judge' else out/'retrievals'/method
        rs=[read(f) for f in folder.glob('*.json')];result=evaluate(inputs,labels,rs)
        result['tolerance_hits']=[sum(r['step_distance']<=k for r in result['per_trajectory']) for k in range(4)]
        records[method]={r['trajectory_id']:r for r in result['per_trajectory']}
        for row in rs:
            ordered=[r['source_step_id'] for r in row['ranking']]
            expected=next(n+1 for n,x in enumerate(ordered) if x in gold[row['trajectory_id']])
            ev=records[method][row['trajectory_id']]
            assert ev['top1_hit']==(expected==1) and abs(ev['reciprocal_rank']-1/expected)<1e-12
        result['repository_bootstrap_exact_95ci_pp']=interval(ids,{tid:float(records[method][tid]['top1_hit']) for tid in ids},repos)
        metrics[method]=result
    paired={}
    for method in ['anchors-backtrace','baseline']:
        l={tid:records[method][tid]['top1_hit'] for tid in ids};r={tid:records['causal-judge'][tid]['top1_hit'] for tid in ids}
        paired[method]={**paired_counts([l[t] for t in ids],[r[t] for t in ids]),
                        'delta_pp':sum(int(r[t])-int(l[t]) for t in ids)/102*100,
                        'repository_bootstrap_delta_95ci_pp':interval(ids,{t:int(r[t])-int(l[t]) for t in ids},repos)}
    running=0.0
    for k,(name,record) in enumerate(sorted(paired.items(),key=lambda kv:kv[1]['exact_mcnemar_two_sided_p'])):
        running=max(running,min(1,(2-k)*record['exact_mcnemar_two_sided_p']));record['holm_p']=running
    states=[read(f) for f in (out/'judge').glob('*/state.json')]
    if len(states)!=102:raise ValueError('Missing judge states')
    for s in states:
        cid=hashlib.sha256(s['trajectory_id'].encode()).hexdigest()[:16];r=read(out/'rankings'/f'{cid}.json')
        assert r['decision_fallback']==(s['status']!='valid')
        if s['status']=='valid':assert r['ranking'][0]['source_step_id']==s['decision']['step_id']
    costs={name:account(out/path) for name,path in [('graph','targets'),('graph_pilot','pilot'),('judge','judge'),('judge_pilot','judge_pilot')]}
    status=dict(Counter(s['status'] for s in states))
    data={'n':102,'model':m['protocol']['config']['model'],'enable_thinking':False,'n_independent_runs':1,
          'metrics':metrics,'paired_exploratory':paired,'judge_status':status,'costs':costs,
          'graph_status':read(out/'run_status.json'),'graph_fingerprint':m['fingerprint'],
          'unique_repositories':len(set(repos.values())),'unique_tasks':len({i.task_id for i in inputs})}
    save(out/'metrics_causal.json',data)
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    rows=[]
    for tid in ids:
        row={'trajectory_id':tid,'repository':repos[tid],'gold':';'.join(sorted(gold[tid]))}
        for method in records:
            row[method+'_step']=records[method][tid]['top1_step_id'];row[method+'_hit']=records[method][tid]['top1_hit']
        row['judge_status']=next(s['status'] for s in states if s['trajectory_id']==tid);rows.append(row)
    with (analysis/'paired_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    historical_cases=ROOT.parents[1]/'baseline_diagnosis_2026-10-12/shared_valid_failures.json'
    if historical_cases.exists():
        examples=[]
        for case in read(historical_cases):
            tid=case['trajectory_id'];s=next(s for s in states if s['trajectory_id']==tid)
            examples.append({**case,'same_graph_static_step':records['anchors-backtrace'][tid]['top1_step_id'],
                             'new_step':records['causal-judge'][tid]['top1_step_id'],
                             'new_exact':records['causal-judge'][tid]['top1_hit'],
                             'new_status':s['status'],'new_decision':s['decision']})
        save(analysis/'preselected_common_failures.json',examples)
    table=['| 方法 | Top-1 /102 | Tol@1 | Tol@2 | Tol@3 |','|---|---:|---:|---:|---:|']
    for name,res in metrics.items():table.append('| '+name+' | '+' | '.join(str(v) for v in res['tolerance_hits'])+' |')
    total=sum(c['known_total_tokens'] for c in costs.values());change=paired['anchors-backtrace'];new=metrics['causal-judge']['top1_hits'];old=metrics['anchors-backtrace']['top1_hits']
    comparison='增加' if new>=old else '减少'
    historical=read(ROOT/'outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1/metrics.json')
    text=f'''# Qwen 非思考模式：最早未恢复错误裁决

一句话改动：在依赖图排序后，结合失败测试和完整轨迹证据，选出最早引入且未被修正的决定性错误。

同一批 Qwen 新图上，旧锚点回溯 {old}/102，新裁决 {new}/102，{comparison} {abs(new-old)} 条（{change['delta_pp']:+.2f} 个百分点）。只报告这次运行的观测效果，不保证未见数据泛化。

{chr(10).join(table)}

全部图和裁决请求显式 enable_thinking=false。正式构图状态：{data['graph_status']['status_counts']}；裁决状态：{status}。无效裁决按预先冻结规则回退旧锚点排序，全部保留102分母。下游剩余排序沿用旧排名，Top-k不是模型独立提出多个候选的指标，因此主表仅比较Top-1及时间容差。

相对旧锚点，新增命中 {change['right_only']}，丢失命中 {change['left_only']}，共同命中 {change['both_correct']}。仓库配对bootstrap差值95%区间为 {change['repository_bootstrap_delta_95ci_pp']} 个百分点。McNemar/Holm仅作为配对独立假设下的探索性统计，详见附录；单次推理不能估计模型随机性，也没有独立未见测试集。

历史外部 Qwen thinking 实验 All-at-Once 为 {historical['metrics']['all_at_once']['tolerance_hits'][0]}/102，TrajAudit 为 {historical['metrics']['agent_locate']['tolerance_hits'][0]}/102。它们未修改、未重跑；thinking设置不同，不能用这两项声称公平胜过外部baseline。

## 资源与协议

正式构图：{costs['graph']}。

正式裁决：{costs['judge']}。

技术pilot独立记录：构图 {costs['graph_pilot']}；裁决 {costs['judge_pilot']}。

本轮全部已知tokens {total:,}，未知usage请求 {sum(c['unknown_usage_requests'] for c in costs.values())}；不把未知用量当零，不换算未核实价格。旧排序复用同一新图、无需额外模型请求；新方法的额外成本是一次完整轨迹裁决/案例。

## 可以支持与不能支持的结论

- 观测结论：上述固定102条中，新裁决相对同图旧排序的变化为 {change['delta_pp']:+.2f}pp。证据为原始请求、响应、逐例state和独立计分。
- 边界：改动同时加入完整轨迹的模型语义判断；尚未运行去图裁决消融，不能把增益归因于依赖图本身。RootSE已参与问题诊断及开发，不能把本轮当作未见测试集。
- 后续检查：用独立仓库划分和重复运行确认稳定性，并检验去图裁决，以判断依赖图是否值得其构建成本。不得据本轮标签继续修改提示词后覆盖结果。

冻结协议 {m['fingerprint']}。所有外部基线保持历史版本，本轮只新增我们的模块、入口、配置和结果。
'''
    (analysis/'analysis-report.md').write_text(text)
    (out/'EXPERIMENT_SUMMARY.md').write_text(text)
    (analysis/'stats-appendix.md').write_text('# 统计边界\n\n102条轨迹，'+str(data['unique_tasks'])+'任务、'+str(data['unique_repositories'])+'仓库，1次运行。不能提供跨运行mean±std。二元配对命中使用exact McNemar，无正态分布假设；两项对比用Holm。轨迹并非完全独立，因此p值仅探索性。仓库bootstrap 5000次、seed20261012，反映样本组成，不反映模型随机性。\n\n'+json.dumps(paired,ensure_ascii=False,indent=2))
    save(analysis/'integrity.json',{'complete_case_sets':True,'independent_rank_count_check':True,
         'all_requests_thinking_false':True,'response_reasoning_characters':sum(c['reasoning_characters'] for c in costs.values()),
         'snapshot_hashes_verified':True,'n_rankings':408})
    print(json.dumps({'hits':{k:v['top1_hits'] for k,v in metrics.items()},'judge_status':status,'paired':paired,'costs':costs},indent=2))

if __name__=='__main__':main()
