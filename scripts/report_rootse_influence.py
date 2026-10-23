#!/usr/bin/env python3
"""Evaluate the frozen influence run only after all formal inference is complete."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from run_rootse_influence import ROOT,SOURCE,DEFAULT,read,sha
from suite_transport import save,digest
from compare_rootse_models import paired_counts
from report_rootse_qwen_ours import account
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.rankers.decision_influence import propagate,shuffled_weights
from loghound_agent_debug.rootse_evaluation import evaluate

METHODS=['legacy','previous-judge','fixed-point1','equal','weighted','shuffled','q-only','q-equal','q-weighted','q-shuffled']
NAMES={'legacy':'历史锚点PageRank','previous-judge':'上一轮整体裁决','fixed-point1':'固定0.1 DAG',
       'equal':'均值等权 DAG','weighted':'影响加权 DAG','shuffled':'打乱边权 DAG（固定seed）',
       'q-only':'仅错误引入评分 q','q-equal':'q × 均值等权','q-weighted':'q × 影响加权','q-shuffled':'q × 打乱边权'}


def ci(ids: list[str],values: dict[str,float],repos: dict[str,str]) -> list[float]:
    groups=defaultdict(list)
    for tid in ids:groups[repos[tid]].append(values[tid])
    names=sorted(groups);sums=np.array([sum(groups[n]) for n in names]);sizes=np.array([len(groups[n]) for n in names])
    sample=np.random.default_rng(20261013).integers(0,len(names),(5000,len(names)))
    v=sums[sample].sum(1)/sizes[sample].sum(1)*100
    return np.quantile(v,[.025,.975]).tolist()


def main() -> None:
    out=DEFAULT;m=read(out/'manifest.json')
    if m['status']!='complete' or read(out/'inference/run_status.json')['n_jobs']!=627: raise ValueError('Full run not complete')
    for name,h in m['protocol']['code'].items():
        if sha(out/'snapshot'/name)!=h or sha(ROOT/name)!=h: raise ValueError('Code/config drift: '+name)
    for name,h in m['protocol']['dataset'].items():
        if sha(ROOT.parent/'TrajAudit'/name)!=h: raise ValueError('Dataset drift')
    for name,h in m['protocol']['graphs'].items():
        if sha(SOURCE/'graphs'/name)!=h: raise ValueError('Graph drift')
    for name,h in m['protocol']['historical'].items():
        if sha(SOURCE/name)!=h: raise ValueError('Historical ranking drift')
    items,labels=load_rootse(ROOT.parent/'TrajAudit/RootSE',protocol='postmortem')
    ids=sorted(i.trajectory_id for i in items);byitem={i.trajectory_id:i for i in items}
    gold={l.trajectory_id:set(l.root_cause_steps) for l in labels};repos={i.trajectory_id:i.repository or i.task_id for i in items}
    metrics={};records={};rankings={}
    for method in METHODS:
        folder=SOURCE/'retrievals/anchors-backtrace' if method=='legacy' else (SOURCE/'rankings' if method=='previous-judge' else out/'rankings'/method)
        rs=[read(p) for p in sorted(folder.glob('*.json'))];ev=evaluate(items,labels,rs)
        ev['tolerance_hits']=[sum(r['step_distance']<=k for r in ev['per_trajectory']) for k in range(4)]
        records[method]={r['trajectory_id']:r for r in ev['per_trajectory']};rankings[method]={r['trajectory_id']:r for r in rs}
        for tid,r in rankings[method].items():
            ordered=[x['source_step_id'] for x in r['ranking']];first=next(i+1 for i,v in enumerate(ordered) if v in gold[tid])
            assert records[method][tid]['top1_hit']==(first==1)
            assert abs(records[method][tid]['reciprocal_rank']-1/first)<1e-12
        ev['repository_bootstrap_95ci_pp']=ci(ids,{t:float(records[method][t]['top1_hit']) for t in ids},repos)
        metrics[method]=ev
    contrasts={}
    for left,right in [('equal','weighted'),('q-equal','q-weighted'),('q-only','q-weighted')]:
        l=[records[left][t]['top1_hit'] for t in ids];r=[records[right][t]['top1_hit'] for t in ids]
        contrasts[f'{right}_vs_{left}']={**paired_counts(l,r),'left':left,'right':right,
            'delta_pp':100*(sum(r)-sum(l))/102,'repository_delta_95ci_pp':ci(ids,{t:int(b)-int(a) for t,a,b in zip(ids,l,r)},repos)}
    running=0
    for i,(name,d) in enumerate(sorted(contrasts.items(),key=lambda kv:kv[1]['exact_mcnemar_two_sided_p'])):
        running=max(running,min(1,(3-i)*d['exact_mcnemar_two_sided_p']));d['holm_p']=running
    features={read(p)['trajectory_id']:read(p) for p in (out/'features').glob('*.json')}
    assert set(features)==set(ids)
    weight_values=[];zero_seed=[];zero_q=[];cases_status=Counter();tie_counts=Counter();offsets={};gold_coverage=Counter()
    for tid,feat in features.items():
        n=feat['n_steps'];pairs=[tuple(p) for p in feat['pairs']];w=feat['weights'];f=feat['seeds'];q=feat['q']
        graph=read(SOURCE/'graphs'/f"{hashlib.sha256(tid.encode()).hexdigest()[:16]}.json")
        assert pairs==sorted({(int(d['source_ordinal']),int(d['target_ordinal'])) for d in graph['dependencies']})
        matrix=np.zeros((n,n))
        for (i,j),v in zip(pairs,w):matrix[i,j]=v
        matrix/=np.maximum(1,matrix.sum(0))
        independent=np.linalg.solve(np.eye(n)-matrix,np.array(f))
        assert np.allclose(independent,propagate(n,pairs,w,f),rtol=1e-10,atol=1e-12)
        roots=[s.ordinal for s in byitem[tid].steps if s.source_step_id in gold[tid]]
        equal_scores=propagate(n,pairs,[feat['uniform_mean']]*len(w),f)
        gold_coverage['positive_equal_propagation']+=any(equal_scores[i]>0 for i in roots)
        gold_coverage['positive_weighted_propagation']+=any(independent[i]>0 for i in roots)
        gold_coverage['positive_q']+=any(q[i]>0 for i in roots)
        gold_coverage['positive_q_weighted']+=any(q[i]*independent[i]>0 for i in roots)
        gold_coverage['q_only_correct_but_weighted_root_zero']+=(records['q-only'][tid]['top1_hit'] and not any(independent[i]>0 for i in roots))
        for method in ['weighted','q-weighted']:
            expected=independent if method=='weighted' else independent*np.array(q)
            r=rankings[method][tid]
            for row in r['ranking']:
                idx=next(s.ordinal for s in byitem[tid].steps if s.source_step_id==row['source_step_id'])
                assert abs(row['components']['decision_influence']-expected[idx])<1e-10
            if any(expected):tie_counts[method]+=int(sum(abs(x-max(expected))<1e-12 for x in expected)>1)
        weight_values+=w
        if not any(f):zero_seed.append(tid)
        if not any(q):zero_q.append(tid)
        cases_status['weights:'+feat['weights_status']]+=1;cases_status['nodes:'+feat['nodes_status']]+=1
    for method in METHODS:
        counter=Counter()
        for tid in ids:
            ords={s.source_step_id:s.ordinal for s in byitem[tid].steps}
            delta=ords[records[method][tid]['top1_step_id']]-min(ords[g] for g in gold[tid])
            counter['exact' if delta==0 else ('later' if delta>0 else 'earlier')]+=1
        offsets[method]=dict(counter)
    completeness_sensitivity={}
    for name,selected in {
        'complete_weights':[t for t in ids if features[t]['weights_status']=='valid'],
        'complete_nodes':[t for t in ids if features[t]['nodes_status']=='valid'],
        'complete_both':[t for t in ids if features[t]['weights_status']=='valid' and features[t]['nodes_status']=='valid'],
    }.items():
        completeness_sensitivity[name]={'n':len(selected),'hits':{method:sum(records[method][t]['top1_hit'] for t in selected) for method in METHODS}}
    permutation=[]
    for seed in range(20261013,20261113):
        hits=Counter()
        for tid in ids:
            feat=features[tid];s=propagate(feat['n_steps'],[tuple(p) for p in feat['pairs']],shuffled_weights(feat['weights'],tid,seed),feat['seeds'])
            for name,values in [('weighted',s),('q-weighted',[a*b for a,b in zip(s,feat['q'])])]:
                step=byitem[tid].steps[max(range(len(s)),key=lambda i:values[i])].source_step_id if any(values) else records['legacy'][tid]['top1_step_id']
                hits[name]+=step in gold[tid]
                if seed==20261013:
                    assert step==records['shuffled' if name=='weighted' else 'q-shuffled'][tid]['top1_step_id']
        permutation.append({'seed':seed,**hits})
    permutation_summary={}
    for method in ['weighted','q-weighted']:
        values=np.array([r[method] for r in permutation]);observed=metrics[method]['top1_hits']
        permutation_summary[method]={'observed_hits':observed,'shuffle_mean_hits':float(values.mean()),
            'shuffle_std_hits':float(values.std(ddof=1)),'shuffle_min_max_hits':[int(values.min()),int(values.max())],
            'conditional_fraction_shuffles_at_least_observed':float((values>=observed).mean()),'n_shuffles':100}
    costs={phase:account(out/'inference'/phase) for phase in ['weights','nodes']}
    costs['pilot_v2_reused_for_v3']=account(out/'pilot');costs['rejected_pilot_v1']=account(ROOT/'outputs/rootse-influence-qwen38-nothinking-v1/pilot')
    statuses={};failure_reasons={};coverage={};evidence_fields={}
    for phase in ['weights','nodes']:
        paths=(out/'inference'/phase).glob('*/*/state.json' if phase=='weights' else '*/state.json')
        ss=[read(p) for p in paths];statuses[phase]=dict(Counter(s['status'] for s in ss))
        failure_reasons[phase]=dict(Counter(s.get('failure_reason','partial rows: see missing_ids/rejected_rows') for s in ss if s['status']!='valid'))
        expected=m['n_edges'] if phase=='weights' else sum(c['n_steps'] for c in m['cases'])
        coverage[phase]={'valid_rows':sum(len(s.get('rows') or []) for s in ss),'expected_rows':expected,
            'fallback_rows':expected-sum(len(s.get('rows') or []) for s in ss),
            'rejected_row_reasons':dict(Counter(r['reason'] for s in ss for r in s.get('rejected_rows',[])))}
        if phase=='weights':
            evidence_fields={name:dict(Counter(r[name].split(':',1)[1] for s in ss for r in (s.get('rows') or [])))
                             for name in ['source_ref','target_ref']}
    preserved=read(out/'preservation_before.json');changed=[name for name,h in preserved.items() if not Path(name).exists() or sha(Path(name))!=h]
    if changed: raise ValueError('Protected files changed: '+str(changed[:5]))
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    topology=read(out/'topology_diagnostics.json')
    result={'n':102,'n_model_runs':1,'enable_thinking':False,'fingerprint':m['fingerprint'],'metrics':metrics,'topology':topology,
            'contrasts':contrasts,'permutation_summary':permutation_summary,'permutations':permutation,'costs':costs,
            'statuses':statuses,'failure_reasons':failure_reasons,'coverage':coverage,'evidence_fields':evidence_fields,'case_statuses':dict(cases_status),'temporal_offsets':offsets,
            'completeness_sensitivity':completeness_sensitivity,'gold_score_coverage':dict(gold_coverage),
            'weight_histogram':dict(Counter(str(w) for w in weight_values)),'n_edges':len(weight_values),
            'zero_seed_cases':zero_seed,'zero_q_cases':zero_q,'top_score_tie_cases':dict(tie_counts),
            'fallback_counts':{method:sum(bool(r.get('influence_fallback')) for r in rankings[method].values()) for method in METHODS},
            'unique_repositories':len(set(repos.values())),'preservation':{'checked_files':len(preserved),'changed':changed}}
    save(out/'metrics.json',result)
    rows=[]
    for tid in ids:
        row={'trajectory_id':tid,'repository':repos[tid],'gold':';'.join(sorted(gold[tid]))}
        for method in METHODS:row[method+'_step']=records[method][tid]['top1_step_id'];row[method+'_hit']=records[method][tid]['top1_hit']
        row.update(weights_status=features[tid]['weights_status'],nodes_status=features[tid]['nodes_status']);rows.append(row)
    with (analysis/'paired_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    previous=read(SOURCE/'analysis/preselected_common_failures.json')
    save(analysis/'preselected_common_cases.json',[{'trajectory_id':c['trajectory_id'],'gold':c['gold'],
         'predictions':{method:records[method][c['trajectory_id']]['top1_step_id'] for method in METHODS}} for c in previous])
    table=['| 方法 | Top1 /102 | Tol@1 | Tol@2 | MRR |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        ev=metrics[method];table.append(f"| {NAMES[method]} | {ev['top1_hits']} | {ev['tolerance_hits'][1]} | {ev['tolerance_hits'][2]} | {ev['mrr']:.4f} |")
    primary=contrasts['weighted_vs_equal'];secondary=contrasts['q-weighted_vs_q-equal'];qonly=contrasts['q-weighted_vs_q-only']
    maintext=f'''# 决策影响度传播实验

一句话：在固定依赖图上估计每条边的直接决策影响度，按权重传播失败信号，并独立检验错误引入评分的作用。

主结果：均值等权 {metrics['equal']['top1_hits']}/102，影响加权 {metrics['weighted']['top1_hits']}/102；净变化 {primary['delta_pp']:+.2f}pp，新增命中{primary['right_only']}、丢失{primary['left_only']}。这是同图、同锚点、同传播公式、同总边权量下的比较。

{chr(10).join(table)}

q加权相对q等权变化 {secondary['delta_pp']:+.2f}pp；q加权相对仅q评分变化 {qonly['delta_pp']:+.2f}pp。若q-only不低于q-weighted，不能把节点语义判断的收益归因给影响传播。历史7/102锚点PageRank与19/102整体裁决是原记录，与本轮DAG传播公式不同，只作参考。

## 权重是否有信息

100次预定打乱结果：{json.dumps(permutation_summary,ensure_ascii=False)}。

打乱保留每个案例的拓扑与权重多重集合，但改变权重分配，含入边总量变化。此分布只度量本次固定输出在随机分配下的敏感性，不是100次模型独立重复，也不是严格因果显著性检验。不能挑选其中最好的一次作为方法成绩。

固定图的结构限制：5268步骤中1484没有入边，3188只有一个前驱；有入边节点中单前驱占{topology['single_parent_fraction_among_linked']*100:.2f}%。因此本实验的加权作用很大一部分是改变链上传播衰减，多来源相互比较的空间有限。这是候选图结构统计，不是真实依赖召回率，也不能预先证明效果受它决定。

事后机制诊断（金标仅用于评价，未进入预测）：{dict(gold_coverage)}。其中positive表示至少一个标注根因的对应分量>0；零传播分量在乘法融合时无法由正q恢复。该统计帮助区分覆盖与排序问题，不用于修改权重或筛选案例。

## 执行与成本

正式状态：{statuses}。条目覆盖：{coverage}。案例状态：{dict(cases_status)}。无锚点案例{len(zero_seed)}，q全零案例{len(zero_q)}；回退按冻结规则执行，所有102例计分。正式权重共{len(weight_values)}条；参考字段ID存在只说明可定位，不验证语义判断正确。

边权引用字段分布：{evidence_fields}。当前节点仍是原轨迹步骤（含决策、动作、观察），没有进一步拆分成原子决策；若引用目标观察而非选择本身，更不能将该条目自动解释为已经验证的决策影响。

完整输出子集的描述性敏感性检查：{json.dumps(completeness_sensitivity,ensure_ascii=False)}。这些是事前按输出完整性定义的条件子集，存在选择偏差，不替代102分母主结果，不外推格式修复后的成绩。

API记录：{json.dumps(costs,ensure_ascii=False)}。

全部请求显式thinking=false，响应reasoning字符数总和{sum(c['reasoning_characters'] for c in costs.values())}。历史构图成本是复用成本，未重复调用；本轮新增成本分权重、节点、v2技术pilot及未通过的v1技术pilot，未知usage不当作零。

v1只完成6次技术pilot：五档以外的连续值、增漏边和int/string ID导致未通过门槛。v2改为局部8目标批次、0–1连续值和无歧义整数ID规范化，14次pilot仍有少量漏项/引用失败。v1/v2均未开始正式运行、未计算定位指标。v3在看金标命中前固定逐条严格校验与中性回退：缺失边=.5，缺失节点q=1，引用验证规则未放宽、不修JSON、不新增边或重试格式。v3重放v2的相同14个API记录验证每阶段>=95%条目覆盖，再运行正式627逻辑请求。导入的pilot只计一次成本，没有重新生成预测。不能将技术试跑描述为三次正式模型实验。

## 统计、证据与边界

102轨迹、{len(set(repos.values()))}仓库、1次模型运行。仓库bootstrap 5000次seed20261013。主对照提升区间 {primary['repository_delta_95ci_pp']}pp；三组exact McNemar+Holm仅为探索性，轨迹相关性与模型随机性限制另见附录。RootSE已参与开发，本轮不提供未见泛化证据。

当前影响是局部轨迹片段上的模型估计，独立替代信息可能不在片段中；没有真实干预和环境重放。图拓扑与旧锚点错误也固定保留。既有图是否漏边、失败锚点是否错误连接，不会被单独改边权解决。缺失节点q=1仅为乘法中性回退，并非确信该节点错误；这会影响q-only的排序，需要结合覆盖率和完整输出子集诊断，不能选择性删样本。q全零/失败回退、平分选较早步骤都可能影响精确定位，不能把这些机制解释成因果发现。

以独立线性方程求解逐例复查DAG传播，逐排名复算命中/MRR；验证快照、数据、固定图和历史排名哈希。受保护历史文件{len(preserved)}个，变化0；外部基线未改。推理源码冻结指纹 {m['fingerprint']}。

## Claim Candidates

- Claim：本轮加权相对均值等权变化{primary['delta_pp']:+.2f}pp。
  - Source evidence：metrics.json、paired_cases.csv、原始inference状态与响应。
  - Allowed wording：本次固定102条上的观测结果。
  - Forbidden stronger wording：决策影响度已被证明是因果效应，或在未见数据上必然提高。
  - Uncertainty：单轮、开发集、代理边权、固定不完整图和锚点。
  - Next check：结合打乱、q-only和逐例增损判断是否值得继续；有收益时再做独立数据和真实干预校准。
  - Decision：保留数值描述，机制结论需结合消融。
'''
    (analysis/'analysis-report.md').write_text(maintext);(out/'EXPERIMENT_SUMMARY.md').write_text(maintext)
    (analysis/'stats-appendix.md').write_text('# 统计附录\n\n二元Top1无需正态假设；exact McNemar要求独立配对单位，但同仓库轨迹相关，因此仅探索性。三项事前比较Holm校正。仓库bootstrap重采样仓库并保留组内轨迹，5000次seed20261013，区间反映样本组成，不估计模型重复运行波动。仅一次运行，不能报告跨运行mean±std。权重打乱的mean±std是条件随机分配统计，不是模型稳定性。\n\n'+json.dumps(contrasts,ensure_ascii=False,indent=2))
    save(analysis/'integrity.json',{'snapshot_dataset_graph_hashes':True,'independent_linear_solve':True,
         'independent_rank_counts':True,'all_requests_thinking_false':True,'protected_files':len(preserved),'changed_files':changed})
    print(json.dumps({'hits':{k:v['top1_hits'] for k,v in metrics.items()},'contrasts':contrasts,'permutation':permutation_summary,'statuses':statuses,'costs':costs},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
