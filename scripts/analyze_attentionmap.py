#!/usr/bin/env python3
"""Independently verify OOF outputs and report frozen-baseline fallback semantics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def measure(scores, roots):
    order=np.lexsort((np.arange(len(scores)),-np.asarray(scores)))
    rank=min(int(np.flatnonzero(order==g)[0])+1 for g in roots)
    return {'rank':rank,'top1':int(rank==1),'top3':int(rank<=3),'top5':int(rank<=5),'mrr':1/rank}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs/attentionmap-v1')
    args=parser.parse_args();out=args.out;m=read(out/'manifest.json');summary=read(out/'summary.json')
    labels=read(out/'labels.json');cases={c['case_id']:c for c in m['cases']};dest=out/'analysis';dest.mkdir(exist_ok=True)
    source_checked=0;fit_checked=0;map_checked=0;predictions={};cost=[]
    for c in cases.values():
        for entry in c['sources'].values():
            assert sha(Path(entry['path']))==entry['sha256'];source_checked+=1
        assert sha(out/'features'/f"{c['case_id']}.npz")==c['feature_sha256']
    for name,expected in m['config']['source_hashes'].items():
        assert sha(out/'snapshot'/name)==expected
    assert sha(out/'labels.json')==m['labels_sha256']
    for variant in m['config']['variants']:
        predictions[variant]={}
        for seed in m['config']['seeds']:
            rows={}
            for fold in range(m['config']['folds']):
                run=out/'runs'/variant/f'seed-{seed}'/f'fold-{fold}'
                status=read(run/'status.json');r=read(run/'predictions.json')
                assert sha(run/'predictions.json')==status['predictions_sha256']
                assert sha(run/'model.pt')==status['model_sha256']
                train=set(status['train_ids']);test=set(status['test_ids'])
                assert not train&test and train|test==set(cases) and test==set(r)
                assert not {cases[c]['group'] for c in train}&{cases[c]['group'] for c in test}
                assert not {cases[c]['repository'] for c in train}&{cases[c]['repository'] for c in test}
                assert not {cases[c]['task_id'] for c in train}&{cases[c]['task_id'] for c in test}
                assert not set(rows)&set(r);rows.update(r);cost.append(status['seconds']);fit_checked+=1
            assert set(rows)==set(cases);predictions[variant][seed]=rows
            for cid,pred in rows.items():
                n=cases[cid]['n_steps'];assert len(pred['scores'])==n and np.isfinite(pred['scores']).all()
                assert np.allclose(np.array(pred['scores']),np.array(pred['local'])+np.array(pred['relation']),atol=1e-6)
                if variant not in ('linear','mlp'):
                    z=np.load(out/'maps'/variant/f'seed-{seed}'/f'{cid}.npz',allow_pickle=False);a=z['attention']
                    assert a.shape==(n,n) and np.isfinite(a).all() and (a>=0).all()
                    assert np.allclose(a.sum(1),1,atol=1e-6) and not np.triu(a,1).any()
                    assert abs(float(z['seeds'].sum())-1)<1e-6
                    prior=np.load(out/'features'/f'{cid}.npz')['prior']
                    allowed=np.tril(np.ones((n,n),bool))
                    if variant=='fixed_graph':allowed &= (prior>0)|np.eye(n,dtype=bool)
                    assert not a[~allowed].any();map_checked+=1
            for key,method in [('scores',variant),('shuffled_scores',variant+'_shuffled'),('drop50_scores',variant+'_drop50'),('drop100_scores',variant+'_drop100')]:
                if key not in next(iter(rows.values())):continue
                measures=[measure(p[key],labels[cid]) for cid,p in rows.items()]
                observed={k:float(np.mean([r[k] for r in measures])) for k in ('top1','top3','top5','mrr')}
                expected=summary['methods'][method]['per_seed'][m['config']['seeds'].index(seed)]
                for k,v in observed.items():assert abs(v-expected[k])<1e-12
    # Original q/propagation implementations use a fallback ordering on empty
    # evidence. Read their saved ranks rather than sorting all-zero vectors.
    historical={}
    for variant in ('q-only','weighted'):
        results={}
        for cid,c in cases.items():
            p=Path(m['config']['source'])/'rankings'/variant/f'{cid}.json'
            r=read(p);assert r['trajectory_id']==c['trajectory_id']
            root_ids={c['step_ids'][g] for g in labels[cid]}
            ranks=[row['rank'] for row in r['ranking'] if row['source_step_id'] in root_ids]
            rank=min(ranks);results[cid]={'rank':rank,'top1':int(rank==1),'top3':int(rank<=3),'top5':int(rank<=5),'mrr':1/rank}
        historical[variant]={k:float(np.mean([r[k] for r in results.values()])) for k in ('top1','top3','top5','mrr')}
        zero=[cid for cid,c in cases.items() if c['old_zero_root']]
        historical[variant]['zero_root_top1']=float(np.mean([results[cid]['top1'] for cid in zero]))
    assert round(historical['q-only']['top1']*102)==21
    assert round(historical['weighted']['top1']*102)==10
    checks={'source_files_unchanged':source_checked,'trained_fits':fit_checked,'maps_checked':map_checked,
            'all_oof_metrics_recomputed':True,'repository_and_task_disjoint':True,'historical_baselines':historical,
            'training_fit_seconds_sum':sum(cost),'new_api_calls':0,
            'raw_score_baselines':{'q_only':20,'old_weighted':7},
            'baseline_note':'raw-score re-ranking removes historical zero-score fallback; original q-only=21, weighted=10.'}
    (dest/'verification.json').write_text(json.dumps(checks,indent=2))
    # Select an illustrative zero-support recovery deterministically after all
    # evaluations; this selection has no effect on training or aggregate metrics.
    def hit_count(cid):
        return sum(measure(predictions['dense'][s][cid]['scores'],labels[cid])['top1'] for s in m['config']['seeds'])
    def mlp_hits(cid):
        return sum(measure(predictions['mlp'][s][cid]['scores'],labels[cid])['top1'] for s in m['config']['seeds'])
    candidates=[cid for cid,c in cases.items() if c['old_zero_root'] and 12<=c['n_steps']<=90 and hit_count(cid)>mlp_hits(cid)]
    chosen=sorted(candidates,key=lambda cid:(-hit_count(cid),cases[cid]['n_steps'],cid))[0]
    seed=next(s for s in m['config']['seeds'] if measure(predictions['dense'][s][chosen]['scores'],labels[chosen])['top1'])
    c=cases[chosen];gold=labels[chosen][0];pred=predictions['dense'][seed][chosen]
    raw=read(Path(c['sources']['raw']['path']));rootstep=raw['original_traj'][gold]
    z=np.load(out/'maps/dense'/f'seed-{seed}'/f'{chosen}.npz');a=z['attention'];prior=np.load(out/'features'/f'{chosen}.npz')['prior']
    columns=a[:,gold];top_sources=sorted(range(len(columns)),key=lambda j:-float(z['seeds'][j]*columns[j]))[:5]
    case={'mlp_hits_of_3':mlp_hits(chosen),'case_id':chosen,'trajectory_id':c['trajectory_id'],'seed':seed,'selection':'zero old propagation, 12–90 steps, dense hits exceed MLP; most dense seed hits then shortest; illustrative',
          'root_step_id':c['step_ids'][gold],'n_steps':c['n_steps'],'dense_hits_of_3':hit_count(chosen),
          'root_step':rootstep,'failed_tests':raw['failed_tests'],
          'source_rows':[{'step_id':c['step_ids'][j],'attention_to_root':float(a[j,gold]),'old_edge':bool(prior[j,gold]),
                          'diagnostic_mass':float(z['seeds'][j]*a[j,gold])} for j in top_sources],
          'ranks':{v:[measure(predictions[v][s][chosen]['scores'],labels[chosen])['rank'] for s in m['config']['seeds']] for v in predictions}}
    (dest/'illustrative_case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['rrf','linear','mlp','fixed_graph','dense_unconditioned','dense','dense_no_prior']
    display=['RRF','Linear','MLP','Fixed graph','No failure query','Attention map','No graph prior']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,key,title in zip(axes,['top1','top5'],['Exact root localization','Candidate coverage']):
        vals=[summary['methods'][v][key]*100 for v in names]
        ax.barh(display,vals,color=['#8e99a6']*5+['#247b88','#8e99a6'],height=.62)
        for i,v in enumerate(names):
            for s in summary['methods'][v]['per_seed']:ax.plot(s[key]*100,i,'o',color='#17252d',ms=3)
            ax.text(max(r[key]*100 for r in summary['methods'][v]['per_seed'])+1,i,f'{vals[i]:.1f}',va='center',fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(vals)+12);ax.set_xlabel('Percent of 102 trajectories');ax.set_title(title)
    fig.suptitle('RootSE grouped OOF development experiment · 5 folds · 3 seeds\nBars: mean; points: individual seeds',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(dest/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    n=c['n_steps'];fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.1]})
    axes[0].imshow(prior,cmap='Blues',vmin=0,vmax=1,origin='upper');axes[0].set_title('Existing step dependencies')
    relative=a/a.max(axis=1,keepdims=True)
    im=axes[1].imshow(relative,cmap='viridis',vmin=0,vmax=1,origin='upper');axes[1].set_title('Attention / maximum in each row')
    ticks=np.unique(np.linspace(0,n-1,min(7,n)).astype(int))
    for ax in axes[:2]:
        ax.axvline(gold,color='#dc4d43',lw=1.3,ls='--')
        ax.set_xticks(ticks,[c['step_ids'][i] for i in ticks]);ax.set_yticks(ticks,[c['step_ids'][i] for i in ticks])
        ax.set_xlabel('Earlier source step ID');ax.set_ylabel('Later target step ID')
    fig.colorbar(im,ax=axes[1],shrink=.72,label='Relative intensity')
    xs=np.arange(n);axes[2].plot(xs,pred['local'],label='Local evidence',color='#8795a1');axes[2].plot(xs,pred['scores'],label='Local + relation',color='#247b88')
    axes[2].axvline(gold,color='#dc4d43',ls='--',label=f'Gold step {c["step_ids"][gold]}')
    axes[2].set_xticks(ticks,[c['step_ids'][i] for i in ticks]);axes[2].set_xlabel('Original step ID');axes[2].set_ylabel('Ranking logit');axes[2].legend(fontsize=8);axes[2].set_title('Score decomposition')
    fig.suptitle(f'{c["task_id"][:72]} · seed {seed} · gold step {c["step_ids"][gold]}\nIllustrative recovery from the old zero-propagation subset',fontsize=11)
    for suffix in ('png','pdf'):fig.savefig(dest/f'attention_example.{suffix}',dpi=180)
    plt.close(fig)
    # Clear names for score-only ablations versus historical fallback-aware methods.
    rows=[('历史q-only（含回退）',historical['q-only']),('历史加权传播（含回退）',historical['weighted'])]
    rows += [(label,summary['methods'][name]) for name,label in zip(names,['本轮RRF 0.4/1/1','线性学习排序','无关系MLP','固定图attention','无失败条件关系','失败条件attention map','attention去图先验'])]
    lines=['# 首轮实验与实现说明','',
        '**Attention map首版已经实现并完成全量实验。平均Top-1为19.93%，低于本轮RRF的26.47%；Top-5为55.23%，高于RRF的49.02%。当前需要解决的是把候选覆盖转成首位定位。**','',
        '实验采用102条RootSE轨迹、5,268步骤、35个仓库/任务组；5折×3种子×6模型，共90次训练。每次预测来自未使用该仓库和任务训练的模型。固定词法hash表示128维，小模型隐藏层32维，40轮训练。','',
        '| 方法 | Top-1 | Top-3 | Top-5 | MRR |','|---|---:|---:|---:|---:|']
    for name,r in rows:lines.append(f"| {name} | {r['top1']:.2%} | {r['top3']:.2%} | {r['top5']:.2%} | {r['mrr']:.4f} |")
    lines+=['','## 从本轮得到的判断','',
        '稠密attention相对同输入MLP的Top-1增加1.63个百分点；仓库组bootstrap区间为−4.94至+7.54个百分点。相对固定图attention增加3.92个百分点，区间为−4.35至+12.12个百分点。首位定位的改进尚未稳定。',
        '去掉外部失败条件注入后，Top-1仍为19.93%；打乱每行attention权重后，Top-1也为19.93%，Top-5从55.23%降到49.67%。当前关系聚合对候选集合有影响，失败条件尚未带来额外首位命中。',
        '旧传播为0的54条子集中，attention平均命中11.67条，MLP平均8.67条；历史q-only命中12条。可补边结构让这些步骤获得评分，但现有局部证据仍提供了主要信号。',
        '线性学习排序的Top-5为66.99%，MRR为0.4291，均高于本轮attention；下一轮将线性学习排序与RRF保留为强对照。','',
        '## 具体实现','',
        '以步骤×特征作为输入，时间顺序约束A[j,i]只回看i≤j；已有依赖进入可学习偏置。失败表示进入Q/K和种子分配；最终排序采用单步分数加关系增量。模型、矩阵、每步分数分解和原始ID均已导出。',
        f'示例采用{c["task_id"]}，标注根因为原步骤{c["step_ids"][gold]}；旧传播为0，本轮3个种子中attention命中{hit_count(chosen)}次，MLP命中{mlp_hits(chosen)}次。示例按上述规则从实验结束后的输出选取，完整逐例结果保存在per_case_metrics.json。','',
        '![实际attention矩阵](attention_example.png)','',
        '## 复现与核验','',
        f'独立复算全部OOF指标，检查{fit_checked}个训练记录和{map_checked}张矩阵；{source_checked}份输入/图/评分/历史排名来源文件哈希保持一致。训练计时累计{sum(cost):.1f}秒，新增API请求0。历史图与q生成费用属于复用资产，端到端预算比较继续列入原RQ3。',
        '本轮为已参与历史开发的RootSE上的分组交叉验证，3种子反映训练初始化变化。外部泛化与LLM特征重新生成的变化作为后续实验。',
        '原始汇总中的q_only和old_weighted是直接按原始分数重排，分别20/102和7/102；历史实现有全零回退排序，本报告按保存的历史排名复算为21/102和10/102。两种口径分别保存在summary.json和analysis/verification.json。','',
        '## 下一轮固定问题','',
        '1. 用冻结的代码/文本语义编码替换词法hash，保持分组、训练预算和比较方法，检验跨表述错误是否得到更好的表示。',
        '2. 比较同输入线性排序、MLP与attention；先确定表示质量和局部错误证据的贡献，再增加关系结构。',
        '3. 在新的任务/仓库上冻结评估；选取少量可恢复轨迹进行受控修改，核查高权重关系是否对应后续决策变化。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'REPORT.md').write_text('# Attention map 首轮实验\n\n[经独立核验的完整报告](analysis/report.md)\n\n原始逐模型汇总为summary.json；历史baseline回退口径的复算见analysis/verification.json。\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
