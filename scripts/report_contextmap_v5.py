#!/usr/bin/env python3
"""Package complete v5 diagnostics, clustered statistics and figures."""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_contextmap_v5 import OUTPUTS, PARENTS, ROOT
from run_contextmap import read, sha
from suite_transport import save, now

DEST=ROOT.parents[2]/'output/attentionmap_v5_evidence_2026-10-30'
NAMES={'initial':'v4历史单轮','review_same':'第二轮原上下文','review_evidence':'第二轮候选证据',
       'balanced_repeat':'v4同期重跑','coverage':'交接片段与图覆盖'}
ENGLISH={'initial':'v4 historical','review_same':'Same-context review','review_evidence':'Evidence review',
         'balanced_repeat':'v4 repeat','coverage':'Episode / graph coverage'}


def table(methods: dict) -> str:
    lines=['| 条件 | Top-1 | Top-5 | MRR@5 |','|---|---:|---:|---:|']
    for m,s in methods.items():
        lines.append(f"| {NAMES[m]} | {s['top1']}/{s['n']} | {s['top5']}/{s['n']} | {s['metrics']['mrr_at5']['mean']:.4f} |")
    return '\n'.join(lines)


def figures(results: dict) -> None:
    dest=DEST/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.6),constrained_layout=True)
    for row,(d,r) in enumerate(results.items()):
        methods=list(r['methods'])
        for col,k in enumerate(('top1','top5')):
            ax=axes[row,col]
            means=np.array([r['methods'][m]['metrics'][k]['mean']*100 for m in methods])
            low=np.array([r['methods'][m]['metrics'][k]['ci95'][0]*100 for m in methods])
            high=np.array([r['methods'][m]['metrics'][k]['ci95'][1]*100 for m in methods])
            ax.barh(np.arange(3),means,height=.55,color=['#b9c5d0','#708b9e','#187b80'],
                    xerr=[means-low,high-means],error_kw={'color':'#30475d','capsize':3})
            ax.set_yticks(np.arange(3),[ENGLISH[m] for m in methods]);ax.invert_yaxis()
            for i,m in enumerate(methods):
                s=r['methods'][m];ax.text(min(high[i]+1,102),i,f"{s[k]}/{s['n']}",va='center',fontsize=9)
            ax.set_xlim(0,110);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Accuracy (%)')
            ax.set_title(('RootSE two-stage' if d=='rootse' else 'Who&When development')+' · '+k.replace('top','Top-'),loc='left',fontweight='bold')
            ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('v5 matched controls; 95% group bootstrap intervals',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(dest/('01-main.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),constrained_layout=True)
    specifications=[('rootse',['initial_top1','initial_rank2_to5','initial_miss'],
        ['First pass rank 1','First pass ranks 2–5','First pass misses'],['review_same','review_evidence'],'top1'),
        ('whowhen',['full_fits_16k','full_exceeds_16k'],['Full context ≤16K','Full context >16K'],
         ['balanced_repeat','coverage'],'top5')]
    for ax,(d,keys,labels,methods,k) in zip(axes,specifications):
        strata=results[d]['strata'];x=np.arange(len(keys))
        for offset,m in enumerate(methods):
            values=[strata[g][m]['metrics'][k]['mean']*100 for g in keys]
            bars=ax.bar(x+(offset-.5)*.3,values,width=.28,label=ENGLISH[m],color=['#708b9e','#187b80'][offset])
            for bar,g in zip(bars,keys):
                s=strata[g][m];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{s[k]}/{s['n']}",ha='center',fontsize=9)
        ax.set_xticks(x,labels);ax.set_ylim(0,114);ax.set_ylabel(k.replace('top','Top-')+' accuracy (%)')
        ax.set_title('RootSE: initial candidate quality' if d=='rootse' else 'Who&When: need for compression',loc='left')
        ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for suffix in ('png','pdf'):fig.savefig(dest/('02-mechanism-strata.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    results={d:read(out/'analysis/summary.json') for d,out in OUTPUTS.items()}
    adjusted=read(OUTPUTS['rootse']/'analysis/primary_family_holm.json')
    for d,out in OUTPUTS.items():
        target=DEST/d;target.mkdir(exist_ok=True)
        for name in ('summary.json','predictions.json','cases.csv','primary_family_holm.json'):
            shutil.copy2(out/'analysis'/name,target/name)
        for name in ('manifest.json','input_audit.json','coverage_after_freeze.json','analysis_code_before_results.json'):
            shutil.copy2(out/name,target/name)
        for name in ('input_reconstruction.json','replay.json'):
            shutil.copy2(out/'verification'/name,target/name)
    figures(results)
    report=['# v5实验分析','','主问题：候选证据展开是否优于同样两轮的原上下文重读；图覆盖分配是否优于同期v4。RootSE为102条/35组，Who&When为184条/137组；两套均为开发诊断，每条件一次调用。','',
            '## RootSE','',table(results['rootse']['methods']),'','## Who&When','',table(results['whowhen']['methods']),'',
            '主比较及所有分层见stats-appendix.md。历史v4是参照，主对照均为本轮新增调用。RootSE两轮部署成本包含已缓存第一轮，新增实付用量另列。']
    report+=['','## 可支持的结论']
    for key,c in adjusted.items():
        report += ['',f"- 观察：{key}，{NAMES[c['a']]}相对{NAMES[c['b']]}变化{100*c['mean']:+.2f}个百分点。",
            f"  - 证据：逐条request/SSE/state，{key.split('/')[0]}/predictions.json；得{len(c['gained'])}条、失{len(c['lost'])}条。",
            f"  - 不确定性：95%组bootstrap区间[{100*c['ci95'][0]:+.2f},{100*c['ci95'][1]:+.2f}]，Holm p={c['p_holm']:.4f}。",
            '  - 可采用表述：本轮固定数据上的配对观测差异。',
            '  - 不作更强表述：不把一次运行、开发数据结果称为稳定的独立泛化优势。',
            '  - 下一步：结合预定机制分层，检验效应来自新增证据还是重复调用；后续独立数据需重新冻结。',
            '  - 决策：'+('保留本轮配对差异' if c['p_holm']<.05 else '弱化为描述性发现')]
    (DEST/'analysis-report.md').write_text('\n'.join(report)+'\n')
    stats=['# v5统计附录','','同一轨迹两种新条件配对。按原有任务/重复轨迹组整体bootstrap 5,000次；主检验为20,000次整组配对符号置换，随机种子沿用20261028。四项Top-1/Top-5比较统一Holm。组内不假设各轨迹独立；符号置换假设零效应下组差值可交换符号。区间与置换p来自不同过程。单次服务运行不提供跨运行标准差。','',
        '| 比较 | 差异pp | 95%CI pp | 得/失 | p | Holm p |','|---|---:|---:|---:|---:|---:|']
    for name,c in adjusted.items():
        stats.append(f"| {name} | {100*c['mean']:+.2f} | [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] | {len(c['gained'])}/{len(c['lost'])} | {c['p']:.4f} | {c['p_holm']:.4f} |")
    for d,r in results.items():
        for name,strata in r['strata'].items():
            stats+=['',f'## {d} / {name}','',table(strata)]
        stats+=['',f'## {d} 实际用量','','| 条件 | 请求 | Prompt | Completion | Total | 未知usage请求 |','|---|---:|---:|---:|---:|---:|']
        for m,s in r['methods'].items():
            cost=s['recorded_call_cost']
            stats.append(f"| {NAMES[m]} | {cost.get('requests',0)} | {cost.get('prompt_tokens',0):,} | {cost.get('completion_tokens',0):,} | {cost.get('total_tokens',0):,} | {cost.get('unknown_usage_requests',0)} |")
            if 'deployment_two_stage_cost' in s:
                cost=s['deployment_two_stage_cost']
                stats.append(f"| {NAMES[m]} 两轮累计 | {cost.get('requests',0)} | {cost.get('prompt_tokens',0):,} | {cost.get('completion_tokens',0):,} | {cost.get('total_tokens',0):,} | {cost.get('unknown_usage_requests',0)} |")
    (DEST/'stats-appendix.md').write_text('\n'.join(stats)+'\n')
    (DEST/'figure-catalog.md').write_text('''# 图表说明

01-main：比较两套数据的Top-1/Top-5。数据为各summary.json，数字是命中数/完整分母；误差线是95%组bootstrap区间。RootSE灰色历史单轮与后两种两轮方法调用数不同；主判断看两个同期新条件和配对差值。Who&When已用于误差分析。检查增益是否同时保持Top-5、是否超过重复调用波动。

02-mechanism-strata：左侧按第一轮根因位置分组，观察二轮能否把已召回根因升至首位、是否破坏已有正确首选；右侧按完整上下文是否超过16K，观察覆盖方案在真正需要压缩的轨迹上是否有效。柱高为分层准确率，标注分子/分母，无误差线；属于预定机制描述，不作为额外显著性主检验。小子集的一个案例变化占比较大。

每张图均提供PNG和矢量PDF，图表只读取已完成且核验的实际结果。
''')
    save(DEST/'combined_summary.json',results)
    save(DEST/'artifact_manifest.json',{'time':now(),'files':{str(p.relative_to(DEST)):sha(p)
         for p in DEST.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}})
    print(DEST,flush=True)


if __name__=='__main__':
    main()
