#!/usr/bin/env python3
"""Offline analysis bundle; never requests a model or modifies predictions."""
import argparse, hashlib, json, sys
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORDER=['all_at_once','step_by_step','binary_search','TRAIL','FAMAS','agent_locate']
NAMES=['All-at-Once','Step-by-Step','Binary Search','TRAIL','FAMAS','TrajAudit']

def read(p): return json.loads(p.read_text())
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve();source=a.source.resolve();bundle=out/'analysis';fig=bundle/'figures';fig.mkdir(parents=True,exist_ok=True)
    manifest=read(out/'manifest.json');metrics=read(out/'metrics.json');ev=read(out/'evaluation.json')
    assert len(manifest['cases'])==612
    tally={m:Counter() for m in ORDER};hits={m:[0]*4 for m in ORDER};truncated=[];empty=[];calls=0;attempts=[];sources={};stage=Counter()
    for c in manifest['cases']:
        directory=out/c['directory'];s=read(directory/'state.json');raw=sources.setdefault(c['source_file'],read(source/c['source_file']))
        tally[c['method']][s['status']]+=1
        assert s['status'] not in ['running','prepared','worker_error']
        for k in range(4):
            hits[c['method']][k]+=int(s['status']=='valid' and abs(int(s['failure_id'])-int(raw['failure_id']))<=k)
        for p in sorted(directory.glob('calls/*/state.json')):
            cs=read(p);calls+=1;attempts.extend(cs['attempts'])
            for att in cs['attempts']:
                if att.get('http_status')==200:
                    response=read(p.parent/f"response-{att['number']}.json")
                    assert response.get('usage')==att.get('usage')
                    assert response['model']=='deepseek-v4.1-flash'
            if not cs.get('response_file'):continue
            response=read(p.parent/cs['response_file']);choice=response['choices'][0];message=choice['message'];usage=response.get('usage',{})
            record={'method':c['method'],'case':c['directory'],'call':p.parent.name,'stage':cs.get('upstream_caller',['unknown'])[0],
                    'completion_tokens':usage.get('completion_tokens'),'reasoning_tokens':usage.get('completion_tokens_details',{}).get('reasoning_tokens'),
                    'content_present':bool(message.get('content')),'function_call_present':bool(message.get('function_call')),'tool_calls_present':bool(message.get('tool_calls'))}
            if choice['finish_reason']!='stop':
                truncated.append(record)
                if c['method']=='FAMAS':stage[record['stage']]+=1
            if choice['finish_reason']=='stop' and not message.get('content'):empty.append(record)
    for m in ORDER:
        assert sum(tally[m].values())==102 and hits[m]==metrics['metrics'][m]['tolerance_hits']
        assert dict(tally[m])==metrics['metrics'][m]['status_counts']
    total_tokens=sum((x.get('usage') or {}).get('total_tokens',0) for x in attempts)
    assert total_tokens==metrics['accounting']['known_total_tokens'] and len(attempts)==metrics['accounting']['api_requests']
    audit={'case_states':612,'call_states':calls,'requests':len(attempts),'known_tokens':total_tokens,
           'unknown_usage_requests':sum(not x.get('usage') for x in attempts),'independent_tolerance_counts_match':True,
           'response_usage_matches_call_state':True,'repositories':len({r['repo'] for r in sources.values()}),
           'unique_tasks':len({(r['repo'],r['instance_id']) for r in sources.values()}),'truncation_count':len(truncated),
           'reasoning_only_truncations':sum(not r['content_present'] and r['reasoning_tokens']==32768 for r in truncated),
           'famas_truncation_stages':dict(stage),'truncated_calls':truncated,'stop_without_content':empty}
    # This narrative describes the frozen v1 run; refuse to reuse it for a different result.
    assert [hits[m] for m in ORDER] == [[56,64,68,73],[26,33,36,39],[49,56,60,64],[40,51,59,62],[27,35,36,38],[27,33,35,35]], 'Different results: create a new analysis narrative'
    assert (calls,len(attempts),total_tokens,len(truncated),audit['reasoning_only_truncations']) == (2777,2801,74117594,174,171)
    write(bundle/'audit.json',audit)
    rows=['| 方法 | Tol@0 | Tol@1 | Tol@2 | Tol@3 | 有效输出 | 请求 | 已知 tokens |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,n in zip(ORDER,NAMES):
        v=metrics['metrics'][m];rows.append('| '+n+' | '+' | '.join(f'{h}/102 ({h/102:.2%})' for h in hits[m])+f" | {v['valid_predictions']}/102 | {v['api_requests']} | {v['known_total_tokens']:,} |")
    table='\n'.join(rows);(bundle/'main-results.md').write_text(table+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.dpi':180})
    y=np.arange(6);acc=np.array([hits[m][0]/102*100 for m in ORDER]);paper=np.array([24.5,22.6,11.8,30.4,30.4,52.0])
    f,ax=plt.subplots(figsize=(10,5.5));ax.barh(y-.17,acc,.32,label='This run: DeepSeek-V4.1-Flash',color='#2563a6');ax.barh(y+.17,paper,.32,label='Paper: Claude Sonnet 4.5 (descriptive only)',color='#a7b9cc')
    for i,v in enumerate(acc):ax.text(v+.5,i-.17,f'{v:.1f}% ({hits[ORDER[i]][0]}/102)',va='center',fontsize=9)
    ax.set(yticks=y,yticklabels=NAMES,xlim=(0,75),xlabel='Exact localization accuracy (%, all 102 cases)',title='RootSE: completed single-run baseline suite');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=9);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    f.text(.01,.01,'Different backbones / serving protocols; paper comparison does not isolate algorithm effects.',fontsize=9);f.tight_layout(rect=(0,.15,1,1))
    for ext in ['png','pdf']:f.savefig(fig/f'exact-comparison.{ext}')
    plt.close(f)
    f,ax=plt.subplots(figsize=(10,5.5));left=np.zeros(6)
    for key,label,color in [('valid','Valid output','#277f8e'),('output_truncated','Output truncated','#de8649'),('no_prediction','No prediction','#bd6570'),('empty_response','Empty response','#8c7cad')]:
        vals=np.array([tally[m][key] for m in ORDER]);ax.barh(y,vals,left=left,label=label,color=color)
        for i,v in enumerate(vals):
            if v:ax.text(left[i]+v/2 if v>=3 else 103,i,str(v),ha='center',va='center',color='white' if v>=3 else '#333333',fontsize=10)
        left+=vals
    ax.set(yticks=y,yticklabels=NAMES,xlim=(0,105),xlabel='Number of cases (102 per method)',title='Completion is a material constraint under 32,768 tokens per call');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=9);f.tight_layout()
    for ext in ['png','pdf']:f.savefig(fig/f'completion-outcomes.{ext}',bbox_inches='tight')
    plt.close(f)
    pairrows=['| 候选 vs All-at-Once | 差值 pp | 仅 All 命中 | 仅候选命中 | McNemar p | Holm p |','|---|---:|---:|---:|---:|---:|']
    for r in metrics['paired_vs_all_at_once']:
        pairrows.append(f"| {NAMES[ORDER.index(r['candidate'])]} | {r['delta_pp']:+.2f} | {r['left_only']} | {r['right_only']} | {r['exact_mcnemar_two_sided_p']:.6g} | {r['holm_adjusted_p']:.6g} |")
    stats=f'''# 统计附录

实验单位是原始失败轨迹，6种方法配对共享102条。包含{audit['unique_tasks']}个不同任务、{audit['repositories']}个仓库，任务/仓库内相关性存在。每个方法每条轨迹仅一次生成，没有独立seed重复，不提供跨seed均值、标准差或稳定性结论。温度0不保证服务确定性。

Tol@k要求有效预测且 |预测步骤−标注步骤|≤k，k=0..3；所有失败一律记0，所有方法分母102。步骤ID连续且从0起始。重新读取原始failure_id、全部612份案例state独立核算与metrics一致。2777份调用state中2801次attempt已逐项计入，成功响应usage与调用state完全一致，未知usage24次未当作零成本。

配对差异为候选−All-at-Once。以下是按102个配对独立假设计算的探索性双侧精确McNemar检验；5项比较Holm校正。相关性使p值及Wilson区间不能作为严格的总体显著性证明，也不刻画模型生成波动。主结论使用计数与差值，需新增独立重复/任务或仓库分层设计后确认。

{chr(10).join(pairrows)}

Binary Search低6.86pp，p=0.189；没有检出差异不能推出两方法等效。其余方法在该独立假设下较低，但完成率、输入与默认thinking差异使结果不能解释为算法本身优劣。Wilson95%区间保存于metrics.json，不在主图画跨骨干推断误差条。

FAMAS成功/失败参考轨迹是额外输入；无参考补丁协议不是“所有方法无参考轨迹”。44/102没有成功参考。TrajAudit采用generic版本与最多5次调查；不能把本次26.47%直接解释为论文结果不成立。
'''
    (bundle/'stats-appendix.md').write_text(stats)
    analysis=f'''# RootSE 基线分析

六方法×102条已全部结束。主比较采用统一移动云deepseek-v4.1-flash、temperature=0、每次输出32768、16案例并发。All-at-Once为56/102（54.90%），Binary Search为49/102（48.04%）。这是单轮固定预算和服务协议下的观测结果。

{table}

## 失败与资源

413有效、174输出截断、13无预测、12空正文；没有输入超上下文、终态服务失败或脚本失败。174个截断响应都用满32768输出tokens，171个没有正文且全部是reasoning tokens。FAMAS截断13个发生在动作抽取，37个在最终定位，不能将主要失败归咎于动作JSON解析。TrajAudit的12个stop空正文响应没有function_call或tool_calls字段提供替代答案；不从思考内容恢复预测。

总计2801请求（2777 HTTP200、23 HTTP502、1传输错误），2777逻辑调用，已知74,117,594 tokens；24个失败尝试用量未知。重试均按冻结策略执行；全部失败保留分母。每次调用同预算并不意味着每方法同总算力。调用跨度129.06分钟，各方法并发重叠，不用于单方法速度排名。

## 可用于后续研究的判断

| 候选判断 | 证据 | 允许措辞 | 不允许的更强措辞 | 不确定性与下一步 | 决策 |
|---|---|---|---|---|---|
| 全轨迹基线必须保留 | All-at-Once 56/102，本轮最高 | 本协议下观测最佳 | 普遍优于所有复杂定位方法 | 单轮、小样本相关；未来固定独立重复设计 | 保留为迁移方法主对照 |
| Binary Search仍有比较价值 | 49/102，较All少7；配对14比7 | 当前未证明与All有差异 | 与All等效或显著更差 | 独立假设p=.189，需更大独立样本和重复 | 保留第二主对照 |
| 多阶段完成率是关键限制 | 截断174，其中171为纯思考耗尽 | 当前服务预算下失败较多 | 方法能力上限只有当前准确率 | 需预注册思考开关/预算消融；新目录全量运行 | 不改本轮，记录未来实验 |
| 论文对照只能描述 | 骨干与输入/实现存在差异 | 与论文表格的观测差值 | 复现证明论文错误 | SBS公开CSV与表格不一致，需作者澄清 | 保留两套数值与来源 |

迁移LogHound到agent的下一步应验证结构化锚点和回溯是否优于强All-at-Once，并控制总tokens、输入证据与失败处理；本轮没有运行完整LogHound图方法，不宣称迁移已经有效。此前规则图结果与本轮模型基线分开保留。

## 图与来源

![精确定位对照](figures/exact-comparison.png)

图1展示完整102分母下的点估计。论文柱只做描述性对照；Step-by-Step论文22.6%，公开CSV去重20/102并不一致。

![完成状态](figures/completion-outcomes.png)

图2分离完成输出与终态失败；有效不代表定位正确。它支持失败率解释，不支持排除失败后重新排名。

数据来自../metrics.json、../evaluation.json、../paper_audit.json及原始state/response。统计假设见stats-appendix.md；审计明细见audit.json；图及脚本哈希见provenance.json。没有Obsidian项目绑定，本轮仅保存本地分析，不写回Obsidian。
'''
    (bundle/'analysis-report.md').write_text(analysis)
    (bundle/'figure-catalog.md').write_text('''# 图表目录

| 编号 | 文件 | 输入 | 支持的判断 | 限制 |
|---|---|---|---|---|
| 1 | figures/exact-comparison.png / .pdf | metrics.json；论文表格 | 本轮All-at-Once观测最高 | 不同骨干，单轮，非因果比较 |
| 2 | figures/completion-outcomes.png / .pdf | 独立重读612案例state | 失败类型和完成率差异 | 有效输出不等于命中 |
| 表1 | main-results.md | 原始gold+state核算 | Tol@0..3、102分母、请求用量 | 无多seed均值/方差 |

全部图由scripts/analyze_rootse_suite.py离线生成，PNG用于预览，PDF用于导出。均不删除失败样本或改变预测。
''')
    provenance={'script_sha256':sha(Path(__file__)),'inputs':{n:sha(out/n) for n in ['manifest.json','metrics.json','evaluation.json','paper_audit.json']},'python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,'figures':{p.name:sha(p) for p in sorted(fig.iterdir())}}
    write(bundle/'provenance.json',provenance)
    report=f'''---
type: experiment-report
date: 2026-10-06
experiment_line: rootse-baselines
round: 1
purpose: round-review
status: complete
source_artifacts: [metrics.json, paper_audit.json, analysis/audit.json, reproduction_check.json]
linked_experiments: [../../experiments/rootse_suite/PROCESS.md]
linked_results: [EXPERIMENT_SUMMARY.md, analysis/analysis-report.md]
---

# RootSE Baselines / Round 1 / 全量基线总结 / 2026-10-06

六方法各102条全部完成；All-at-Once精确命中56条（54.90%），本轮最高，Binary Search49条（48.04%）。多阶段方法受到输出未完成的明显影响。后续LogHound迁移研究应以强全轨迹基线为主要对照，同时控制完成率和总资源。这里Round 1指六方法统一32k套件v1，不涵盖此前不同骨干的独立试跑。

## 实验身份、协议与主结果

模型统一移动云deepseek-v4.1-flash，temperature=0、每次输出32768、16案例并发、网络/5xx最多3尝试。原始根因标签只在全量终态后离线计分；无Claude调用，无LLM理由裁判。官方源码固定提交664e211b011bb0730816b06800bb841950fe0b38，协议指纹{manifest['fingerprint']}。

{table}

## 论文、统计与图表解释

论文Claude Sonnet4.5的Tol@0依次为24.5%、22.6%、11.8%、30.4%、30.4%、52.0%。不能将跨骨干差异归因于方法。作者Step-by-Step CSV有106行/102独立案例，首/末行固定去重均20/102，与22.6%不一致；没有根据正确性选重复行。其余五法CSV与表格一致。

五项对All-at-Once配对计数及探索性McNemar/Holm见[统计附录](analysis/stats-appendix.md)。102条涉及{audit['unique_tasks']}个任务、{audit['repositories']}个仓库，单轮且样本相关，不能报告跨seed稳定性或无条件显著性。Binary Search少7条，配对All独有14/候选独有7，p=.189；不构成等效结论。

[主比较图](analysis/figures/exact-comparison.png)展示完整分母的观测准确率，[完成状态图](analysis/figures/completion-outcomes.png)解释输出失败限制，PDF版本同目录。图表不支持算法固有能力排名。

## 失败、资源与可比性

413有效、174截断、13无预测、12空正文。174截断中171仅产生思考并耗满32768 tokens。FAMAS截断13例发生在动作抽取、37例最终定位；TrajAudit12空正文均没有结构化函数/工具调用可作为答案。没有输入超限或终态服务失败，失败全部保留102分母。

2777逻辑调用、2801请求；已知74,117,594 tokens（输入61,273,601，输出12,843,993），23次502及1次传输错误usage未知；不把其费用算零。活动跨度129.06分钟，方法并发重叠。不同方法总调用数不同，同每调用预算不代表同总成本。

FAMAS另有成功/失败参考轨迹，44例没有成功参考；TrajAudit原生观察折叠、2000字符查看限制和最多5步调查保留；TRAIL默认thinking与论文适配不同。统一拒绝非stop输出比部分原方法更严格。这些因素均限制与论文的直接比较，完整差异见[自动实验总结](EXPERIMENT_SUMMARY.md)。此前8192预算All-at-Once60/102，本轮56/102，独立请求的差异不能被解释为提高预算导致性能下降。

## 对研究判断的影响与后续动作

All-at-Once已经是强对照，不能只比较此前14B结果。当前多阶段低分主要需要结合输出完成率解释，不能据此放弃结构化定位。建议后续固定新实验协议后验证LogHound证据锚点/回溯：与All-at-Once及Binary Search对比，增加预算/思考与输入证据消融，并设计独立重复。本轮不修改协议，不重跑失败样本，不宣称完整LogHound迁移有效。

## 复现、审计与产物索引

在仓库执行`bash scripts/reproduce_rootse_review.sh /path/to/moma_ai.key`：先运行原冻结套件入口，再在独立绘图环境离线重建本报告及图。已有全部终态会复用。此复核入口限定本轮v1目录；新实验使用原套件入口并另建分析，不套用本轮结论。完整接口调用日志、state、request/response均保留。绘图不会读取密钥。

原入口复用审计见[reproduction_check.json](reproduction_check.json)：612案例/2777调用state哈希不变，2801请求前后相同、新增0，75测试通过。独立计分与usage审计见[analysis/audit.json](analysis/audit.json)。[分析报告](analysis/analysis-report.md)、[统计附录](analysis/stats-appendix.md)、[图表目录](analysis/figure-catalog.md)、[provenance](analysis/provenance.json)均由脚本生成。没有Obsidian绑定，仅在本地交付。
'''
    (out/'2026-10-06--rootse-baselines--r1--round-review.md').write_text(report)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['truncated_calls','stop_without_content']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
