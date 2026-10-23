#!/usr/bin/env python3
"""Offline final evaluation: full denominators, paper audit, paired comparisons."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from suite_transport import digest,save,now
from report_rootse_baseline import summarize,tolerance,audit_released
from compare_rootse_models import paired_counts

ROOT=Path(__file__).resolve().parents[1]
NAMES={'all_at_once':'All-at-Once','step_by_step':'Step-by-Step','binary_search':'Binary Search','TRAIL':'TRAIL','FAMAS':'FAMAS','agent_locate':'TrajAudit'}
ORDER=list(NAMES)


def request_accounting(case_dir:Path)->tuple[list[dict],list[dict]]:
    attempts=[];calls=[]
    for p in sorted(case_dir.glob('calls/*/state.json')):
        s=json.loads(p.read_text())
        if digest(json.loads((p.parent/'request.json').read_text()))!=s['request_sha256']:
            raise ValueError('Request hash mismatch: '+str(p))
        calls.append(s)
        attempts.extend(s['attempts'])
    return attempts,calls


def totals(attempts:list[dict])->dict:
    starts=[datetime.fromisoformat(a['started_at']).timestamp() for a in attempts]
    ends=[datetime.fromisoformat(a['started_at']).timestamp()+a.get('elapsed_seconds',0) for a in attempts]
    return {'api_requests':len(attempts),'known_total_tokens':sum((a.get('usage') or {}).get('total_tokens',0) for a in attempts),
            'known_input_tokens':sum((a.get('usage') or {}).get('prompt_tokens',0) for a in attempts),
            'known_output_tokens':sum((a.get('usage') or {}).get('completion_tokens',0) for a in attempts),
            'attempts_without_usage':sum(not a.get('usage') for a in attempts),
            'http_status_counts':dict(Counter(str(a.get('http_status',a['status'])) for a in attempts)),
            'request_time_sum_seconds':sum(a.get('elapsed_seconds',0) for a in attempts),
            'active_span_seconds':max(ends)-min(starts) if starts else 0}


def compare_methods(records:dict[str,list[dict]])->list[dict]:
    baseline={r['trajectory_id']:r for r in records['all_at_once']}
    result=[]
    for method in ORDER[1:]:
        candidate={r['trajectory_id']:r for r in records[method]}
        ids=sorted(baseline)
        pair=paired_counts([baseline[i]['tolerance_hits'][0] for i in ids],[candidate[i]['tolerance_hits'][0] for i in ids])
        result.append({'baseline':'all_at_once','candidate':method,**pair,
                       'delta_pp':100*(pair['right_only']-pair['left_only'])/102})
    # Holm adjustment for the five prespecified comparisons against All-at-Once.
    previous=0.0
    for rank,index in enumerate(sorted(range(len(result)),key=lambda i:result[i]['exact_mcnemar_two_sided_p'])):
        value=min(1.0,(len(result)-rank)*result[index]['exact_mcnemar_two_sided_p'])
        previous=max(previous,value);result[index]['holm_adjusted_p']=previous
    return result


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();source=a.source.resolve()
    manifest=json.loads((out/'manifest.json').read_text());reference=json.loads((out/'snapshot/paper_reference.json').read_text())
    hashes=json.loads((out/'snapshot/dataset_manifest.json').read_text())
    for rel,d in hashes.items():
        if hashlib.sha256((source/rel).read_bytes()).hexdigest()!=d:raise ValueError('Dataset changed')
    records=defaultdict(list);attempts_by=defaultdict(list);all_attempts=[];all_rows=[];failures=[];raw_cache={}
    for case in manifest['cases']:
        directory=out/case['directory'];path=directory/'state.json'
        if not path.exists():raise ValueError('Incomplete suite: missing '+str(path))
        state=json.loads(path.read_text())
        if state['status'] in {'running','prepared','worker_error'}:raise ValueError('Incomplete suite: '+str(path))
        rel=case['source_file']
        if rel not in raw_cache:raw_cache[rel]=json.loads((source/rel).read_text())
        raw=raw_cache[rel];gold=str(raw['failure_id'])
        flags=tolerance(state.get('failure_id'),gold,state['status']=='valid')
        attempts,calls=request_accounting(directory)
        rec={**state,'gold_step':gold,'tolerance_hits':flags,'record_directory':case['directory'],
             'logical_calls':len(calls),'accounting':totals(attempts)}
        records[case['method']].append(rec);attempts_by[case['method']]+=attempts;all_attempts+=attempts
        all_rows.append({'method':case['method'],'trajectory_id':rec['trajectory_id'],'status':rec['status'],
                         'prediction':rec.get('failure_id',''),'gold_step':gold,
                         **{f'tol{k}':int(flags[k]) for k in range(4)},'logical_calls':len(calls),
                         'api_requests':len(attempts),'known_tokens':rec['accounting']['known_total_tokens'],
                         'unknown_usage_requests':rec['accounting']['attempts_without_usage'],'record_directory':case['directory']})
        if state['status']!='valid':failures.append({'method':case['method'],'trajectory_id':rec['trajectory_id'],'status':rec['status'],'reason':rec.get('failure_reason',''),'record_directory':case['directory']})
    metrics={};groups={}
    for method,rows in records.items():
        if len(rows)!=102 or len({r['trajectory_id'] for r in rows})!=102:raise ValueError('Each method requires 102 unique trajectories')
        metrics[method]={**summarize(rows),**totals(attempts_by[method]),'logical_calls':sum(r['logical_calls'] for r in rows)}
        grouped=defaultdict(list)
        for r in rows:grouped[r['agent']+'/'+r['trajectory_model']].append(r)
        groups[method]={g:summarize(v) for g,v in grouped.items()}
    if set(metrics)!=set(ORDER):raise ValueError('Missing method')
    paired=compare_methods(records)
    audits=[]
    for method in ORDER:
        audits.append(next(r for r in audit_released(source,records[method],reference) if r['method']==method))
    summary={'model':manifest['protocol']['config']['model'],'n_per_method':102,'metrics':metrics,'paired_vs_all_at_once':paired,
             'accounting':totals(all_attempts),'protocol_fingerprint':manifest['fingerprint'],'evaluated_at':now()}
    save(out/'metrics.json',summary);save(out/'evaluation.json',records);save(out/'groups.json',groups)
    save(out/'paper_audit.json',audits);save(out/'failures.json',failures)
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    table=['| 方法 | Tol@0 | Tol@1 | Tol@2 | Tol@3 | 有效输出 | 论文Tol@0 |','|---|---:|---:|---:|---:|---:|---:|']
    cost=['| 方法 | 逻辑调用 | 实际请求 | 已知tokens | 未知usage请求 | 终态失败 |','|---|---:|---:|---:|---:|---|']
    for method in ORDER:
        m=metrics[method]
        table.append('| '+NAMES[method]+' | '+' | '.join(f'{x*100:.2f}% ({h}/102)' for x,h in zip(m['tolerance_accuracy'],m['tolerance_hits']))+f" | {m['valid_predictions']}/102 | {reference['tolerance_percent'][method][0]}% |")
        cost.append(f"| {NAMES[method]} | {m['logical_calls']} | {m['api_requests']} | {m['known_total_tokens']:,} | {m['attempts_without_usage']} | {dict((k,v) for k,v in m['status_counts'].items() if k!='valid')} |")
    pairs=['| 对比All-at-Once | 精确命中变化(pp) | 仅All命中 | 仅候选命中 | 精确McNemar p | Holm p |','|---|---:|---:|---:|---:|---:|']
    for r in paired:pairs.append(f"| {NAMES[r['candidate']]} | {r['delta_pp']:+.2f} | {r['left_only']} | {r['right_only']} | {r['exact_mcnemar_two_sided_p']:.4g} | {r['holm_adjusted_p']:.4g} |")
    group_table=['| 轨迹来源 | N | '+' | '.join(NAMES[m] for m in ORDER)+' |','|---|---:|'+'---:|'*len(ORDER)]
    for group in sorted(groups['all_at_once']):
        group_table.append('| '+group+f" | {groups['all_at_once'][group]['n']} | "+' | '.join(f"{groups[m][group]['tolerance_hits'][0]}/{groups[m][group]['n']}" for m in ORDER)+' |')
    prior_path=ROOT/'outputs/rootse-full-deepseek-v41-flash-v1/metrics.json'
    budget_note='先前8192预算结果不在本目录；不据此补造预算对照。'
    if prior_path.exists() and summary['model'] == 'deepseek-v4.1-flash':
        old=json.loads(prior_path.read_text());new=metrics['all_at_once']
        budget_note=f"此前同模型8192预算All-at-Once为{old['tolerance_hits'][0]}/102、有效{old['valid_predictions']}/102；本次32768预算为{new['tolerance_hits'][0]}/102、有效{new['valid_predictions']}/102。两次为独立请求，温度0不保证确定性，差异不能全部归因于输出预算。旧轮的3种骨干结果保持在各自目录。"
    refs=json.loads((ROOT/'experiments/rootse_suite/reference_audit.json').read_text())
    best=max(ORDER,key=lambda m:metrics[m]['tolerance_hits'][0])
    best_names='、'.join(NAMES[m] for m in ORDER if metrics[m]['tolerance_hits'][0]==metrics[best]['tolerance_hits'][0])
    total=summary['accounting']
    report=f'''# RootSE 全部基线实验总结

本轮完整运行6种方法×102条，共612个方法/样本任务。骨干统一为`{summary['model']}`，without-reference，temperature=0，每次输出预算32768 tokens。所有终态失败均留在102分母。单轮精确命中最高的方法为{best_names}（{metrics[best]['tolerance_hits'][0]}/102）；这不是跨数据集或多随机种子结论。

## 主结果及论文对照

{chr(10).join(table)}

论文列为[TrajAudit v3](https://arxiv.org/abs/2605.26563v3)的Claude Sonnet4.5结果，本轮没有调用Claude。跨骨干、预算及服务协议的差异使它成为复现对照，不能据此声称复现了论文同一实验，或把提升全部归因于算法。

官方公开CSV审计：All-at-Once 25/102、Binary Search 12/102、TRAIL/FAMAS各31/102、TrajAudit53/102，与论文精度相符；Step-by-Step CSV有106行/102个独立案例，固定首行和末行去重均20/102，不能复现论文22.6%。完整计数及逐方法配对见paper_audit.json，未按预测是否正确选择重复行。

## 同骨干方法比较

{chr(10).join(pairs)}

五项对比预先固定为各方法对All-at-Once，报告双侧精确McNemar并用Holm校正。统计量仅描述此102条配对；多个案例来自同任务/仓库，单次推理及样本相关性限制外推。Wilson区间见metrics.json，同样不是模型随机性区间。总体排序需结合失败率、额外输入及资源开销解释。

## 资源与失败

{chr(10).join(cost)}

共{total['api_requests']}次实际请求，已知{total['known_total_tokens']:,} tokens（输入{total['known_input_tokens']:,}，输出{total['known_output_tokens']:,}），{total['attempts_without_usage']}次usage未知，未视为零成本。按全部请求起止范围统计的本轮墙钟跨度约{total['active_span_seconds']/60:.2f}分钟，16个隔离案例进程并发，各方法运行重叠；不能用该跨度比较单方法独占延迟。未根据token用量推算实际费用。

失败详情见failures.json，原始响应及每次重试见各案例calls目录。finish_reason非stop直接标为output_truncated，不从思考或残缺JSON抽取答案；输入超限独立分类。网络/429/5xx最多3次，其他服务失败终止该案例。输出解释未交给LLM裁判，不报告Justification Accuracy。

## 分组结果

{chr(10).join(group_table)}

## 实验实现与可比性

- 官方源固定提交`{manifest['protocol']['config']['source_commit']}`。直接调用其算法函数，不重写算法提示词；数据、代码、模板及714份FAMAS参考轨迹逐文件哈希保留在manifest.json。推理对象的gold_patch、根因ID和expected字段已置空/隔离，仅评估脚本读取标签。
- All-at-Once完整输入一次定位；Step-by-Step按官方Yes/No/Passed逐步停止；Binary Search按官方半区选择缩小范围；TRAIL先提取错误再按标注准则选最早偏差。
- FAMAS保留官方动作抽取、语义聚类、λ=0.9频谱分析及最终LLM定位。每案例最多10份参考轨迹；{refs['without_refs']}例缺参考、{refs['without_passing_refs']}例没有成功参考、{refs['below_three_total_trajectories']}例不满足总计三条轨迹。其成功/失败轨迹是额外输入，不等同于提供参考补丁。官方动作/思考及观察摘要截断、聚类前100个键、规则回退、ID范围裁剪均保留，不能说FAMAS的LLM读到了每条完整观察。
- TrajAudit使用官方generic OpenAI兼容版本，diagnosis=True、context-aware-truncation=True，诊断后最多5次调查行动；观察折叠与按需查看中的2000字符限制保留。与论文Claude专用实现存在后端适配差异。
- 方法自带的JSON提取/修复、格式重试和算法回退保留；不是统一解析器消融。统一传输层将temperature固定0、max_tokens固定32768，忽略TRAIL原代码的enable_thinking=False，思考及流式设置以本轮snapshot/config.json为准；原调用参数也另存于每次call/state.json。FAMAS原10000/8000/1024预算被统一预算覆盖。未完成API响应一律中止案例，这比部分官方方法尝试解析残缺输出更严格。
- 逐调用缓存先记录请求，后保存原始响应和usage。恢复时从方法入口重放已有响应；请求哈希改变则拒绝继续，已完成案例直接复用。崩溃时无法确认的远端请求标unknown而非零成本。单条最多1024逻辑调用是固定异常循环保护，触发则单独计失败。

## 预算与前序实验

{budget_note}

历史DeepSeek模型选型与近百万输入容量测试保留在experiments/rootse_full/deepseek_capacity_probes.json；该测试不代表本轮其他模型的容量，也不证明模型有效使用所有长上下文。当前方法差异、真实诊断效能与容量上限需分开解释。

## 复现与产物

仓库根目录执行：

```bash
bash scripts/reproduce_rootse_suite.sh /path/to/moma_ai.key
```

入口安装锁定依赖、执行测试、固定数据与代码、断点运行全部方法，最终离线评估。已有终态案例及中间响应复用，不重复付费。新独立运行可传第二个输出目录参数。

- metrics.json / predictions.csv / evaluation.json：全量、逐案例及统计结果。
- paper_audit.json / groups.json / failures.json：论文CSV复核、分组和失败。
- `<method>/<case>/calls/`：每次实际请求、响应、重试及usage。
- `<method>/<case>/upstream/`：官方原始CSV及TrajAudit调查轨迹。
- snapshot/、environment.json、manifest.json：冻结代码、参数、版本和文件哈希。

协议指纹`{manifest['fingerprint']}`。过程记录见experiments/rootse_suite/PROCESS.md。全量预测固定后再生成此总结，没有依据标签反向调整提示词或筛选案例。
'''
    if (out/'analysis/analysis-report.md').exists():
        report+='\n本轮完整复核、失败审计和图表见[研究总结](2026-10-06--rootse-baselines--r1--round-review.md)与[分析报告](analysis/analysis-report.md)。一键重建含图报告：`bash scripts/reproduce_rootse_review.sh /path/to/moma_ai.key`。\n'
    (out/'EXPERIMENT_SUMMARY.md').write_text(report)
    save(out/'evaluation_provenance.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'label_access':'offline_after_all_612_terminal','generated_at':now()})
    print(json.dumps({'summary_path':str(out/'EXPERIMENT_SUMMARY.md'),'hits':{m:metrics[m]['tolerance_hits'] for m in ORDER},'accounting':total},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
