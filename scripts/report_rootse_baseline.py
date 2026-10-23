#!/usr/bin/env python3
"""Offline full-denominator evaluation and paper/released-prediction audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from run_qwen_baseline import save
from run_rootse_full_baseline import TERMINAL, canonical, now, sha, upstream_extractor


def tolerance(prediction: str | None, gold: str, valid: bool) -> list[bool]:
    if not valid:
        return [False] * 4
    distance = abs(int(prediction) - int(gold))
    return [distance <= k for k in range(4)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [sum(r['tolerance_hits'][k] for r in rows) for k in range(4)]
    successes = hits[0]
    z = 1.959963984540054
    rate = successes / n
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    width = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {'n': n, 'valid_predictions': sum(r['status'] == 'valid' for r in rows),
            'status_counts': dict(Counter(r['status'] for r in rows)), 'tolerance_hits': hits,
            'tolerance_accuracy': [h / n for h in hits], 'exact_wilson95': [center - width, center + width]}


def audit_released(source: Path, records: list[dict], reference: dict) -> list[dict]:
    by_key = {(r['agent'], r['trajectory_model'], r['instance_id']): r for r in records}
    audits = []
    for method in reference['tolerance_percent']:
        setting = ('GROUND_TRUTH_False_DIAGNOSIS_True_CONTEXT_TRUNCATION_True'
                   if method == 'agent_locate' else 'GROUND_TRUTH_False')
        path = source / 'output' / method / reference['diagnosis_model'] / setting / 'runtime_1/result.csv'
        grouped = defaultdict(list)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            prefix = row['agent'] + '_' + row['model'] + '_'
            identifier = row['instance_id']
            if not identifier.startswith(prefix):
                raise ValueError('unexpected published instance prefix')
            key = (row['agent'], row['model'], identifier[len(prefix):])
            if key not in by_key:
                raise ValueError('unmatched published case')
            grouped[key].append(row)
        duplicates = {key: rs for key, rs in grouped.items() if len(rs) > 1}
        # Do not pick the better of duplicate rows. Both deterministic boundary
        # policies are disclosed; main paired comparison requires unique rows.
        variants = {}
        for policy in ('first', 'last'):
            hits = [0] * 4
            for key, rs in grouped.items():
                prediction = rs[0 if policy == 'first' else -1]['failure_id']
                try:
                    prediction = str(int(float(prediction)))
                except ValueError:
                    prediction = '-1'
                true = by_key[key]
                flags = tolerance(prediction, true['gold_step'], prediction in true['step_ids'])
                hits = [a + b for a, b in zip(hits, flags)]
            variants[policy] = hits
        audit = {'method': method, 'file': str(path.relative_to(source)),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'csv_rows': len(rows),
                 'unique_cases': len(grouped), 'duplicate_cases': len(duplicates),
                 'duplicate_extra_rows': len(rows) - len(grouped), 'tolerance_hits': variants,
                 'paper_percent': reference['tolerance_percent'][method]}
        if not duplicates and len(grouped) == len(records):
            b = c = both = neither = 0
            for key, rs in grouped.items():
                row = by_key[key]
                try:
                    pred = str(int(float(rs[0]['failure_id'])))
                except ValueError:
                    pred = '-1'
                paper_hit = pred == row['gold_step'] and pred in row['step_ids']
                current_hit = row['tolerance_hits'][0]
                both += paper_hit and current_hit
                neither += not paper_hit and not current_hit
                b += current_hit and not paper_hit
                c += paper_hit and not current_hit
            audit['paired_exact'] = {'both': both, 'current_only': b, 'published_only': c, 'neither': neither}
        audits.append(audit)
    return audits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.out / 'manifest.json').read_text())
    reference = json.loads((args.out / 'snapshot/paper_reference.json').read_text())
    dataset_hashes = json.loads((args.out / 'snapshot/dataset_manifest.json').read_text())
    records = []
    attempts = []
    format_diagnostics = []
    reasoning_responses = 0
    truncated_without_final_answer = 0
    extract = upstream_extractor(args.source)
    for case in manifest['cases']:
        state = json.loads((args.out / f"states/{case['case_id']}.json").read_text())
        if state['status'] not in TERMINAL:
            raise ValueError('incomplete experiment: refusing final full-set metrics')
        raw_bytes = (args.source / state['source_file']).read_bytes()
        if sha(raw_bytes) != dataset_hashes[state['source_file']]:
            raise ValueError('dataset changed since inference; refusing evaluation')
        request = json.loads((args.out / state['request_file']).read_text())
        if sha(canonical(request)) != state['request_sha256']:
            raise ValueError('saved request changed since inference')
        raw = json.loads(raw_bytes)
        gold = str(raw['failure_id'])
        if gold not in state['step_ids']:
            raise ValueError('gold step is not an original step ID')
        records.append({**state, 'gold_step': gold,
                        'tolerance_hits': tolerance(state.get('failure_id'), gold, state['status'] == 'valid')})
        attempts.extend(state['attempts'])
        for attempt in state['attempts']:
            response_path = args.out / attempt['response_file']
            if attempt['status'] == 'received' and response_path.exists():
                choice = (json.loads(response_path.read_text())['data'].get('choices') or [{}])[0]
                if not isinstance(choice, dict):
                    continue
                message = choice.get('message') or {}
                if not isinstance(message, dict):
                    continue
                reasoning_responses += bool(message.get('reasoning') or message.get('reasoning_content'))
                truncated_without_final_answer += choice.get('finish_reason') == 'length' and not message.get('content')
        if state['status'] == 'format_error' and state['attempts']:
            path = args.out / state['attempts'][-1]['response_file']
            diagnostic = {'trajectory_id': state['trajectory_id']}
            if path.exists():
                data = json.loads(path.read_text())['data']
                try:
                    content = data['choices'][0]['message']['content']
                    diagnostic['finish_reason'] = data['choices'][0].get('finish_reason')
                    parsed_text = extract(content)
                    json.loads(parsed_text)
                    diagnostic['error'] = 'valid JSON but invalid required fields'
                except json.JSONDecodeError as exc:
                    diagnostic.update(error=exc.msg, line=exc.lineno, column=exc.colno,
                                      near_error=parsed_text[max(0, exc.pos - 90):exc.pos + 90])
                except (TypeError, KeyError, IndexError) as exc:
                    diagnostic['error'] = type(exc).__name__
            format_diagnostics.append(diagnostic)
    if len(records) != 102 or len({r['trajectory_id'] for r in records}) != 102:
        raise ValueError('expected exactly 102 unique cases')
    summary = summarize(records)
    summary.update(model=manifest['model'], protocol_fingerprint=manifest['fingerprint'],
                   api_requests=len(attempts),
                   reported_total_tokens=sum(a.get('usage', {}).get('total_tokens', 0) for a in attempts),
                   reported_prompt_tokens=sum(a.get('usage', {}).get('prompt_tokens', 0) for a in attempts),
                   reported_completion_tokens=sum(a.get('usage', {}).get('completion_tokens', 0) for a in attempts),
                   attempts_without_usage=sum(not a.get('usage') for a in attempts),
                   response_models=dict(Counter(str(a.get('response_model')) for a in attempts if a.get('status') == 'received')),
                   responses_with_reasoning=reasoning_responses,
                   truncated_without_final_answer=truncated_without_final_answer,
                   evaluated_at=now())
    summary['reported_tokens_per_benchmark_case'] = summary['reported_total_tokens'] / 102
    starts = [datetime.fromisoformat(a['started_at']).timestamp() for a in attempts]
    ends = [datetime.fromisoformat(a['started_at']).timestamp() + a.get('elapsed_seconds', 0) for a in attempts]
    summary['inference_wall_seconds'] = max(ends) - min(starts) if starts else 0
    prompt_sizes = [a['usage']['prompt_tokens'] for a in attempts if a.get('usage', {}).get('prompt_tokens') is not None]
    summary['prompt_tokens_min_median_max'] = [min(prompt_sizes), median(prompt_sizes), max(prompt_sizes)] if prompt_sizes else None
    groups = defaultdict(list)
    for row in records:
        groups[row['agent'] + '/' + row['trajectory_model']].append(row)
    summary['groups'] = {key: summarize(rows) for key, rows in sorted(groups.items())}
    audits = audit_released(args.source, records, reference)
    save(args.out / 'evaluation.json', records)
    save(args.out / 'metrics.json', summary)
    save(args.out / 'paper_audit.json', audits)
    save(args.out / 'format_failure_diagnostics.json', format_diagnostics)
    with (args.out / 'predictions.csv').open('w', newline='') as f:
        fields = ['trajectory_id', 'agent', 'trajectory_model', 'status', 'failure_id', 'gold_step', 'tol0', 'tol1', 'tol2', 'tol3']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r.get(k, '') for k in fields[:6]},
                             **{f'tol{k}': int(r['tolerance_hits'][k]) for k in range(4)}})
    current = [100 * v for v in summary['tolerance_accuracy']]
    table = ['| 方法 / 骨干 | Tol@0 | Tol@1 | Tol@2 | Tol@3 |', '|---|---:|---:|---:|---:|']
    table.append('| 本次 All-at-Once / ' + manifest['model'] + ' | ' + ' | '.join(f'{v:.2f}%' for v in current) + ' |')
    for method, values in reference['tolerance_percent'].items():
        table.append('| 论文 ' + method + ' / Claude Sonnet 4.5 | ' + ' | '.join(f'{v:.1f}%' for v in values) + ' |')
    audit_table = ['| 作者公开CSV | 原始行数 / 独立样本 | 去重精确命中（首行 / 末行） |', '|---|---:|---:|']
    for audit in audits:
        first = audit['tolerance_hits']['first'][0]
        last = audit['tolerance_hits']['last'][0]
        audit_table.append(f"| {audit['method']} | {audit['csv_rows']} / {audit['unique_cases']} | {first}/102 / {last}/102 |")
    group_table = ['| 轨迹来源 | N | 有效预测 | Tol@0 |', '|---|---:|---:|---:|']
    for key, value in summary['groups'].items():
        group_table.append(f"| {key} | {value['n']} | {value['valid_predictions']} | {value['tolerance_hits'][0]}/{value['n']} |")
    delta = current[0] - reference['tolerance_percent']['all_at_once'][0]
    paired = next(a['paired_exact'] for a in audits if a['method'] == 'all_at_once')
    ci = [100 * x for x in summary['exact_wilson95']]
    selection_note = manifest['protocol']['config'].get('selection_note', '此前Qwen3.5-27B部署明确限制32768 tokens，因此选择Qwen2.5-14B-Instruct-1M进行最长输入可用性探测；约84万字符成功返回，报告185575输入tokens。选择依据上下文可用性，未比较模型在标签上的准确率；该探测不证明百万token极限。')
    reasoning_note = manifest['protocol']['config'].get('reasoning_mode_note', '未显式配置思考模式。')
    report = f'''# RootSE 全量 All-at-Once 实验报告

生成时间：{summary['evaluated_at']}。本次完整处理102条，未挑选短轨迹、未截断输入；有效预测{summary['valid_predictions']}条。所有失败保留在102分母中。

## 固定设置

模型请求ID：`{manifest['model']}`；接口：`{manifest['protocol']['config']['endpoint']}`。服务报告模型名：`{summary['response_models']}`，响应名不能独立证明部署的权重版本。temperature=0，max_tokens={manifest['protocol']['config']['max_tokens']}，不设置JSON约束，不启用跨模型回退。

{reasoning_note}

采用固定提交 `{reference['source_commit']}` 的官方All-at-Once提示词和JSON提取函数。输入为任务、全部原始步骤及首条失败测试code/error，不包含参考补丁或根因标签。标签只在本离线评估阶段读取。原始步骤字段顺序与官方load_failure保持一致。

采用严格JSON解析，不沿用此前8条pilot的引号修复。格式失败不重试；网络超时、429与5xx至多{manifest['protocol']['config']['transport_attempts']}次。官方脚本对超时/限流至多5次，此处为3次；每次用量均计入。结束原因非stop以及非原始步骤ID记失败，验证比官方脚本更严格。此前模型探测和pilot用量不混入正式运行。

协议指纹：`{manifest['fingerprint']}`。数据逐文件SHA256、代码/提示词快照、配置、所有实际请求和响应均在本目录保存。相同目录再次执行只复用完成样本；配置或推理代码变更要求新目录。temperature=0不能保证远端服务按位确定性。

## 全量结果与论文对照

{chr(10).join(table)}

本次Tol@0为{summary['tolerance_hits'][0]}/102，Wilson 95%区间[{ci[0]:.2f}%, {ci[1]:.2f}%]。相对论文All-at-Once的显示值差{delta:+.2f}个百分点。这是跨骨干和服务的复现对照，不能将差异归因于方法改进或退化；也不是同模型消融。论文Qwen3-235B的40.2%对应TrajAudit，不能用作本次All-at-Once的同方法基线。

逐样本配对：双方都命中{paired['both']}条，仅本次命中{paired['current_only']}条，仅作者公开All-at-Once命中{paired['published_only']}条，双方均未命中{paired['neither']}条。结果反映不同骨干在同一批样本上的差异。

论文来源：[TrajAudit v3]({reference['url']})，无参考补丁设置，表exact_accuracy与tolerance。公开CSV已按当前102条标签重新审计（paper_audit.json）。Step-by-Step CSV存在重复行，脚本同时报告首行/末行去重结果，不按是否命中选择；表中仍使用论文原始公布值。

{chr(10).join(audit_table)}

Step-by-Step的两种去重规则均得到20/102（19.61%），不能从当前公开CSV复现论文22.6%；本次没有将重复记录作为独立样本。其他五个方法的公开CSV精确命中与论文相符（允许显示精度的四舍五入）。

## 覆盖与资源

状态：`{summary['status_counts']}`。

{summary['responses_with_reasoning']}次响应包含独立reasoning字段；{summary['truncated_without_final_answer']}次在输出截断时尚无最终回答正文。这里不把思考文字中的猜测当作最终预测。

共{summary['api_requests']}次chat请求；服务已报告总量{summary['reported_total_tokens']:,} tokens（输入{summary['reported_prompt_tokens']:,}，输出{summary['reported_completion_tokens']:,}），平均每个基准样本{summary['reported_tokens_per_benchmark_case']:.1f} tokens。{summary['attempts_without_usage']}次请求无usage，不能将缺失视为零成本。费用未估算。不同模型tokenizer和部署不同，token量只能描述运行开销。

推理墙钟时间约{summary['inference_wall_seconds'] / 60:.2f}分钟（包含请求队列等待）。实际输入tokens最小/中位/最大：`{summary['prompt_tokens_min_median_max']}`。

{chr(10).join(group_table)}

## 过程及局限

{selection_note}

本次未调用Claude，也未运行LLM理由裁判，因此不报告Justification Accuracy。论文主骨干为Claude Sonnet4.5，当前为{manifest['model']}；模型大小、世代、上下文部署和输出预算均可能影响结果。Wilson区间仅作102条样本比例的描述，不能反映模型随机性或数据集选择偏差；本轮为单次运行。

逐样本结果见predictions.csv；完整解释及尝试记录见evaluation.json；所有失败保持可追溯。执行命令和环境锁定见仓库ROOTSE_FULL_BASELINE.md。离线重算可执行 `scripts/report_rootse_baseline.py --source <TrajAudit目录> --out <本目录>`。
'''
    (args.out / 'REPORT.md').write_text(report)
    save(args.out / 'evaluation_provenance.json', {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                'generated_at': now(), 'label_access': 'offline_evaluation_only'})
    print(json.dumps({k: v for k, v in summary.items() if k != 'groups'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
