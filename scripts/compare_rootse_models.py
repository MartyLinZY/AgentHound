#!/usr/bin/env python3
"""Compare two frozen full-set runs after checking identical inputs except model."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from run_qwen_baseline import save


def paired_counts(left: list[bool], right: list[bool]) -> dict:
    if len(left) != len(right):
        raise ValueError('unpaired results')
    both = sum(a and b for a, b in zip(left, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    neither = len(left) - both - left_only - right_only
    n = left_only + right_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / 2 ** n) if n else 1.0
    return {'both_correct': both, 'left_only': left_only, 'right_only': right_only,
            'neither_correct': neither, 'exact_mcnemar_two_sided_p': p}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    args = p.parse_args()
    dirs = [args.baseline, args.candidate]
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in dirs]
    metrics = [json.loads((d / 'metrics.json').read_text()) for d in dirs]
    cases = [{r['trajectory_id']: r for r in json.loads((d / 'evaluation.json').read_text())} for d in dirs]
    if set(cases[0]) != set(cases[1]) or len(cases[0]) != 102:
        raise ValueError('comparison requires the same 102 unique trajectories')
    for key in ['dataset_hash', 'prompt_hashes', 'code_hashes']:
        if manifests[0]['protocol'][key] != manifests[1]['protocol'][key]:
            raise ValueError('inference protocol differs: ' + key)
    for key in ['temperature', 'max_tokens', 'context_policy', 'parser']:
        if manifests[0]['protocol']['config'][key] != manifests[1]['protocol']['config'][key]:
            raise ValueError('settings differ: ' + key)
    ids = sorted(cases[0])
    rows = []
    for tid in ids:
        pair = [c[tid] for c in cases]
        requests = [json.loads((d / r['request_file']).read_text()) for d, r in zip(dirs, pair)]
        for request in requests:
            request.pop('model')
        if requests[0] != requests[1]:
            raise ValueError('input differs beyond model: ' + tid)
        if pair[0]['gold_step'] != pair[1]['gold_step']:
            raise ValueError('labels differ')
        rows.append({'trajectory_id': tid, 'gold_step': pair[0]['gold_step'],
                     'baseline_step': pair[0].get('failure_id'), 'candidate_step': pair[1].get('failure_id'),
                     'baseline_status': pair[0]['status'], 'candidate_status': pair[1]['status'],
                     'baseline_hit': pair[0]['tolerance_hits'][0], 'candidate_hit': pair[1]['tolerance_hits'][0]})
    paired = paired_counts([r['baseline_hit'] for r in rows], [r['candidate_hit'] for r in rows])
    result = {'baseline_model': metrics[0]['model'], 'candidate_model': metrics[1]['model'], 'n': 102,
              'identical_request_payloads_except_model': True, 'paired_exact': paired,
              'baseline_hits': metrics[0]['tolerance_hits'], 'candidate_hits': metrics[1]['tolerance_hits'],
              'delta_percentage_points': [(b - a) * 100 / 102 for a, b in zip(metrics[0]['tolerance_hits'], metrics[1]['tolerance_hits'])],
              'baseline_tokens': metrics[0]['reported_total_tokens'], 'candidate_tokens': metrics[1]['reported_total_tokens']}
    result['baseline_only_candidate_status'] = dict(Counter(r['candidate_status'] for r in rows if r['baseline_hit'] and not r['candidate_hit']))
    save(args.candidate / 'MODEL_COMPARISON.json', result)
    with (args.candidate / 'model_comparison_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ['| 模型 | Tol@0 | Tol@1 | Tol@2 | Tol@3 | 有效输出 | 报告tokens |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        table.append('| ' + m['model'] + ' | ' + ' | '.join(f'{v * 100:.2f}% ({h}/102)' for v, h in zip(m['tolerance_accuracy'], m['tolerance_hits'])) + f" | {m['valid_predictions']}/102 | {m['reported_total_tokens']:,} |")
    text = f'''# 强模型与14B的完整RootSE对照

两轮均为All-at-Once、without-reference、完整102条，不截断轨迹。脚本已逐条确认请求体除model外完全一致，数据、推理代码、提示词哈希及解析规则一致。

{chr(10).join(table)}

候选相对14B的Tol@0变化：{result['delta_percentage_points'][0]:+.2f}个百分点。共同命中{paired['both_correct']}条，只有14B命中{paired['left_only']}条，只有候选命中{paired['right_only']}条，共同未命中{paired['neither_correct']}条。逐条见model_comparison_cases.csv。

在仅14B命中的案例中，候选状态为`{result['baseline_only_candidate_status']}`，用于区分定位错误与未完成输出。

精确双侧McNemar p={paired['exact_mcnemar_two_sided_p']:.6g}，仅描述本102条配对差异；同仓库/任务相关性、单次运行与模型服务的随机性限制其外推能力。不将模型升级后的增益说成诊断算法的改进。

按常用0.05阈值，本轮差异{'达到' if paired['exact_mcnemar_two_sided_p'] < 0.05 else '未达到'}该检验的显著性标准。观察到的分数差与统计证据强度应分开解读。

候选Qwen3.5-397B-A17B默认思考，服务探测返回独立reasoning字段；14B为Instruct部署。虽然输出预算同为8192，模型架构、世代、tokenizer与实际思考开销并未控制，因此这不是只改变参数规模的消融。若有输出截断、格式失败或服务错误，一律留在102分母中。

论文All-at-Once/Claude Sonnet4.5为25/102（24.5%）。两轮与论文比较均存在骨干差异；完整论文表格和CSV核对见同目录REPORT.md。模型探测费用不混入正式运行。

复现：在仓库执行 `bash scripts/reproduce_rootse_strong.sh /path/to/moma_ai.key`。已有结果自动续跑，不重复调用已完成样本；缺少14B基准结果时会先执行14B全量实验。
'''
    (args.candidate / 'MODEL_COMPARISON.md').write_text(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
