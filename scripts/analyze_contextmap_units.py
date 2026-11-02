#!/usr/bin/env python3
"""Audit frozen unit/window experiments and produce paired development diagnostics."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_contextmap import account, grouped
from analyze_contextmap_controls import verify_call
from run_contextmap_units import OUT, SOURCE, METHODS, prepare, reconstruct
from run_contextmap import read, sha, run_one
from suite_transport import digest, save
from loghound_agent_debug.context_map import tokens
from loghound_agent_debug.context_units import structural_units, build_unit_views

NAMES = {'v1_repeat': 'v1 同步重跑', 'units': '完整单元', 'units_window': '完整单元＋关联窗口',
         'attention': 'v1 历史 Attention', 'full': '历史全文', 'graph': '历史调用图', 'window': '历史检索窗口'}
ALL_METHODS = (*METHODS, 'attention', 'full', 'graph', 'window')
CONTRASTS = (('units', 'v1_repeat'), ('units_window', 'units'), ('units_window', 'v1_repeat'))


def complete_field(view: dict, step: dict, field: str) -> bool:
    ranges = sorted((f['start'], f['end']) for f in view['fragments']
                    if f['step_id'] == step['id'] and f['field'] == field)
    position = 0
    for start, end in ranges:
        if start > position:
            return False
        position = max(position, end)
    return position == len(step[field])


def data_audit(out: Path = OUT) -> dict:
    """Inspect all request construction without accessing labels or answers."""
    manifest = prepare(out)
    rows = []
    for case in manifest['cases']:
        cid = case['case_id']
        original = read(SOURCE / 'prepared/attention' / f'{cid}.json')
        full = read(SOURCE / 'prepared/full' / f'{cid}.json')
        safe = reconstruct(full)
        mapping = read(SOURCE / 'maps' / f'{cid}.json')
        prepared = {m: read(out / 'prepared' / m / f'{cid}.json') for m in METHODS}
        if prepared['v1_repeat']['messages'] != original['messages'] or prepared['v1_repeat']['view'] != original['view']:
            raise ValueError('v1 repeat drift')
        if prepared['units']['view']['fragments'] != prepared['units_window']['view']['fragments']:
            raise ValueError('Window changes facts')
        if prepared['units']['view']['selected_ids'] != prepared['units_window']['view']['selected_ids']:
            raise ValueError('Window changes selected nodes')
        rebuilt = build_unit_views(safe, mapping, original['view'])
        for method, packet in prepared.items():
            view = packet['view']
            if method != 'v1_repeat' and view != rebuilt[method]:
                raise ValueError('Unit view cannot be reconstructed')
            expected = [dict(m) for m in original['messages']]
            expected[1]['content'] = expected[1]['content'].replace(original['view']['content'], view['content'], 1)
            if packet['messages'] != expected or digest(expected) != packet['messages_sha256']:
                raise ValueError('Non-context instructions changed')
            if tokens(view['content']) != view['context_tokens'] or view['context_tokens'] > 16384:
                raise ValueError('Context budget mismatch')
            if set(view['presentation_order']) != set(view['selected_ids']) or len(view['presentation_order']) != len(view['selected_ids']):
                raise ValueError('Missing/duplicated presentation node')
            for f in view['fragments']:
                source = safe['steps'][int(f['step_id'])][f['field']]
                if f['text'] != source[f['start']:f['end']]:
                    raise ValueError('Non-original evidence')
                if method != 'v1_repeat' and (f['start'], f['end']) not in {(a, b) for a, b, _ in structural_units(source)}:
                    raise ValueError('Split atomic unit')
            decisions = [s for s in safe['steps'] if s['id'] in view['selected_ids'] and (s['thought'] or s['response'])]
            row = {'method': method, 'case_id': cid, 'selected_nodes': len(view['selected_ids']),
                   'context_tokens_proxy': view['context_tokens'], 'selected_nodes_with_decision': len(decisions),
                   'selected_complete_decisions': sum(all(complete_field(view, s, f) for f in ('thought', 'response')) for s in decisions)}
            rows.append(row)
        links = lambda v: sorted(l for l in v['content'].splitlines() if l.startswith('Reading links from step '))
        if links(prepared['units']['view']) != links(prepared['units_window']['view']):
            raise ValueError('Window changes relation hints')
    result = {'cases': manifest['n'], 'verified_requests': len(rows), 'label_or_answer_reads': 0,
              'frozen_fingerprint': manifest['fingerprint'], 'rows': rows,
              'methods': {m: {k: sum(r[k] for r in rows if r['method'] == m)
                              for k in ('selected_nodes', 'context_tokens_proxy', 'selected_nodes_with_decision', 'selected_complete_decisions')}
                          for m in METHODS}}
    save(out / 'input_audit.json', result)
    return result


def pair(a: dict, b: dict, cases: dict) -> dict:
    values = {cid: a[cid]['top1'] - b[cid]['top1'] for cid in cases}
    return {**grouped(values, cases, paired=True),
            'gained_ids': [cid for cid, v in values.items() if v > 0],
            'lost_ids': [cid for cid, v in values.items() if v < 0],
            'gained': sum(v > 0 for v in values.values()), 'lost': sum(v < 0 for v in values.values())}


def analyze(out: Path = OUT) -> dict:
    manifest = prepare(out)
    gate = read(out / 'run_status.json')
    if gate['fingerprint'] != manifest['fingerprint'] or gate['completed'] != 3 * manifest['n']:
        raise ValueError('Experiment incomplete')
    cases = {c['case_id']: c for c in manifest['cases']}
    baseline = read(SOURCE / 'manifest.json')
    if sha(SOURCE / 'labels.json') != baseline['prepared_hashes']['labels.json']:
        raise ValueError('Original labels changed')
    labels = read(SOURCE / 'labels.json')
    historical = read(out / 'historical_hashes.json')
    for p, expected in historical.items():
        if sha(Path(p)) != expected:
            raise ValueError('Historical run modified: ' + p)
    rows, predictions, checks, hashes = [], {}, Counter(), {}
    sources = {cid: reconstruct(read(SOURCE / 'prepared/full' / f'{cid}.json')) for cid in cases}
    for method in ALL_METHODS:
        origin, frozen = (out, manifest) if method in METHODS else (SOURCE, baseline)
        predictions[method] = {}
        for cid, case in cases.items():
            prepared_path = origin / 'prepared' / method / f'{cid}.json'
            if sha(prepared_path) != frozen['prepared_hashes'][str(prepared_path.relative_to(origin))]:
                raise ValueError('Prepared input hash changed')
            packet = read(prepared_path)
            directory = origin / 'runs' / method / cid
            state, call, verified = verify_call(directory, packet, frozen)
            checks.update(verified)
            ranking = state['decision']['ranked_step_ids'] if state['status'] == 'valid' else []
            gold = labels[cid][0]
            rank = ranking.index(gold) + 1 if gold in ranking else None
            offset = int(ranking[0]) - int(gold) if ranking else None
            step, view = sources[cid]['steps'][int(gold)], packet['view']
            has_decision = bool(step['thought'] or step['response'])
            expanded = gold in view['selected_ids']
            row = {'method': method, 'case_id': cid, 'group': case['group'], 'repository': case['repository'],
                   'trajectory_id': case['trajectory_id'], 'status': state['status'], 'gold': gold,
                   'prediction': ranking[0] if ranking else None, 'ranked_step_ids': ranking,
                   'top1': int(rank == 1), 'top5': int(rank is not None), 'mrr_at5': 1 / rank if rank else 0.,
                   'offset': offset, 'late': int(offset is not None and offset > 0), 'early': int(offset is not None and offset < 0),
                   'root_expanded': int(expanded), 'root_has_decision': int(has_decision),
                   'root_complete_decision': int(has_decision and expanded and all(complete_field(view, step, f) for f in ('thought', 'response'))),
                   'selected_nodes': len(view['selected_ids']), 'context_tokens_proxy': view['context_tokens'],
                   'reason': state.get('decision', {}).get('reason'), 'started_at': state['started_at'], 'finished_at': state['finished_at'],
                   'requests': len(call['attempts']),
                   'transport_seconds': sum(a.get('elapsed_seconds', 0.) for a in call['attempts'])}
            predictions[method][cid] = row
            rows.append(row)
            if method in METHODS:
                for p in directory.rglob('*'):
                    if p.is_file():
                        hashes[str(p.relative_to(out))] = sha(p)
    actual_counts = Counter(r['status'] for r in rows if r['method'] in METHODS)
    if dict(actual_counts) != gate['status_counts']:
        raise ValueError('Run status counts differ')
    summaries = {}
    for method in ALL_METHODS:
        values = list(predictions[method].values())
        summaries[method] = {'n': len(values), 'valid': sum(r['status'] == 'valid' for r in values),
            **{k: sum(r[k] for r in values) for k in ('top1', 'top5', 'late', 'early', 'root_expanded', 'root_has_decision', 'root_complete_decision')},
            'metrics': {k: grouped({r['case_id']: r[k] for r in values}, cases) for k in ('top1', 'top5', 'mrr_at5')},
            'expanded_hits': sum(r['root_expanded'] * r['top1'] for r in values),
            'expanded_late': sum(r['root_expanded'] * r['late'] for r in values),
            'status_counts': dict(Counter(r['status'] for r in values)),
            'mean_selected_nodes': float(np.mean([r['selected_nodes'] for r in values])),
            'transport_seconds_median': float(np.median([r['transport_seconds'] for r in values])),
            'transport_seconds_p90': float(np.quantile([r['transport_seconds'] for r in values], .9)),
            'cost': account((out if method in METHODS else SOURCE) / 'runs' / method),
            'started_at': min(r['started_at'] for r in values), 'finished_at': max(r['finished_at'] for r in values)}
    contrasts = {f'{a}_minus_{b}': {'a': a, 'b': b, **pair(predictions[a], predictions[b], cases)} for a, b in CONTRASTS}
    running = 0.
    for i, name in enumerate(sorted(contrasts, key=lambda k: contrasts[k]['p'])):
        running = max(running, min(1., (len(contrasts) - i) * contrasts[name]['p']))
        contrasts[name]['p_holm'] = running
    complete = {cid: c for cid, c in cases.items() if all(predictions[m][cid]['status'] == 'valid' for m in METHODS)}
    sensitivity = {'n': len(complete), 'excluded_ids': sorted(set(cases) - set(complete)),
                   'top1': {m: sum(predictions[m][cid]['top1'] for cid in complete) for m in METHODS},
                   'contrasts': {f'{a}_minus_{b}': pair(predictions[a], predictions[b], complete) for a, b in CONTRASTS} if complete else {}}
    result = {'n': len(cases), 'groups': len({c['group'] for c in cases.values()}), 'runs_per_new_condition': 1,
              'fingerprint': manifest['fingerprint'], 'methods': summaries, 'contrasts': contrasts,
              'repeat_v1_diagnostic': pair(predictions['v1_repeat'], predictions['attention'], cases),
              'complete_case_sensitivity': sensitivity, 'new_diagnostic_cost': account(out / 'runs'),
              'input_audit': read(out / 'input_audit.json')['methods']}
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    save(analysis / 'summary.json', result)
    save(analysis / 'predictions.json', predictions)
    save(analysis / 'raw_hashes.json', hashes)
    changed = sorted({cid for c in contrasts.values() for key in ('gained_ids', 'lost_ids') for cid in c[key]})
    packets = {}
    for cid in changed:
        gold = labels[cid][0]
        records = {m: read(out / 'prepared' / m / f'{cid}.json')['view'] for m in METHODS}
        packets[cid] = {'case': cases[cid], 'gold': gold,
                        'original_root': sources[cid]['steps'][int(gold)],
                        'conditions': {m: {'prediction': predictions[m][cid],
                            'root_position': records[m]['presentation_order'].index(gold) if gold in records[m]['presentation_order'] else None,
                            'root_fragments': [f for f in records[m]['fragments'] if f['step_id'] == gold],
                            'selected_ids': records[m]['selected_ids'], 'presentation_order': records[m]['presentation_order'],
                            'windows': records[m].get('presentation_windows', [])} for m in METHODS}}
    save(analysis / 'case_packets.json', packets)
    save(analysis / 'verification.json', {**dict(checks), 'historical_files_unchanged': len(historical),
                                        'frozen_new_requests': manifest['n'] * 3, 'same_fact_window_pairs': manifest['n']})
    with (analysis / 'paired_cases.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(result, analysis)
    figures(result, analysis)
    return result


def report(result: dict, out: Path) -> None:
    lines = ['# 完整单元与关联窗口实验结果', '',
             f"102 条既有开发轨迹，{result['groups']} 个仓库/任务联合组；三组共 306 项任务，v1 同步原样重跑。主指标所有轨迹为分母，无效结果计 0。", '',
             '|方法|Top-1|Top-5|MRR@5|晚判|早判|无效|根因展开|根因完整决策|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, s in result['methods'].items():
        lines.append(f"|{NAMES[m]}|{s['top1']}/102|{s['top5']}/102|{s['metrics']['mrr_at5']['mean']:.4f}|{s['late']}|{s['early']}|{102-s['valid']}|{s['root_expanded']}|{s['root_complete_decision']}/{s['root_has_decision']}|")
    lines += ['', '根因完整决策指有非空 thought/response 的标注节点，其这两个字段均被完整展示；分母固定为有决策文字的根因数，未展开计未覆盖。', '',
              '|同步配对比较|Top-1 差值|95% 组区间|新增/丢失|Holm p|', '|---|---:|---:|---:|---:|']
    for c in result['contrasts'].values():
        lines.append(f"|{NAMES[c['a']]} − {NAMES[c['b']]}|{100*c['mean']:+.2f} pp|[{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] pp|+{c['gained']}/−{c['lost']}|{c['p_holm']:.4f}|")
    repeat = result['repeat_v1_diagnostic']
    lines += ['', f"v1 同一请求重跑相对历史记录：新增 {repeat['gained']}、丢失 {repeat['lost']}，净差 {100*repeat['mean']:+.2f} pp。该项用于描述服务重复差异，方法改进以同步比较为主。", '',
              f"三组共同有效 {result['complete_case_sensitivity']['n']} 条，Top-1：{result['complete_case_sensitivity']['top1']}。完整敏感性配对结果见 summary.json。", '',
              '|同步条件|输入 token|输出 token|已知总 token|未知用量请求|', '|---|---:|---:|---:|---:|']
    for m in METHODS:
        c = result['methods'][m]['cost']
        lines.append(f"|{NAMES[m]}|{c.get('prompt_tokens',0):,}|{c.get('completion_tokens',0):,}|{c.get('total_tokens',0):,}|{c.get('unknown_usage_requests',0)}|")
    lines += ['', '费用统计为已有图缓存下的新诊断用量；沿用历史昂贵构图，不视为从零端到端成本。所有原始输入、响应和失败保留。', '',
              '完整单元改变节点内预算分配，可能改变选中节点；同事实关联窗口只改变局部顺序。窗口来自现有阅读关系，未自动标注提出/实现/纠正角色。历史全文与图基线运行时间不同，仅作背景。', '',
              '## Claim Candidates', '']
    for c in result['contrasts'].values():
        lines += [f"- Claim: 本开发集 {NAMES[c['a']]} 相对 {NAMES[c['b']]} 的 Top-1 观测差值 {100*c['mean']:+.2f} pp。",
                  '  - Source evidence: summary.json、paired_cases.csv、完整冻结请求及原始 SSE。',
                  '  - Allowed wording: 描述观测差值、胜负案例和组配对区间。',
                  '  - Forbidden stronger wording: 单次开发运行不支持普适、稳定或单一机制因果提升。',
                  f"  - Uncertainty: CI={c['ci95']}，Holm p={c['p_holm']:.4f}；每条件一次。",
                  '  - Next check: 根据节点覆盖与错例区分信息遗漏和证据利用，再冻结方案到独立轨迹。',
                  '  - Decision: ' + ('weaken' if c['ci95'][0] <= 0 <= c['ci95'][1] else 'keep')]
    (out / 'analysis-report.md').write_text('\n'.join(lines) + '\n')
    (out / 'stats-appendix.md').write_text(
        '# 统计附录\n\nTop-1 为主指标；Top-5、MRR@5、方向和覆盖为辅助描述。102 条开发轨迹配对，35 个仓库/任务联合组用于相关性处理。'
        '三组在一次随机交错调度中运行。按输入长度选 2 个 pilot，结果直接复用，其 6 请求在主队列之前。每条件一次，不报告虚构的跨运行标准差。\n\n'
        '主比较在运行前指定为 units−v1_repeat、units_window−units、units_window−v1_repeat；统一 Holm 校正。'
        '沿用 v1 grouped：5000 次整组 bootstrap，按抽中轨迹总数加权；20000 次组内整体交换方法标签，双侧 p 加一校正；随机种子 20261028。'
        '二元变量直接使用非参数方法，不依赖正态假设。区间反映组抽样变化，不包含完整服务随机性；组独立与零假设下标签可交换是检验条件。'
        '同批随机交错减少时间混淆，但既有开发集与单次调用仍不构成确认性检验。\n\n'
        '失败全保留且主结果计 0；三组共同有效交集作为敏感性统计。单次历史 full/graph/window/attention 仅背景对照。'
        '重复 v1 的胜负是复现敏感性描述，不计入三个方法改进比较。根因完整决策固定分母，不仅在已展开节点内计算，避免覆盖率变化被隐藏。\n')


def figures(result: dict, out: Path) -> None:
    directory = out / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    labels = ['v1 repeat', 'Atomic units', 'Units + windows']
    means = np.array([result['methods'][m]['metrics']['top1']['mean'] for m in METHODS]) * 100
    intervals = np.array([result['methods'][m]['metrics']['top1']['ci95'] for m in METHODS]) * 100
    fig, ax = plt.subplots(figsize=(7.8, 4.6), layout='constrained')
    ax.bar(range(3), means, color=['#7C8C98', '#377FA3', '#3A927E'], width=.6)
    ax.errorbar(range(3), means, yerr=np.maximum(0, [means-intervals[:, 0], intervals[:, 1]-means]),
                fmt='none', capsize=4, color='#333')
    for i, m in enumerate(METHODS):
        ax.text(i, intervals[i, 1]+2, f"{result['methods'][m]['top1']}/102", ha='center')
    ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 100), ylabel='Top-1 accuracy (%)',
           title='Matched 16k context budget; same-batch diagnosis')
    for ext in ('png', 'pdf'): fig.savefig(directory / f'01-top1.{ext}', dpi=180)
    plt.close(fig)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    left = np.zeros(3)
    for key, name, color in [('top1', 'Correct', '#3A927E'), ('late', 'Late', '#C97B56'), ('early', 'Early', '#9074AC')]:
        values = np.array([result['methods'][m][key] for m in METHODS])
        bars = ax.barh(range(3), values, left=left, label=name, color=color)
        ax.bar_label(bars, labels=[str(v) if v else '' for v in values], label_type='center', color='white')
        left += values
    if (left < 102).any(): ax.barh(range(3), 102-left, left=left, color='#AAA', label='Invalid')
    ax.set(yticks=range(3), yticklabels=labels, xlim=(0, 102), xlabel='Trajectories', title='Error direction')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2)
    for offset, key, name, color in [(-.18, 'root_expanded', 'Root expanded', '#7C8C98'), (.18, 'root_complete_decision', 'Complete decision', '#377FA3')]:
        bars = bx.bar(np.arange(3)+offset, [result['methods'][m][key] for m in METHODS], .35, label=name, color=color)
        bx.bar_label(bars)
    bx.set(xticks=range(3), xticklabels=labels, ylim=(0, 112), ylabel='Count (of 102)', title='Evidence coverage')
    bx.legend(loc='upper center', bbox_to_anchor=(.5, -.15))
    for ext in ('png', 'pdf'): fig.savefig(directory / f'02-direction-coverage.{ext}', dpi=180)
    plt.close(fig)
    (out / 'figure-catalog.md').write_text(
        '# 图表目录\n\n01-top1.png/pdf：同批三条件 Top-1，误差线为 5000 次组 bootstrap 的95%区间；柱标精确命中数。'
        '比较的是实际诊断效果，方法差异须读配对区间，不能仅比较柱高。用于决定是否保留完整单元与局部窗口。\n\n'
        '02-direction-coverage.png/pdf：左为正确/晚判/早判/无效完整计数，右为根因展开和非空根因决策全文覆盖；无误差线。'
        '用于判断收益是否只是把预测提前、是否以丢失根因为代价增加决策完整性。源数据均为 summary.json 和 paired_cases.csv。'
        '两图均为 102 条开发轨迹、每条件一次，不代表独立测试或跨运行稳定性。\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-only', action='store_true')
    args = parser.parse_args()
    if args.input_only:
        r = data_audit()
        print({k: v for k, v in r.items() if k != 'rows'})
    else:
        r = analyze()
        print({'top1': {m: s['top1'] for m, s in r['methods'].items()}, 'cost': r['new_diagnostic_cost']})
