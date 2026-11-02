#!/usr/bin/env python3
"""Exploratory decomposition of unit selection and local-window outcomes."""
from collections import Counter

from run_contextmap_units import OUT, SOURCE, METHODS, reconstruct
from run_contextmap import read
from suite_transport import save
from analyze_contextmap_units import complete_field, NAMES


def main() -> None:
    manifest = read(OUT / 'manifest.json')
    predictions = read(OUT / 'analysis/predictions.json')
    rows = []
    for c in manifest['cases']:
        cid = c['case_id']
        p = {m: predictions[m][cid] for m in METHODS}
        gold = p['units']['gold']
        views = {m: read(OUT / 'prepared' / m / f'{cid}.json')['view'] for m in METHODS}
        steps = reconstruct(read(SOURCE / 'prepared/full' / f'{cid}.json'))['steps']
        row = {'case_id': cid, 'trajectory_id': c['trajectory_id'], 'repository': c['repository'], 'gold': gold,
               'unit_delta': p['units']['top1'] - p['v1_repeat']['top1'],
               'window_delta': p['units_window']['top1'] - p['units']['top1'],
               'root_selection_change': p['units']['root_expanded'] - p['v1_repeat']['root_expanded'],
               'root_decision_change': p['units']['root_complete_decision'] - p['v1_repeat']['root_complete_decision'],
               'conditions': {m: {k: p[m][k] for k in ('prediction', 'top1', 'top5', 'late', 'early', 'status')} for m in METHODS}}
        old_prediction = p['v1_repeat']['prediction']
        new_prediction = p['units']['prediction']
        for label, prediction in [('repeat', old_prediction), ('units', new_prediction)]:
            wrong = prediction is not None and prediction != gold
            both = wrong and all(i in views['units']['selected_ids'] for i in (gold, prediction))
            together = both and any(gold in g and prediction in g for g in views['units_window']['presentation_windows'])
            row[label + '_wrong_both_expanded'] = bool(both)
            row[label + '_wrong_same_window'] = bool(together)
            for m in ('units', 'units_window'):
                order = views[m]['presentation_order']
                row[label + '_gold_before_prediction_' + m] = bool(both and order.index(gold) < order.index(prediction))
        row['root_fields'] = {}
        for m in METHODS:
            root = steps[int(gold)]
            row['root_fields'][m] = {field: {'available_chars': len(root[field]),
                'shown_chars': sum(f['end'] - f['start'] for f in views[m]['fragments'] if f['step_id'] == gold and f['field'] == field),
                'complete': complete_field(views[m], root, field)} for field in ('thought', 'response', 'action', 'observation')}
        rows.append(row)
    partitions = {}
    for field in ('root_selection_change', 'root_decision_change'):
        partitions[field] = {}
        for value in (-1, 0, 1):
            subset = [r for r in rows if r[field] == value]
            partitions[field][str(value)] = {'n': len(subset),
                'top1': {m: sum(r['conditions'][m]['top1'] for r in subset) for m in METHODS},
                'unit_gained': sum(r['unit_delta'] > 0 for r in subset),
                'unit_lost': sum(r['unit_delta'] < 0 for r in subset)}
    windows = {}
    for prefix in ('repeat', 'units'):
        both = [r for r in rows if r[prefix + '_wrong_both_expanded']]
        together = [r for r in both if r[prefix + '_wrong_same_window']]
        windows[prefix] = {'wrong_both_expanded': len(both), 'wrong_same_window': len(together),
            'gold_before_prediction_units': sum(r[prefix + '_gold_before_prediction_units'] for r in both),
            'gold_before_prediction_units_window': sum(r[prefix + '_gold_before_prediction_units_window'] for r in both),
            'same_window_unit_hits': sum(r['conditions']['units']['top1'] for r in together),
            'same_window_window_hits': sum(r['conditions']['units_window']['top1'] for r in together)}
    residual = {}
    for m in METHODS:
        wrong = [p for p in predictions[m].values() if not p['top1']]
        residual[m] = {'wrong_total': len(wrong), 'root_in_top2to5': sum(p['top5'] for p in wrong),
            'wrong_with_root_expanded': sum(p['root_expanded'] for p in wrong),
            'wrong_with_complete_root_decision': sum(p['root_complete_decision'] for p in wrong)}
    fields = {m: {field: {'available': sum(r['root_fields'][m][field]['available_chars'] > 0 for r in rows),
                         'complete': sum(r['root_fields'][m][field]['available_chars'] > 0 and r['root_fields'][m][field]['complete'] for r in rows)}
                  for field in ('thought', 'response', 'action', 'observation')} for m in METHODS}
    result = {'scope': 'exploratory mechanism audit, not confirmatory subgroup tests', 'partitions': partitions,
              'windows': windows, 'residual': residual, 'field_completeness': fields, 'rows': rows}
    save(OUT / 'analysis/mechanism_diagnostics.json', result)
    lines = ['# 数据调整与错误机制', '', '本页为冻结结果后的探索性分解；分组用于解释观察，不作新的显著性结论。', '',
             '|输入变化分组|数量|v1 重跑命中|完整单元命中|窗口命中|单元新增/丢失|', '|---|---:|---:|---:|---:|---:|']
    names = {'root_selection_change': '根因展开', 'root_decision_change': '根因完整决策'}
    for field, groups in partitions.items():
        for v, s in groups.items():
            lines.append(f"|{names[field]} { {'-1':'失去','0':'不变','1':'新增'}[v]}|{s['n']}|{s['top1']['v1_repeat']}|{s['top1']['units']}|{s['top1']['units_window']}|+{s['unit_gained']}/−{s['unit_lost']}|")
    lines += ['', '|方法|总未命中|根因在 2–5 名|已展开仍未命中|完整决策仍未命中|', '|---|---:|---:|---:|---:|']
    for m, s in residual.items():
        lines.append(f"|{NAMES[m]}|{s['wrong_total']}|{s['root_in_top2to5']}|{s['wrong_with_root_expanded']}|{s['wrong_with_complete_root_decision']}|")
    lines += ['', '关联窗口的直接覆盖：']
    for prefix, s in windows.items():
        lines += [f"- 相对 {prefix} 错误首选，根因与该候选均在新版展开的有 {s['wrong_both_expanded']} 例，其中真正被放入同一局部窗口的有 {s['wrong_same_window']} 例。根因展示在该候选之前：{s['gold_before_prediction_units']}→{s['gold_before_prediction_units_window']}。"]
    lines += ['', '字段完整覆盖计数（仅以对应原文字段非空为分母）：', '', '|字段|v1 重跑|完整单元|', '|---|---:|---:|']
    for field in fields['units']:
        a, b = fields['v1_repeat'][field], fields['units'][field]
        lines.append(f"|{field}|{a['complete']}/{a['available']}|{b['complete']}/{b['available']}|")
    (OUT / 'analysis/mechanism-report.md').write_text('\n'.join(lines) + '\n')
    print({'partitions': partitions, 'windows': windows, 'residual': residual})


if __name__ == '__main__':
    main()
