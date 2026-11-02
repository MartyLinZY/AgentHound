#!/usr/bin/env python3
"""Audit four presentation cells and report paired development-set diagnostics."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_contextmap import account, grouped, read_sse
from run_contextmap_controls import OUT, SOURCE, prepare, METHODS as NEW_METHODS
from run_contextmap import read, sha
from suite_transport import digest, save
from loghound_agent_debug.context_map import parse_answer

METHODS = ('attention_flat', 'chronological_links', 'priority_without_links', 'attention')
NAMES = {'attention_flat': '时间顺序／无关系', 'chronological_links': '时间顺序／有关系',
         'priority_without_links': '关注顺序／无关系', 'attention': '关注顺序／有关系'}
ENGLISH = ('Chronological\nNo links (v1)', 'Chronological\nLinks (v2)',
           'Priority\nNo links (v2)', 'Priority\nLinks (v1)')
CONTRASTS = {
    'order_with_links': ('attention', 'chronological_links', '有关系：关注序 − 时间序'),
    'order_without_links': ('priority_without_links', 'attention_flat', '无关系：关注序 − 时间序'),
    'links_chronological': ('chronological_links', 'attention_flat', '时间序：有关系 − 无关系'),
    'links_priority': ('attention', 'priority_without_links', '关注序：有关系 − 无关系'),
}


def verify_call(directory: Path, prepared: dict, manifest: dict) -> tuple[dict, dict, dict]:
    state = read(directory / 'state.json')
    expected_signature = digest({'experiment': manifest['fingerprint'], 'messages': prepared['messages_sha256']})
    if state['signature'] != expected_signature or state['status'] == 'running':
        raise ValueError('Incomplete or mismatched result')
    if digest(prepared['messages']) != prepared['messages_sha256']:
        raise ValueError('Prepared messages hash mismatch')
    call = directory / 'calls/0001'
    request, callstate = read(call / 'request.json'), read(call / 'state.json')
    cfg = manifest['config']
    expected_request = {'model': cfg['model'], 'messages': prepared['messages'],
                        'temperature': cfg['temperature'], 'max_tokens': cfg['max_tokens'],
                        'stream': True, 'stream_options': {'include_usage': True},
                        'enable_thinking': cfg['enable_thinking']}
    if request != expected_request or digest(request) != callstate['request_sha256']:
        raise ValueError('Actual payload mismatch')
    verified = {'request': 1, 'sse': 0, 'parsed': 0}
    if callstate['status'] == 'received':
        responsepath = call / callstate['response_file']
        response = read(responsepath)
        if response['model'] != cfg['model']:
            raise ValueError('Response model mismatch')
        message = response['choices'][0]['message']
        content, reasoning = read_sse(responsepath.with_suffix('.sse'))
        if content != (message.get('content') or '') or reasoning != (message.get('reasoning_content') or ''):
            raise ValueError('SSE differs from response JSON')
        verified['sse'] = 1
        if state['status'] == 'valid':
            if response['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('Valid state has unfinished generation')
            if parse_answer(content, prepared['step_ids']) != state['decision']:
                raise ValueError('Decision differs from raw response')
            verified['parsed'] = 1
    elif state['status'] == 'valid':
        raise ValueError('Valid result has no received response')
    return state, callstate, verified


def analyze(out: Path = OUT, source: Path = SOURCE) -> dict:
    manifest = prepare(out)
    gate = read(out / 'run_status.json')
    if gate['fingerprint'] != manifest['fingerprint'] or gate['planned'] != 2 * manifest['n']:
        raise ValueError('Run status mismatch')
    if sum(gate['status_counts'].values()) != gate['planned']:
        raise ValueError('Incomplete experiment')
    baseline = read(source / 'manifest.json')
    labels = read(source / 'labels.json')
    if sha(source / 'labels.json') != baseline['prepared_hashes']['labels.json']:
        raise ValueError('Frozen labels changed')
    historical = read(source / 'analysis/raw_hashes.json')
    for path, expected in historical.items():
        if sha(source / path) != expected:
            raise ValueError('Historical response changed: ' + path)
    cases = {c['case_id']: c for c in manifest['cases']}
    if cases != {c['case_id']: c for c in baseline['cases']}:
        raise ValueError('Case population changed')
    rows, predictions, counters, raw_hashes = [], {}, Counter(), {}
    for method in METHODS:
        origin = out if method in NEW_METHODS else source
        frozen = manifest if method in NEW_METHODS else baseline
        predictions[method] = {}
        for cid, case in cases.items():
            prepared_path = origin / 'prepared' / method / f'{cid}.json'
            if sha(prepared_path) != frozen['prepared_hashes'][str(prepared_path.relative_to(origin))]:
                raise ValueError('Prepared evidence changed')
            prepared = read(prepared_path)
            directory = origin / 'runs' / method / cid
            state, call, checked = verify_call(directory, prepared, frozen)
            counters.update(checked)
            ranking = state['decision']['ranked_step_ids'] if state['status'] == 'valid' else []
            gold = labels[cid]
            if len(gold) != 1:
                raise ValueError('Temporal analysis requires one gold step')
            rank = min((ranking.index(g) + 1 for g in gold if g in ranking), default=None)
            ordinal = {step: i for i, step in enumerate(prepared['step_ids'])}
            offset = ordinal[ranking[0]] - ordinal[gold[0]] if ranking else None
            usage = Counter()
            for attempt in call['attempts']:
                if attempt.get('usage'):
                    usage.update({k: attempt['usage'].get(k, 0) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
            row = {'method': method, 'case_id': cid, 'group': case['group'], 'status': state['status'],
                   'gold': gold[0], 'prediction': ranking[0] if ranking else None, 'ranked_step_ids': ranking,
                   'top1': int(rank == 1), 'top5': int(rank is not None), 'mrr_at5': 1 / rank if rank else 0.,
                   'offset': offset, 'late': int(offset is not None and offset > 0),
                   'early': int(offset is not None and offset < 0),
                   'root_expanded': int(gold[0] in prepared['view']['selected_ids']),
                   'context_tokens_proxy': prepared['view']['context_tokens'],
                   'requests': len(call['attempts']), 'started_at': state['started_at'],
                   'finished_at': state['finished_at'], **dict(usage)}
            rows.append(row)
            predictions[method][cid] = row
            if method in NEW_METHODS:
                for path in directory.rglob('*'):
                    if path.is_file():
                        raw_hashes[str(path.relative_to(out))] = sha(path)
    for cid in cases:
        views = [read((out if m in NEW_METHODS else source) / 'prepared' / m / f'{cid}.json')['view'] for m in METHODS]
        if any(v['fragments'] != views[0]['fragments'] or set(v['selected_ids']) != set(views[0]['selected_ids']) for v in views):
            raise ValueError('Four-cell facts differ')
    actual_counts = Counter(r['status'] for r in rows if r['method'] in NEW_METHODS)
    if dict(actual_counts) != gate['status_counts']:
        raise ValueError('Completion status differs from actual case states')
    summaries = {}
    for method in METHODS:
        selected = list(predictions[method].values())
        summaries[method] = {
            'n': len(selected), 'valid': sum(r['status'] == 'valid' for r in selected),
            'hits': sum(r['top1'] for r in selected), 'top5_hits': sum(r['top5'] for r in selected),
            'late': sum(r['late'] for r in selected), 'early': sum(r['early'] for r in selected),
            'metrics': {metric: grouped({r['case_id']: r[metric] for r in selected}, cases)
                        for metric in ('top1', 'top5', 'mrr_at5')},
            'status_counts': dict(Counter(r['status'] for r in selected)),
            'expanded_n': sum(r['root_expanded'] for r in selected),
            'expanded_hits': sum(r['top1'] * r['root_expanded'] for r in selected),
            'expanded_late': sum(r['late'] * r['root_expanded'] for r in selected),
            'cost': account((out if method in NEW_METHODS else source) / 'runs' / method),
            'started_at': min(r['started_at'] for r in selected), 'finished_at': max(r['finished_at'] for r in selected)}
    contrasts = {}
    for name, (a, b, title) in CONTRASTS.items():
        left, right = predictions[a], predictions[b]
        values = {cid: left[cid]['top1'] - right[cid]['top1'] for cid in cases}
        contrasts[name] = {**grouped(values, cases, paired=True), 'title': title, 'a': a, 'b': b,
                           'gained': sum(v > 0 for v in values.values()), 'lost': sum(v < 0 for v in values.values()),
                           'gained_ids': [cid for cid, v in values.items() if v > 0],
                           'lost_ids': [cid for cid, v in values.items() if v < 0]}
    running = 0.
    for i, name in enumerate(sorted(contrasts, key=lambda k: contrasts[k]['p'])):
        running = max(running, min(1., (len(contrasts) - i) * contrasts[name]['p']))
        contrasts[name]['p_holm'] = running
    interaction = grouped({cid: predictions['attention'][cid]['top1'] - predictions['chronological_links'][cid]['top1']
                           - predictions['priority_without_links'][cid]['top1'] + predictions['attention_flat'][cid]['top1']
                           for cid in cases}, cases)
    complete = [cid for cid in cases if all(predictions[m][cid]['status'] == 'valid' for m in METHODS)]
    sensitivity = {'n': len(complete), 'excluded_ids': [cid for cid in cases if cid not in complete],
                   'hits': {m: sum(predictions[m][cid]['top1'] for cid in complete) for m in METHODS}}
    result = {'experiment': manifest['fingerprint'], 'n': len(cases), 'groups': len({c['group'] for c in cases.values()}),
              'runs_per_method': 1, 'methods': summaries, 'contrasts': contrasts, 'interaction': interaction,
              'complete_case_sensitivity': sensitivity,
              'new_diagnostic_cost': account(out / 'runs'), 'rejected_preflight_cost': account(out / 'preflight_rejected'),
              'caveat': 'Development diagnostic only: two v1 cells and two v2 cells were run in different batches.'}
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    save(analysis / 'summary.json', result)
    save(analysis / 'predictions.json', predictions)
    save(analysis / 'raw_hashes.json', raw_hashes)
    save(analysis / 'verification.json', {**dict(counters), 'same_fact_groups': len(cases),
        'v1_run_files_unchanged': len(historical), 'v2_source_hashes_checked': len(manifest['source_hashes']),
        'v2_prepared_hashes_checked': len(manifest['prepared_hashes']), 'analysis_script_sha256': sha(Path(__file__))})
    fields = sorted({k for r in rows for k in r})
    with (analysis / 'paired_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows)
    reports(result, analysis)
    figures(result, analysis)
    return result


def reports(result: dict, out: Path) -> None:
    lines = ['# Attention v2 呈现对照结果', '',
        f"102 条 RootSE 开发轨迹、{result['groups']} 个仓库/任务组，每种呈现一次服务运行。新增两格与 v1 两格共用原文片段、诊断提示词和模型配置。", '',
        '| 顺序／关系 | 来源 | 有效 | Top-1 | Top-5 | MRR@5 | 晚判 | 早判 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        s = result['methods'][m]
        lines.append(f"| {NAMES[m]} | {'本轮' if m in NEW_METHODS else 'v1 历史'} | {s['valid']}/102 | {s['hits']}/102 ({s['hits']/102:.2%}) | {s['top5_hits']}/102 | {s['metrics']['mrr_at5']['mean']:.4f} | {s['late']} | {s['early']} |")
    sensitivity = result['complete_case_sensitivity']
    lines += ['', f"四格均有有效输出的共同案例（辅助描述）：n={sensitivity['n']}；" + '，'.join(f"{NAMES[m]} {sensitivity['hits'][m]}/{sensitivity['n']}" for m in METHODS) + '。完整 102 条仍为主评估。']
    lines += ['', '| 比较（左减右） | Top-1 差值 | 组 bootstrap 95% CI | 新增／丢失 | Holm p |',
              '|---|---:|---:|---:|---:|']
    for c in result['contrasts'].values():
        lines.append(f"| {c['title']} | {100*c['mean']:+.2f} pp | [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] pp | +{c['gained']}/−{c['lost']} | {c['p_holm']:.4f} |")
    c = result['interaction']
    lines += ['', f"交互差中差（有关系的顺序效应 − 无关系的顺序效应）：{100*c['mean']:+.2f} pp，95% CI [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}] pp。未对交互额外计算 p 值。", '',
              '所有命中率以全部 102 条为分母，无效响应计未命中。晚判/早判仅统计有效响应，按原始轨迹顺序比较预测与标注位置。', '',
              '这两格是新推理；另两格复用 v1 历史响应。因此批次差异会混入简单效应，尤其与交互混淆，不能作纯因果解释。单次运行没有重复种子的标准差，组区间不衡量服务随机性。', '',
              '完整决策单元和同方案窗口本轮尚未改动；呈现结果不能代表这些改动的效果。', '',
              '新增正式诊断用量：`' + json.dumps(result['new_diagnostic_cost']) + '`。',
              '被拒凭证试跑另计：`' + json.dumps(result['rejected_preflight_cost']) + '`。', '', '## Claim Candidates', '']
    for c in result['contrasts'].values():
        lines += [f"- Claim: 本开发集本轮比较中，{c['title']}的 Top-1 差值为 {100*c['mean']:+.2f} pp。",
                  '  - Source evidence: summary.json、paired_cases.csv 与原始请求/SSE。',
                  '  - Allowed wording: 描述观测差值及其组配对区间。',
                  '  - Forbidden stronger wording: 不据此断言该呈现有稳定、跨任务或因果优势。',
                  f"  - Uncertainty: 95% CI={c['ci95']}；Holm p={c['p_holm']:.4f}；不同批次、各一次。",
                  '  - Next check: 同步重复四格，并在独立轨迹复现。', '  - Decision: weaken']
    if all(c['ci95'][0] <= 0 <= c['ci95'][1] for c in result['contrasts'].values()):
        lines[2:2] = ['本轮没有观察到足以支持切换呈现方式的证据：四个简单效应的95%配对区间均包含0。晚判仍占主要错误，后续可固定现有呈现，单独评估完整决策单元及沿用/纠正上下文。', '']
    (out / 'analysis-report.md').write_text('\n'.join(lines) + '\n')
    (out / 'stats-appendix.md').write_text(
        '# 统计附录\n\n主指标为 Top-1；Top-5、MRR@5 和错误方向为辅助描述。轨迹为配对单位，仓库/任务组为相关性单位。'
        '四个简单效应在读取新诊断结果之前指定，统一 Holm 校正。Top-1 为二元变量，直接采用非参数组方法，不假定其正态分布。\n\n'
        '沿用 v1 的 grouped 函数：5000 次整组 bootstrap，按抽中轨迹总数加权；随机种子 20261028。'
        '每一简单效应在组内整体交换两方法标签，20000 次随机化，双侧 p 使用加一校正。'
        '该检验以组独立及零假设下组内方法标签可交换为条件；不同时段服务调用使可交换性未获保证，故 p 值只作条件性开发诊断。'
        '交互只报告配对组 bootstrap 区间；共同运行批次与交互混淆，不能解释为独立交互因果效应。\n\n'
        '没有对各方法运行 3—5 个种子；不报告虚构的均值±运行标准差。置信区间仅反映组抽样变化，不含服务波动。'
        '102 条轨迹已被反复用于开发，不把这些统计当作确认性检验；未剔除任何困难案例。全部无效输出在主指标中计 0。\n\n'
        '精确指标、每组95%区间、原始/校正 p 值、效应量、用量、运行时段见 summary.json；案例配对见 paired_cases.csv。\n')


def figures(result: dict, out: Path) -> None:
    directory = out / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    means = np.array([result['methods'][m]['metrics']['top1']['mean'] for m in METHODS]) * 100
    ci = np.array([result['methods'][m]['metrics']['top1']['ci95'] for m in METHODS]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    bars = ax.bar(range(4), means, color=['#87949C', '#246B92', '#246B92', '#87949C'], width=.6)
    ax.errorbar(range(4), means, yerr=np.maximum(0, [means-ci[:, 0], ci[:, 1]-means]), fmt='none', color='#333', capsize=4)
    for i, m in enumerate(METHODS):
        ax.text(i, ci[i, 1] + 2, f"{result['methods'][m]['hits']}/102", ha='center', va='bottom')
    ax.set(xticks=range(4), xticklabels=ENGLISH, ylim=(0, 100), ylabel='Top-1 accuracy (%)',
           title='Same evidence, different presentation; historical v1 + new v2')
    for ext in ('png', 'pdf'): fig.savefig(directory / f'01-top1.{ext}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    left = np.zeros(4)
    for key, label, color in [('hits', 'Correct', '#3D8B75'), ('late', 'Late error', '#CF7955'), ('early', 'Early error', '#9675AC')]:
        values = np.array([result['methods'][m][key] for m in METHODS])
        bars = ax.barh(range(4), values, left=left, color=color, label=label)
        ax.bar_label(bars, labels=[str(v) if v else '' for v in values], label_type='center', color='white')
        left += values
    invalid = 102 - left
    if invalid.any(): ax.barh(range(4), invalid, left=left, color='#999', label='Invalid')
    ax.set(yticks=range(4), yticklabels=ENGLISH, xlim=(0, 102), xlabel='Trajectories (n = 102)',
           title='Error direction relative to the annotated root cause')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.16), ncol=4)
    for ext in ('png', 'pdf'): fig.savefig(directory / f'02-error-direction.{ext}', dpi=180)
    plt.close(fig)
    (out / 'figure-catalog.md').write_text(
        '# 图表目录\n\n01-top1.png/pdf：比较同原文四种呈现的 Top-1。数据来自 summary.json；误差线为5000次组 bootstrap 的95%区间，'
        '灰色为历史 v1，蓝色为本轮 v2。柱上为准确命中数，区间不包含服务运行波动。判断方法差异应读取配对区间，不能仅比较柱高或误差线重叠。'
        '该图用于决定呈现方式是否值得同步复现，而非宣称稳定优胜。\n\n'
        '02-error-direction.png/pdf：展示每格正确、晚判、早判和无效数量。数据来自逐例 predictions.json，方向按原轨迹位置计算。'
        '该图用于检查晚判减少是否伴随早判增加，决定后续是否仍需完整决策与沿用/纠正上下文。无误差线，为单次运行的完整计数。'
        '两个图均复用 v1 两格，图注必须保留不同运行批次、102条开发集的限制。\n')


if __name__ == '__main__':
    result = analyze()
    print(json.dumps({'hits': {m: s['hits'] for m, s in result['methods'].items()},
                      'new_cost': result['new_diagnostic_cost']}, ensure_ascii=False, indent=2))
