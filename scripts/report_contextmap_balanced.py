#!/usr/bin/env python3
"""Package verified v4 numbers and publication-friendly static figures."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_contextmap_balanced import ROOT, OUT as R, V1, V3
from run_whowhen_context import OUT as W
from run_contextmap import read, sha
from suite_transport import save, now

DEST = ROOT.parents[2] / 'output/attentionmap_v4_generalization_2026-10-29'
NAMES = {'units_repeat': 'v3 原请求同步重跑', 'units_aligned': '完整单元＋口径对齐',
         'balanced': '广覆盖＋口径对齐', 'full': '全文 all-at-once'}
ENGLISH = {'units_repeat': 'v3 repeat', 'units_aligned': 'Aligned units', 'balanced': 'Balanced context', 'full': 'Full trajectory'}


def table(methods: dict) -> str:
    lines = ['| 方法 | Top-1 | Top-5 | MRR@5 |', '|---|---:|---:|---:|']
    for method, s in methods.items():
        n = s['n']
        lines.append(f"| {NAMES[method]} | {s['top1']}/{n}（{100*s['top1']/n:.2f}%） | {s['top5']}/{n}（{100*s['top5']/n:.2f}%） | {s['metrics']['mrr_at5']['mean']:.4f} |")
    return '\n'.join(lines)


def figures(results: dict):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    dest = DEST / 'figures'; dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), constrained_layout=True)
    for row, (dataset, result) in enumerate(results.items()):
        methods = list(result['methods'])
        for col, metric in enumerate(('top1', 'top5')):
            ax = axes[row, col]
            values = [result['methods'][m]['metrics'][metric]['mean']*100 for m in methods]
            low = [v-result['methods'][m]['metrics'][metric]['ci95'][0]*100 for v, m in zip(values, methods)]
            high = [result['methods'][m]['metrics'][metric]['ci95'][1]*100-v for v, m in zip(values, methods)]
            colors = ['#8296ab' if m != 'balanced' else '#187b80' for m in methods]
            bars = ax.barh(np.arange(len(methods)), values, color=colors, height=.55,
                           xerr=[low, high], error_kw={'color': '#30475d', 'capsize': 3})
            ax.set_yticks(np.arange(len(methods)), [ENGLISH[m] for m in methods]); ax.invert_yaxis()
            ax.set_xlim(0, 106); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Accuracy (%)')
            for i, (m, v) in enumerate(zip(methods, values)):
                s = result['methods'][m]
                ax.text(min(103, v+high[i]+1.5), i, f"{s[metric]}/{s['n']}", va='center', fontsize=9)
            ax.set_title(f"{'RootSE development' if dataset == 'rootse' else 'Who&When transfer'} · {metric.replace('top', 'Top-')}", loc='left', fontweight='bold')
            ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Same-model controls; error bars: 95% group bootstrap intervals', fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(dest / ('01-accuracy.'+suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    strata = results['whowhen']['strata']
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    keys = ['steps_le5', 'steps_6_to20', 'steps_gt20']; x = np.arange(3)
    for offset, method in enumerate(('full', 'units_aligned', 'balanced')):
        values = [strata[k][method]['metrics']['top5']['mean']*100 for k in keys]
        ax.bar(x+(offset-1)*.23, values, width=.21, label=ENGLISH[method], color=['#b9c6d2','#587997','#187b80'][offset])
    chance = [strata[k]['full']['uniform_random_top5']*100 for k in keys]
    ax.scatter(x, chance, marker='D', c='#b75c35', s=40, label='Uniform random Top-5', zorder=5)
    ax.set_xticks(x, [f"{label}\n(n={strata[k]['full']['n']})" for k, label in zip(keys, ('≤5 steps', '6–20 steps', '>20 steps'))])
    ax.set_ylabel('Top-5 accuracy (%)'); ax.set_ylim(0, 112)
    ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.16))
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    ax.set_title('Who&When: trajectory length changes the difficulty of Top-5', pad=14, loc='left')
    for suffix in ('png', 'pdf'):
        fig.savefig(dest / ('02-length-strata.'+suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    route_file = W / 'analysis/budget_route_summary.json'
    if route_file.exists():
        strata = read(route_file)['strata']
        keys = ['full_route', 'balanced_route']
        fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
        x = np.arange(2)
        for offset, method in enumerate(('full', 'balanced')):
            values = [strata[k][method]['metrics']['top5']['mean']*100 for k in keys]
            bars = ax.bar(x+(offset-.5)*.3, values, width=.28, label=ENGLISH[method],
                          color=['#8296ab', '#187b80'][offset])
            for bar, k in zip(bars, keys):
                s = strata[k][method]
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        f"{s['top5']}/{s['n']}", ha='center')
        ax.set_xticks(x, ['Full context fits 16K\n(n=161)', 'Full context exceeds 16K\n(n=23)'])
        ax.set_ylim(0, 106); ax.set_ylabel('Top-5 accuracy (%)')
        ax.set_title('Who&When: performance where compression is required', loc='left', pad=14)
        ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.16))
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for suffix in ('png', 'pdf'):
            fig.savefig(dest / ('03-budget-strata.'+suffix), dpi=200, bbox_inches='tight')
        plt.close(fig)


def main():
    results = {'rootse': read(R / 'analysis/summary.json'), 'whowhen': read(W / 'analysis/summary.json')}
    adjusted = read(R / 'analysis/primary_family_holm.json')
    for dataset, out in (('rootse', R), ('whowhen', W)):
        target = DEST / dataset; target.mkdir(exist_ok=True)
        for file in ('summary.json', 'predictions.json', 'cases.csv', 'primary_family_holm.json'):
            shutil.copy2(out / 'analysis' / file, target / file)
        for file in ('manifest.json', 'input_audit.json', 'analysis_code_before_results.json', 'analysis_amendment_before_results.json'):
            shutil.copy2(out / file, target / file)
        for file in ('input_reconstruction.json', 'replay.json'):
            shutil.copy2(out / 'verification' / file, target / file)
    shutil.copy2(W / 'dataset_audit.json', DEST / 'whowhen/dataset_audit.json')
    shutil.copy2(W / 'input_contrast_audit.json', DEST / 'whowhen/input_contrast_audit.json')
    figures(results)
    lines = ['# v4 对照与外部迁移：实验数据', '', '本轮使用 qwen3.8-flash、thinking=true、temperature=0；每条每方法一次诊断调用。RootSE 为开发集，Who&When 为方法冻结后的外部迁移数据。无效调用保留在分母并记零；正文解释另见实验结果.md。', '', '## RootSE（102 条）', '', table(results['rootse']['methods']), '',
             '## Who&When（184 条）', '', table(results['whowhen']['methods']), '',
             '外部任务标准答案不提供；仅与本轮相同输入的方法比较。完整单元与广覆盖有 165 条请求完全相同，只有 19 条实际改变上下文。', '',
             '## 配对差异', '', '| 比较 | 指标 | 差异（百分点） | 95% 组 bootstrap 区间 | 得 / 失 | Holm p |', '|---|---|---:|---:|---:|---:|']
    for name, c in adjusted.items():
        lo, hi = c['ci95']; label = name.split('/')[0] + '：' + NAMES[c['a']] + ' − ' + NAMES[c['b']]
        lines.append(f"| {label} | {c['metric']} | {100*c['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {len(c['gained'])} / {len(c['lost'])} | {c['p_holm']:.4f} |")
    lines += ['', '预先定义的八个 Top-1/Top-5 对比统一 Holm 校正。区间按 RootSE 的 35 组、外部 137 个任务/重复轨迹连通组重采样，5,000 次 bootstrap；p 为 20,000 次整组配对符号置换。', '', '## 外部任务与长度分层']
    for name, strata in results['whowhen']['strata'].items():
        lines += ['', f'### {name}', '', table(strata), '', f"均匀随机参照：Top-1 {100*strata['full']['uniform_random_top1']:.2f}%，Top-5 {100*strata['full']['uniform_random_top5']:.2f}%。"]
    lines += ['', '## 实际用量', '', '| 数据 | 方法 | 请求数 | Prompt tokens | Completion tokens | Total tokens |', '|---|---|---:|---:|---:|---:|']
    for dataset, result in results.items():
        for method, s in result['methods'].items():
            cost = s['cost']; lines.append(f"| {dataset} | {NAMES[method]} | {cost.get('requests',0)} | {cost.get('prompt_tokens',0):,} | {cost.get('completion_tokens',0):,} | {cost.get('total_tokens',0):,} |")
    lines += ['', '用量为服务端记录，包含有 usage 的重试；未知用量单列在 summary.json。RootSE 复用旧图，其历史构图成本未计入上述本轮诊断 tokens；外部图只使用本地冻结编码和规则边。', '',
              f"原始请求/SSE：{R}；{W}。", '', '图表：figures/01-accuracy.png、figures/02-length-strata.png、figures/03-budget-strata.png；同时提供矢量 PDF。']
    (DEST / '实验数据.md').write_text('\n'.join(lines)+'\n')
    save(DEST / 'combined_summary.json', results)
    save(DEST / 'artifact_manifest.json', {'created_at': now(), 'files': {str(p.relative_to(DEST)): sha(p) for p in DEST.rglob('*')
         if p.is_file() and p.name != 'artifact_manifest.json' and 'sources' not in p.relative_to(DEST).parts}})
    print(DEST, flush=True)


if __name__ == '__main__':
    main()
