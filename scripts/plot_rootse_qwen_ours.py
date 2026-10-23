#!/usr/bin/env python3
"""Create two source-backed diagnostic figures after complete offline scoring."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/rootse-ours-qwen38-nothinking-causal-v1');a=p.parse_args()
d=json.loads((a.out/'metrics_causal.json').read_text());dest=a.out/'analysis/figures';dest.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
methods=['baseline','anchors-global','anchors-backtrace','causal-judge'];labels=['Global','Anchors + global','Anchors + backtrace','Failure adjudication']
values=[d['metrics'][m]['top1_hits']/102*100 for m in methods]
ci=np.array([d['metrics'][m]['repository_bootstrap_exact_95ci_pp'] for m in methods])
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
bars=ax.bar(labels,values,color=['#9AA3AC','#56B4E9','#0072B2','#D55E00'],width=.62)
ax.errorbar(range(4),values,yerr=np.array([np.array(values)-ci[:,0],ci[:,1]-np.array(values)]),fmt='none',ecolor='#222222',capsize=5)
for bar,m in zip(bars,methods):
    v=d['metrics'][m]['top1_hits'];ax.text(bar.get_x()+bar.get_width()/2,1,f'{v}/102',ha='center',va='bottom',color='white' if v>8 else '#111111',weight='bold')
ax.set_ylim(0,max(ci[:,1])+10);ax.set_ylabel('Exact localization accuracy (%)');ax.set_title('Same Qwen graph, different final ranking',loc='left',weight='bold',pad=28)
ax.text(0,1.015,'Thinking disabled · all 102 cases · repository bootstrap 95% intervals',transform=ax.transAxes,fontsize=9,color='#555555')
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
for ext in ['pdf','png']:fig.savefig(dest/f'01-same-graph-comparison.{ext}',dpi=300)
plt.close(fig)

pairs=d['paired_exploratory']['anchors-backtrace']
names=['Both correct','New judge only','Old ranking only','Both incorrect']
counts=[pairs[k] for k in ['both_correct','right_only','left_only','neither_correct']]
fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
bars=ax.barh(names[::-1],counts[::-1],color=['#9AA3AC','#0072B2','#D55E00','#555555'])
for bar,n in zip(bars,counts[::-1]):ax.text(n+.6,bar.get_y()+bar.get_height()/2,str(n),va='center')
ax.set_xlim(0,max(counts)+8);ax.set_xlabel('Trajectories');ax.set_title('What changed for the same trajectories?',loc='left',weight='bold')
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
for ext in ['pdf','png']:fig.savefig(dest/f'02-paired-outcomes.{ext}',dpi=300)
plt.close(fig)
(a.out/'analysis/figure-catalog.md').write_text('''# 图表说明

数据均来自 metrics_causal.json，无示意数字。

1. 01-same-graph-comparison.pdf/png：同一批非思考Qwen图上的三种旧排序与新裁决，纵轴为102分母精确定位率；误差条为按仓库重采样的95%区间，不是模型重复运行方差。用于判断新增裁决的观测收益，不证明图本身的贡献。检查同图、同分母及失败回退均已包含。
2. 02-paired-outcomes.pdf/png：逐案例共同命中、新增命中、丢失命中和共同错误，四项合计102；没有误差条，因为是本轮固定样本计数。用于检验总体变化是否掩盖退化案例。不能将新增命中数当作未来必然收益。

标题明确thinking关闭。未把历史外部thinking结果混入同协议主图。RootSE已参与开发、单次运行，独立划分及重复运行仍待完成。
''')
print(dest)
