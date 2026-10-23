#!/usr/bin/env python3
"""Two actual figures from the complete frozen influence experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rootse-influence-qwen38-nothinking-v3'
d=json.loads((OUT/'metrics.json').read_text());dest=OUT/'analysis/figures';dest.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
methods=['equal','weighted','shuffled','q-only','q-equal','q-weighted','q-shuffled']
labels=['Equal\nmean weight','Influence\nweighted','Shuffled\nweights','Error score\nonly','Error score\n× equal','Error score\n× influence','Error score\n× shuffled']
values=np.array([d['metrics'][m]['top1_hits']/102*100 for m in methods]);ci=np.array([d['metrics'][m]['repository_bootstrap_95ci_pp'] for m in methods])
fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
bars=ax.bar(labels,values,color=['#9AA3AC','#0072B2','#56B4E9','#E69F00','#B38B5D','#D55E00','#D6A085'],width=.65)
ax.errorbar(range(len(methods)),values,yerr=np.vstack([values-ci[:,0],ci[:,1]-values]),fmt='none',capsize=4,ecolor='#222222')
for bar,m in zip(bars,methods):
    v=d['metrics'][m]['top1_hits'];ax.text(bar.get_x()+bar.get_width()/2,.6,f'{v}/102',ha='center',va='bottom',weight='bold',fontsize=9)
ax.set_ylim(0,max(ci[:,1])+6);ax.set_ylabel('Exact localization accuracy (%)')
ax.set_title('Does estimated decision influence improve localization?',loc='left',weight='bold',pad=29)
ax.text(0,1.015,'Fixed graph · propagated variants share anchors · thinking off · repository bootstrap 95% intervals',transform=ax.transAxes,fontsize=9,color='#555555')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for ext in ['png','pdf']:fig.savefig(dest/f'01-main-comparison.{ext}',dpi=240)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for ax,method,title in zip(axes,['weighted','q-weighted'],['Propagation only','Error score × propagation']):
    values=[p[method] for p in d['permutations']];observed=d['metrics'][method]['top1_hits']
    low=min(min(values),observed);high=max(max(values),observed)
    ax.hist(values,bins=np.arange(low-.5,high+1.5,1),color='#9AA3AC',edgecolor='white')
    ax.axvline(observed,color='#D55E00',linewidth=2,label=f'Estimated weights: {observed}/102')
    ax.set_xlabel('Exact hits out of 102');ax.set_ylabel('Number of weight shuffles');ax.set_title(title,loc='left',weight='bold');ax.legend(fontsize=8)
    ax.set_xticks(range(low,high+1))
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('100 within-trajectory weight shuffles; one frozen model run',fontsize=12,weight='bold')
for ext in ['png','pdf']:fig.savefig(dest/f'02-weight-shuffle-control.{ext}',dpi=240)
plt.close(fig)
summary=d['permutation_summary'];primary=d['contrasts']['weighted_vs_equal']
(OUT/'analysis/figure-catalog.md').write_text(f'''# 图表说明

两图只读取metrics.json，不使用示意数字。

## 01-main-comparison

目的：分离边权本身、错误引入评分与传播组合的贡献。横轴为七个同图变体，纵轴为全102分母精确定位率；误差条为仓库bootstrap95%区间，5000次seed20261013，不是模型重复运行波动。

应观察：主比较加权减等权为{primary['delta_pp']:+.2f}pp；同时比较q-only和q-weighted，避免把语义评分的效果归给边权。解释：只有加权相对同条件等权改善才支持继续检验影响边权；不能凭组合总分宣称传播有效。

校验：固定图/锚点/传播公式，全量失败计入，equal与weighted每例总边权量一致；固定seed的shuffle不是挑选最佳seed。标签说明q为节点错误引入评分。历史不同公式与外部thinking结果未混入主图。

## 02-weight-shuffle-control

目的：检验估计权重的分配是否比随机分配更有信息。横轴为102条上的精确命中数，纵轴为100个打乱seed的出现次数；红线为真实估计边权结果，不是均值。无模型重复运行误差条。

应观察：{json.dumps(summary,ensure_ascii=False)}。

解释：若红线处于打乱分布中部或更低，当前实验不支持权重分配有明确定位信息。不能把100次条件打乱当作100次独立模型实验，不能将红线位置解释为真实因果效应。

共同边界：单次Qwen输出、已参与开发的RootSE，局部日志估计，无真实干预。图不能证明泛化或论文新颖性。
''')
print(dest)
