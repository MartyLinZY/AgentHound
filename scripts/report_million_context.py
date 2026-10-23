#!/usr/bin/env python3
"""Summarize the three fixed RootSE runs and separately account for capacity probes."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/rootse-full-deepseek-v41-flash-v1')
    out = parser.parse_args().out.resolve()
    runs = []
    for directory in [ROOT / 'outputs/rootse-full-qwen25-14b-1m-v1', ROOT / 'outputs/rootse-full-qwen35-397b-a17b-v1', out]:
        path = directory / 'metrics.json'
        if path.exists():
            runs.append(json.loads(path.read_text()))
    if not (out / 'metrics.json').exists():
        raise ValueError('full run must finish before final reporting')
    probes = []
    archived = json.loads((ROOT / 'experiments/rootse_full/deepseek_capacity_probes.json').read_text())
    for data in archived:
        try:
            response = json.loads(data.get('response', ''))
        except ValueError:
            response = {}
        probes.append({'file': data.get('artifact_path', data['file']), 'http_status': data.get('http_status'),
                       'input_kind': data['file'], 'model': response.get('model'),
                       'usage': response.get('usage'), 'elapsed_seconds': data['elapsed_seconds']})
    successful = [p for p in probes if p['http_status'] == 200 and p['usage']]
    largest = max(successful, key=lambda p: p['usage']['prompt_tokens'])
    known_tokens = sum(p['usage']['total_tokens'] for p in successful)
    table = ['| 模型 | Tol@0 | Tol@1 | Tol@2 | Tol@3 | 有效输出 | 状态 |', '|---|---:|---:|---:|---:|---:|---|']
    for m in runs:
        table.append('| ' + m['model'] + ' | ' + ' | '.join(f'{v*100:.2f}% ({h}/102)' for v,h in zip(m['tolerance_accuracy'],m['tolerance_hits'])) + f" | {m['valid_predictions']}/102 | {m['status_counts']} |")
    text = f'''# 百万上下文模型实验

已选择移动云 `deepseek-v4.1-flash`。官方[模型文档](https://api-docs.deepseek.com/quick_start/pricing/)标注1M上下文，[技术报告](https://arxiv.org/abs/2609.19969)标注552B骨干参数。

## 部署容量实测

合成文本为999000次` hello`，max_tokens=1。服务成功返回HTTP200，模型名为`{largest['model']}`，输入{largest['usage']['prompt_tokens']:,} tokens、输出1 token，耗时{largest['elapsed_seconds']:.2f}秒。首次同样请求返回502；成功重试报告917376个cached tokens，不能将其延迟视为冷启动性能。最长真实RootSE输入193317 tokens也已接收成功。

这验证了接近百万输入的接收能力，不是长文检索或根因定位质量测试，也没有精确寻找容量上限。全部{len(probes)}次容量探测已知用量{known_tokens:,} tokens，{sum(p['usage'] is None for p in probes)}次usage未知；独立于正式实验。

## 全量结果

{chr(10).join(table)}

所有实验为完整102条、All-at-Once、without-reference，保持官方提示词、temperature=0、max_tokens=8192。新runner仅更换模型白名单；测试及对照程序验证其余推理代码AST一致。请求逐条比较只改变model；默认思考模式、模型架构、tokenizer和实际算力并不相同。失败始终留在102分母中，输出截断不等于输入上下文超限。

论文All-at-Once使用Claude Sonnet4.5，准确命中25/102（显示24.5%）。完整方法对照和公开CSV复核见[REPORT.md](REPORT.md)；与14B的逐条配对检验见[MODEL_COMPARISON.md](MODEL_COMPARISON.md)。这些是不同骨干的单轮对照，不能将差异归为诊断算法改进。

## 复现及过程

仓库执行：`bash scripts/reproduce_rootse_million.sh /path/to/moma_ai.key`。固定依赖、数据哈希、提示词、代码快照、请求、响应、状态均已保存；已有终态样本不再次请求。配置在百万容量重试返回前冻结，其中selection_note记录的是当时状态；本报告及容量原始响应补充最终验证，不改写历史配置。

容量原始记录已存档于experiments/rootse_full/deepseek_capacity_probes.json，见下表；常规复现不重复容量测试。探测脚本为`scripts/probe_moma_context_v2.py`，添加`--synthetic-million`可重放合成容量探测（需新out路径，会产生API费用）。

| 文件 | HTTP | 已知输入tokens | 耗时秒 |
|---|---:|---:|---:|
'''
    for p in probes:
        text += f"| {p['file']} | {p['http_status']} | {(p['usage'] or {}).get('prompt_tokens', '未知')} | {p['elapsed_seconds']:.2f} |\n"
    (out / 'MILLION_CONTEXT_REPORT.md').write_text(text)
    (out / 'capacity_probe_summary.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2)+'\n')
    print(out / 'MILLION_CONTEXT_REPORT.md')

if __name__ == '__main__':
    main()
