#!/usr/bin/env python3
"""在命令行查看 parquet 数据集的前 N 条记录。

用法（使用项目约定的 mybook 环境）：
    conda run -n mybook python scripts/view_parquet.py <file.parquet> [--n 10] [--full] [--width 100] [--parse]

示例：
    conda run -n mybook python scripts/view_parquet.py data/raw/agentracer/tracertraj-code-test.parquet --n 3
    conda run -n mybook python scripts/view_parquet.py data/raw/swebench_pro/SWE-bench_Pro-test.parquet --parse --n 2
"""
import argparse
import ast
import math
import sys

import pandas as pd


def fmt_value(v, width, truncate, parse):
    """把单元格值格式化成可读文本，处理 None/NaN/长文本/多行/复合类型。"""
    if v is None:
        return "<None>"
    if isinstance(v, float) and math.isnan(v):
        return "<NaN>"
    if isinstance(v, (str, bytes, bytearray)):
        s = v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else v
        if parse:
            stripped = s.strip()
            if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
                try:
                    parsed = ast.literal_eval(stripped)
                    s = repr(parsed)
                except (ValueError, SyntaxError):
                    pass  # 解析失败就按普通字符串显示
    else:
        s = str(v)  # ndarray/list/dict 等复合类型统一转字符串，同样参与截断
    if truncate and len(s) > width:
        s = s[: width - 3] + "..."
    return s


def print_rows(df, n, width, truncate, parse):
    total = len(df)
    shown = min(n, total)
    ident_col = next((c for c in ("instance_id", "trajectory_id", "question_ID", "question")
                      if c in df.columns), df.columns[0])
    for i in range(shown):
        row = df.iloc[i]
        ident = fmt_value(row[ident_col], width, truncate, parse=False)
        print(f"\n{'=' * 70}")
        print(f"[{i}] {ident_col} = {ident}   ({i}/{total})")
        print("=" * 70)
        for col in df.columns:
            s = fmt_value(row[col], width, truncate, parse)
            lines = s.splitlines() or [""]
            print(f"  {col}: {lines[0]}")
            for extra in lines[1:]:
                print(f"      {extra}")
    if shown < total:
        print(f"\n... 共 {total} 条，已显示前 {shown} 条（用 --n 查看更多，--full 取消截断）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="查看 parquet 数据集前 N 条记录")
    ap.add_argument("path", help="parquet 文件路径")
    ap.add_argument("--n", type=int, default=10, help="显示条数（默认 10）")
    ap.add_argument("--full", action="store_true", help="不截断长文本")
    ap.add_argument("--width", type=int, default=100, help="单字段截断宽度（默认 100）")
    ap.add_argument("--parse", action="store_true",
                    help="尝试把字符串化的 JSON 列表/字典（如 fail_to_pass）解析成可读形式")
    args = ap.parse_args(argv)

    try:
        df = pd.read_parquet(args.path)
    except FileNotFoundError:
        print(f"文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"读取失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"文件: {args.path}")
    print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列: {', '.join(df.columns)}")
    print_rows(df, args.n, args.width, truncate=not args.full, parse=args.parse)


if __name__ == "__main__":
    main()