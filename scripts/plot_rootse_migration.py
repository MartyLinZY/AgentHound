#!/usr/bin/env python3
"""Offline figures from completed migration metrics."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    m = json.loads((a.out / "metrics.json").read_text())
    folder = a.out / "analysis/figures"
    folder.mkdir(parents=True, exist_ok=True)
    names = [
        "Old clues / global",
        "New anchors / global",
        "Old clues / backtrace",
        "New anchors / backtrace",
    ]
    variants = ["baseline", "anchors-global", "legacy-backtrace", "anchors-backtrace"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    f, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(4)
    for shift, protocol, color in [
        (-0.18, "postmortem", "#27699c"),
        (0.18, "trace-only", "#91aabe"),
    ]:
        counts = [
            m["variants"][f"{protocol}/{v}"]["tolerance_hits"][0] for v in variants
        ]
        ax.barh(
            y + shift, np.array(counts) / 102 * 100, 0.34, label=protocol, color=color
        )
        for k, count in enumerate(counts):
            ax.text(
                count / 102 * 100 + 0.4,
                k + shift,
                f"{count}/102",
                va="center",
                fontsize=9,
            )
    aa_pair = next(row for row in m["paired"] if row["baseline"] == "all_at_once")
    aa_hits = aa_pair["both_correct"] + aa_pair["left_only"]
    ax.axvline(
        aa_hits / m["n"] * 100,
        color="#9c5138",
        ls="--",
        label=f"All-at-Once {aa_hits}/{m['n']} (postmortem)",
    )
    ax.set(
        yticks=y,
        yticklabels=names,
        xlim=(
            0,
            max(
                65,
                max(v["tolerance_hits"][0] for v in m["variants"].values()) / 102 * 100
                + 10,
            ),
        ),
        xlabel="Exact localization accuracy (%)",
        title="Same-graph ablations: all 102 trajectories retained",
    )
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    f.tight_layout()
    for ext in ["png", "pdf"]:
        f.savefig(folder / f"same-graph-ablations.{ext}", bbox_inches="tight", dpi=180)
    plt.close(f)
    f, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    g = m["graph_stats"]
    axes[0].bar(
        ["Complete", "Partial"],
        [g["complete_graphs"], g["partial_graphs"]],
        color=["#33828b", "#dc8e57"],
    )
    axes[0].set(title="Graph extraction coverage", ylabel="Trajectories", ylim=(0, 110))
    labels = list(m["extraction_status_counts"])
    vals = list(m["extraction_status_counts"].values())
    axes[1].barh(labels, vals, color="#738ea7")
    axes[1].set(title="All extraction targets", xlabel="Targets")
    f.tight_layout()
    for ext in ["png", "pdf"]:
        f.savefig(folder / f"extraction-coverage.{ext}", bbox_inches="tight", dpi=180)
    plt.close(f)
    (a.out / "analysis/figure-catalog.md").write_text("""# 图表目录

    1. figures/same-graph-ablations.png / .pdf：同图四组消融，postmortem和trace-only分别展示；虚线是postmortem全轨迹All-at-Once。仅点估计，不表达跨seed稳定性。
    2. figures/extraction-coverage.png / .pdf：完整/部分构图和每目标抽取终态。部分图按冻结策略保留成功边+规则边，不代表每个依赖都已抽取。

    输入metrics.json；生成scripts/plot_rootse_migration.py。不得以路径或边数量替代准确率结论。
    """)
    print("Saved migration figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
