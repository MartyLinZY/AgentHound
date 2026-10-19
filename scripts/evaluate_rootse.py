#!/usr/bin/env python3
"""Evaluate complete RootSE rankings with Top-K, MRR, and per-agent breakdowns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse

KS = (1, 3, 5, 10)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(inputs: list, labels: list, rankings: list[dict]) -> dict:
    input_ids = {x.trajectory_id for x in inputs}
    label_by_id = {x.trajectory_id: x for x in labels}
    ranking_by_id = {str(x["trajectory_id"]): x for x in rankings}
    if len(ranking_by_id) != len(rankings) or set(ranking_by_id) != input_ids or set(label_by_id) != input_ids:
        raise ValueError("inputs, labels, and rankings must cover the same unique complete trajectory set")
    rows = []
    for trajectory in inputs:
        result = ranking_by_id[trajectory.trajectory_id]
        found = result.get("ranking", [])
        expected_ids = [step.source_step_id for step in trajectory.steps]
        found_ids = [str(x.get("source_step_id")) for x in found]
        if len(found_ids) != len(expected_ids) or len(set(found_ids)) != len(expected_ids) or set(found_ids) != set(expected_ids):
            raise ValueError(f"incomplete or duplicate ranked steps for {trajectory.trajectory_id}")
        if result.get("graph_stats", {}).get("pagerank_converged") is not True:
            raise ValueError(f"PageRank did not converge for {trajectory.trajectory_id}")
        roots = set(label_by_id[trajectory.trajectory_id].root_cause_steps)
        # An incomplete ranking must never look like a lucky hit, so an absent
        # root step is treated as one worse than the last ranked position.
        ranks = {str(x["source_step_id"]): int(x["rank"]) for x in found}
        root_ranks = sorted(ranks[r] for r in roots if r in ranks)
        best = root_ranks[0] if root_ranks else len(ranks) + 1
        rows.append({
            "trajectory_id": trajectory.trajectory_id,
            "agent": trajectory.steps[0].agent if trajectory.steps else "unknown",
            "root_cause_steps": sorted(roots),
            "root_rank": best,
            "mrr": 1.0 / best if root_ranks else 0.0,
            "top1_step_id": found[0]["source_step_id"],
            "top3_step_ids": [x["source_step_id"] for x in found[:3]],
            **{f"top{k}_hit": bool(any(r <= k for r in root_ranks)) for k in KS},
        })

    n = len(rows)
    summary = {
        "dataset": "dengdan1999/RootSE",
        "trajectories": n,
        **{f"step_level_top{k}": sum(x[f"top{k}_hit"] for x in rows) / n if n else 0.0 for k in KS},
        **{f"top{k}_hits": sum(x[f"top{k}_hit"] for x in rows) for k in KS},
        "mrr": sum(x["mrr"] for x in rows) / n if n else 0.0,
        # Share of the top-5 recall the reranker could still recover: the gap
        # between "found in top-5" and "ranked first" is the reranking ceiling.
        "p_top1_given_top5": (
            sum(1 for x in rows if x["top1_hit"]) / max(sum(1 for x in rows if x["top5_hit"]), 1)
        ),
        "by_agent": {},
        "per_trajectory": rows,
    }
    for agent in sorted({x["agent"] for x in rows}):
        subset = [x for x in rows if x["agent"] == agent]
        m = len(subset)
        summary["by_agent"][agent] = {
            "n": m,
            **{f"top{k}": sum(x[f"top{k}_hit"] for x in subset) / m for k in KS},
            "mrr": sum(x["mrr"] for x in subset) / m,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/rootse")
    parser.add_argument("--rankings", default="outputs/rootse/rankings.jsonl")
    parser.add_argument("--run-status", default="outputs/rootse/run_status.json")
    parser.add_argument("--output", default="outputs/rootse/metrics.json")
    args = parser.parse_args()
    status = json.loads(Path(args.run_status).read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise SystemExit("experiment is incomplete; refusing to report partial-set metrics")
    inputs, labels = load_rootse(args.root)
    metrics = evaluate(inputs, labels, _read_jsonl(Path(args.rankings)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_trajectory"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
