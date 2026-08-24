"""Step-level metrics and evidence-tier summaries."""

from __future__ import annotations

import statistics
from typing import Any

from loghound_agent_debug.schema import Label


def evaluate_rows(rankings: list[dict[str, Any]], labels: dict[str, Label]) -> dict[str, Any]:
    """Compute aggregate step-level metrics from rankings and root labels."""
    rows = []
    for result in rankings:
        label = labels[result["trajectory_id"]]
        roots = set(label.root_cause_steps)
        rank_by_step = {item["source_step_id"]: int(item["rank"]) for item in result["ranking"]}
        # A trajectory may have multiple root steps; the first ranked root is
        # used for MRR and the Hit@K indicators.
        root_ranks = sorted(rank_by_step[root] for root in roots if root in rank_by_step)
        first_rank = root_ranks[0] if root_ranks else len(rank_by_step) + 1
        total = len(rank_by_step)
        rows.append({
            "trajectory_id": result["trajectory_id"],
            "evidence_tier": result.get("evidence_tier", "C"),
            "evidence_status": result.get("evidence_status", "no_evidence"),
            "steps": total,
            "root_cause_steps": sorted(roots),
            "root_ranks": root_ranks,
            "first_root_rank": first_rank,
            "hit_at_1": int(any(rank <= 1 for rank in root_ranks)),
            "hit_at_3": int(any(rank <= 3 for rank in root_ranks)),
            "hit_at_5": int(any(rank <= 5 for rank in root_ranks)),
            "mrr": 1.0 / first_rank if root_ranks else 0.0,
        })
    return summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]], include_length: bool = True) -> dict[str, Any]:
    """Summarize per-trajectory rows, optionally adding length buckets."""
    if not rows:
        return {"n": 0}
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)
    summary = {
        "n": len(rows),
        "Hit@1": mean("hit_at_1"),
        "Hit@3": mean("hit_at_3"),
        "Hit@5": mean("hit_at_5"),
        "MRR": mean("mrr"),
        "Mean Rank": mean("first_root_rank"),
        "Median Rank": statistics.median(row["first_root_rank"] for row in rows),
    }
    if include_length:
        summary["by_length"] = {}
        # Length buckets prevent long trajectories from being compared only by
        # raw rank against much shorter trajectories.
        for bucket, selected in {
            "short_T<=10": [row for row in rows if row["steps"] <= 10],
            "medium_10<T<=30": [row for row in rows if 10 < row["steps"] <= 30],
            "long_T>30": [row for row in rows if row["steps"] > 30],
        }.items():
            summary["by_length"][bucket] = summarize_rows(selected, include_length=False) if selected else {"n": 0}
    return summary


def rows_from_results(rankings: list[dict[str, Any]], labels: dict[str, Label]) -> list[dict[str, Any]]:
    """Create CSV-friendly per-trajectory rows using the same rank logic."""
    output = evaluate_rows(rankings, labels)
    # evaluate_rows intentionally returns the summary for the public API; this
    # helper is kept separate for reporting and uses the same calculation path.
    del output
    rows = []
    for result in rankings:
        label = labels[result["trajectory_id"]]
        roots = set(label.root_cause_steps)
        rank_by_step = {item["source_step_id"]: int(item["rank"]) for item in result["ranking"]}
        ranks = sorted(rank_by_step[root] for root in roots if root in rank_by_step)
        rank = ranks[0] if ranks else len(rank_by_step) + 1
        total = len(rank_by_step)
        rows.append({
            "trajectory_id": result["trajectory_id"],
            "evidence_tier": result.get("evidence_tier", "C"),
            "evidence_status": result.get("evidence_status", "no_evidence"),
            "steps": total,
            "root_cause_steps": ",".join(sorted(roots)),
            "root_ranks": ",".join(str(x) for x in ranks),
            "first_root_rank": rank,
            "hit_at_1": int(any(x <= 1 for x in ranks)),
            "hit_at_3": int(any(x <= 3 for x in ranks)),
            "hit_at_5": int(any(x <= 5 for x in ranks)),
            "mrr": 1.0 / rank if ranks else 0.0,
        })
    return rows
