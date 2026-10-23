"""Strict complete-set RootSE evaluation, with ranking and distance separated."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from loghound_agent_debug.schema import Label, TrajectoryInput


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {"trajectories": n, "top1_hits": sum(r["top1_hit"] for r in rows),
            "top3_hits": sum(r["top3_hit"] for r in rows), "top5_hits": sum(r["top5_hit"] for r in rows),
            **{key: sum(r[source] for r in rows) / n if n else 0.0 for key, source in {
                "step_level_top1": "top1_hit", "step_level_top3": "top3_hit", "step_level_top5": "top5_hit",
                "mrr": "reciprocal_rank", "tolerance_1": "within_1", "tolerance_2": "within_2",
                "mean_step_distance": "step_distance"}.items()}}


def evaluate(inputs: list[TrajectoryInput], labels: list[Label], rankings: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject absent/duplicate runs, invalid IDs and non-converged rankings."""
    input_ids = [item.trajectory_id for item in inputs]
    label_by_id = {label.trajectory_id: label for label in labels}
    ranked_by_id = {row["trajectory_id"]: row for row in rankings}
    if not inputs or len(set(input_ids)) != len(inputs) or len(label_by_id) != len(labels) or len(ranked_by_id) != len(rankings):
        raise ValueError("inputs, labels and rankings must be nonempty and unique")
    if set(input_ids) != set(label_by_id) or set(input_ids) != set(ranked_by_id):
        raise ValueError("inputs, labels and rankings must cover the same complete trajectory set")
    rows = []
    groups: dict[str, dict[str, list]] = {key: defaultdict(list) for key in ("agent", "repository", "length")}
    for item in inputs:
        result = ranked_by_id[item.trajectory_id]
        found = result["ranking"]
        ids = [str(row["source_step_id"]) for row in found]
        ordinals = {step.source_step_id: step.ordinal for step in item.steps}
        if len(ids) != len(ordinals) or len(set(ids)) != len(ids) or set(ids) != set(ordinals):
            raise ValueError(f"incomplete or duplicate ranked steps: {item.trajectory_id}")
        if [row.get("rank") for row in found] != list(range(1, len(found) + 1)):
            raise ValueError(f"inconsistent rank order: {item.trajectory_id}")
        if result.get("graph_stats", {}).get("pagerank_converged") is not True:
            raise ValueError(f"PageRank did not converge: {item.trajectory_id}")
        if result.get("protocol", item.protocol) != item.protocol:
            raise ValueError(f"input/ranking protocol mismatch: {item.trajectory_id}")
        roots = set(label_by_id[item.trajectory_id].root_cause_steps)
        if not roots or not roots <= set(ordinals):
            raise ValueError(f"gold step is missing from input: {item.trajectory_id}")
        first_rank = min(ids.index(root) + 1 for root in roots)
        distance = min(abs(ordinals[ids[0]] - ordinals[root]) for root in roots)
        row = {"trajectory_id": item.trajectory_id, "task_id": item.task_id, "repository": item.repository,
               "root_cause_steps": sorted(roots), "top1_step_id": ids[0], "top3_step_ids": ids[:3],
               "top1_hit": first_rank <= 1, "top3_hit": first_rank <= 3, "top5_hit": first_rank <= 5,
               "reciprocal_rank": 1 / first_rank, "step_distance": distance,
               "within_1": distance <= 1, "within_2": distance <= 2}
        rows.append(row)
        groups["agent"][item.steps[0].agent].append(row)
        groups["repository"][item.repository or "unknown"].append(row)
        groups["length"]["<=30" if len(item.steps) <= 30 else ("31-60" if len(item.steps) <= 60 else ">60")].append(row)
    return {"dataset": "dengdan1999/RootSE", **_summary(rows),
            "unique_tasks": len({item.task_id or item.trajectory_id for item in inputs}),
            "by_group": {key: {group: _summary(selected) for group, selected in members.items()} for key, members in groups.items()},
            "per_trajectory": rows}
