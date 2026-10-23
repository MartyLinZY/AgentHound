"""Step-level metrics and evidence-tier summaries."""

from __future__ import annotations

from typing import Any

from loghound_agent_debug.schema import Label


def _row_from_result(result: dict[str, Any], label: Label) -> dict[str, Any]:
    """Build one step-level and agent-level evaluation row.

    Agents are ranked by the best-ranked step belonging to each agent.  Keeping
    only the first occurrence of an agent in the step ranking gives a stable,
    auditable agent ranking without allowing an agent with many steps to win by
    summing duplicated scores.
    """
    roots = set(label.root_cause_steps)
    root_agents = set(label.root_cause_agents)
    ranking = result.get("ranking", [])
    rank_by_step = {item["source_step_id"]: int(item["rank"]) for item in ranking}

    root_ranks = sorted(rank_by_step[root] for root in roots if root in rank_by_step)
    first_rank = root_ranks[0] if root_ranks else len(rank_by_step) + 1

    ranked_agents: list[str] = []
    for item in ranking:
        agent = str(item.get("agent", ""))
        if agent not in ranked_agents:
            ranked_agents.append(agent)
    agent_rank_by_name = {agent: rank for rank, agent in enumerate(ranked_agents, 1)}
    agent_ranks = sorted(agent_rank_by_name[agent] for agent in root_agents if agent in agent_rank_by_name)
    first_agent_rank = agent_ranks[0] if agent_ranks else len(ranked_agents) + 1
    total = len(rank_by_step)

    return {
        "trajectory_id": result["trajectory_id"],
        "evidence_tier": result.get("evidence_tier", "C"),
        "evidence_status": result.get("evidence_status", "no_evidence"),
        "steps": total,
        "root_cause_steps": sorted(roots),
        "root_cause_agents": sorted(root_agents),
        "predicted_agent": ranked_agents[0] if ranked_agents else "",
        "hit_at_1": int(any(rank <= 1 for rank in root_ranks)),
        "hit_at_3": int(any(rank <= 3 for rank in root_ranks)),
        "hit_at_5": int(any(rank <= 5 for rank in root_ranks)),
        "hit_at_10": int(any(rank <= 10 for rank in root_ranks)),
        "agent_hit_at_1": int(any(rank <= 1 for rank in agent_ranks)),
        "agent_hit_at_3": int(any(rank <= 3 for rank in agent_ranks)),
        "agent_hit_at_5": int(any(rank <= 5 for rank in agent_ranks)),
        "root_rank": first_rank,
        "mrr": 1.0 / first_rank if root_ranks else 0.0,
        "agent_mrr": 1.0 / first_agent_rank if agent_ranks else 0.0,
    }


def evaluate_rows(rankings: list[dict[str, Any]], labels: dict[str, Label]) -> dict[str, Any]:
    """Compute aggregate step-level and agent-level metrics from rankings."""
    rows = [_row_from_result(result, labels[result["trajectory_id"]]) for result in rankings]
    return summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]], include_length: bool = True) -> dict[str, Any]:
    """Summarize per-trajectory rows, optionally adding length buckets."""
    if not rows:
        return {"n": 0}
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)
    summary = {
        "n": len(rows),
        "step_level": {
            "Hit@1": mean("hit_at_1"),
            "Hit@3": mean("hit_at_3"),
            "Hit@5": mean("hit_at_5"),
            "Hit@10": mean("hit_at_10"),
            "MRR": mean("mrr"),
        },
    }
    if all("agent_hit_at_1" in row for row in rows):
        summary["agent_level"] = {
            "Hit@1": mean("agent_hit_at_1"),
            "Hit@3": mean("agent_hit_at_3"),
            "Hit@5": mean("agent_hit_at_5"),
            "MRR": mean("agent_mrr"),
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
    rows = []
    for result in rankings:
        row = _row_from_result(result, labels[result["trajectory_id"]])
        row["root_cause_steps"] = ",".join(row["root_cause_steps"])
        row["root_cause_agents"] = ",".join(row["root_cause_agents"])
        rows.append(row)
    return rows
