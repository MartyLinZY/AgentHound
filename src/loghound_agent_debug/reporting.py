"""Case cards and lightweight, auditable error categories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from loghound_agent_debug.schema import Label, TrajectoryInput


def make_case_cards(inputs: dict[str, TrajectoryInput], labels: dict[str, Label], rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build explainable cards for hard-evidence rankings only."""
    cards = []
    for result in rankings:
        if result.get("evidence_tier") != "A":
            # B/C cases are retained in aggregate metrics but excluded from
            # the detailed case review protocol.
            continue
        label = labels[result["trajectory_id"]]
        root_ids = label.root_cause_steps
        top = result["ranking"][:3]
        root_items = [item for item in result["ranking"] if item["source_step_id"] in root_ids]
        cards.append({
            "trajectory_id": result["trajectory_id"],
            "steps": len(inputs[result["trajectory_id"]].steps),
            "root_cause_steps": root_ids,
            "root_cause_agents": label.root_cause_agents,
            "root_ranks": [item["rank"] for item in root_items],
            "top_1": top[0] if top else None,
            "top_3": top,
            "root_components": [item["components"] for item in root_items],
            "root_evidence": [item["evidence"] for item in root_items],
            "clue_hits": result.get("clue_hits", []),
            "graph_stats": result.get("graph_stats", {}),
        })
    return cards


def error_categories(cards: list[dict[str, Any]]) -> dict[str, int]:
    """Count simple, auditable failure modes in the generated case cards."""
    counts = Counter()
    for card in cards:
        if not card["root_evidence"] or not any(evidence for group in card["root_evidence"] for evidence in group):
            counts["root_cause_without_ranked_evidence"] += 1
        if card["root_ranks"] and card["root_ranks"][0] > 5:
            counts["root_cause_ranked_below_top5"] += 1
    return dict(sorted(counts.items()))


def render_case_report(cards: list[dict[str, Any]], errors: dict[str, int]) -> str:
    """Render case cards and error counts as a Markdown report."""
    lines = ["# AgenTracer A-tier case report", "", f"Cases: {len(cards)}", ""]
    for card in cards:
        lines.extend([
            f"## {card['trajectory_id']}",
            "",
            f"- Root cause step(s): `{', '.join(card['root_cause_steps'])}` ({', '.join(card['root_cause_agents']) or 'unknown agent'})",
            f"- Root rank(s): {card['root_ranks']}",
            f"- Root path components: {[item.get('path_normalized', 0.0) for item in card['root_components']]}",
            f"- Clue hits: {card['clue_hits']}",
            f"- PageRank graph stats: {card['graph_stats']}",
            "- Top-3:",
        ])
        for item in card["top_3"]:
            lines.append(f"  - #{item['rank']} step `{item['source_step_id']}` {item['agent']}: score={item['suspiciousness']:.4f}; evidence={', '.join(item['evidence']) or 'none'}")
        lines.append("")
    lines.extend(["## Error categories", ""])
    for name, count in errors.items():
        lines.append(f"- {name}: {count}")
    return "\n".join(lines) + "\n"
