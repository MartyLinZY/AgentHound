"""Case cards and lightweight, auditable error categories."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from loghound_agent_debug.graph import build_graph
from loghound_agent_debug.schema import Label, TrajectoryInput


def nearest_seed_distance(item: TrajectoryInput, ranking: dict[str, Any], root_ids: list[str]) -> int | None:
    graph = build_graph(item)
    ordinal = {step.source_step_id: step.ordinal for step in item.steps}
    roots = [ordinal[root] for root in root_ids if root in ordinal]
    seeds = [ordinal[str(seed["source_step_id"])] for seed in ranking.get("symptom_seeds", []) if str(seed["source_step_id"]) in ordinal]
    if not roots or not seeds:
        return None
    adjacency = [[] for _ in item.steps]
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    best = None
    for root in roots:
        distances = {root: 0}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        for seed in seeds:
            if seed in distances:
                best = distances[seed] if best is None else min(best, distances[seed])
    return best


def make_case_cards(inputs: dict[str, TrajectoryInput], labels: dict[str, Label], rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for result in rankings:
        if result.get("evidence_tier") != "A":
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
            "nearest_seed_graph_distance": nearest_seed_distance(inputs[result["trajectory_id"]], result, root_ids),
            "symptom_seeds": result.get("symptom_seeds", []),
        })
    return cards


def error_categories(cards: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for card in cards:
        if not card["root_evidence"] or not any(evidence for group in card["root_evidence"] for evidence in group):
            counts["root_cause_without_ranked_evidence"] += 1
        if card["nearest_seed_graph_distance"] is None:
            counts["root_cause_no_graph_path_to_seed"] += 1
        if card["root_ranks"] and card["root_ranks"][0] > 5:
            counts["root_cause_ranked_below_top5"] += 1
    return dict(sorted(counts.items()))


def render_case_report(cards: list[dict[str, Any]], errors: dict[str, int]) -> str:
    lines = ["# AgenTracer A-tier case report", "", f"Cases: {len(cards)}", ""]
    for card in cards:
        lines.extend([
            f"## {card['trajectory_id']}",
            "",
            f"- Root cause step(s): `{', '.join(card['root_cause_steps'])}` ({', '.join(card['root_cause_agents']) or 'unknown agent'})",
            f"- Root rank(s): {card['root_ranks']}",
            f"- Nearest seed graph distance: {card['nearest_seed_graph_distance']}",
            f"- Symptom seeds: {card['symptom_seeds']}",
            "- Top-3:",
        ])
        for item in card["top_3"]:
            lines.append(f"  - #{item['rank']} step `{item['source_step_id']}` {item['agent']}: score={item['suspiciousness']:.4f}; evidence={', '.join(item['evidence']) or 'none'}")
        lines.append("")
    lines.extend(["## Error categories", ""])
    for name, count in errors.items():
        lines.append(f"- {name}: {count}")
    return "\n".join(lines) + "\n"
