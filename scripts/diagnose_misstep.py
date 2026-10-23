#!/usr/bin/env python3
"""Summarize RootSE misstep-channel coverage and its independent ranking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--misstep", default="outputs/rootse_misstep/misstep.jsonl")
    parser.add_argument("--rankings", default="outputs/rootse_misstep/rankings.jsonl")
    parser.add_argument("--root", default="data/raw/rootse")
    parser.add_argument("--output", default="outputs/rootse_misstep/misstep_diagnostics.json")
    args = parser.parse_args()
    inputs, labels = load_rootse(args.root)
    input_by_id = {x.trajectory_id: x for x in inputs}
    label_by_id = {x.trajectory_id: x for x in labels}
    missteps = {x["trajectory_id"]: x for x in read_jsonl(Path(args.misstep))}
    rankings = {x["trajectory_id"]: x for x in read_jsonl(Path(args.rankings))}
    if set(missteps) != set(input_by_id) or set(rankings) != set(input_by_id):
        raise ValueError("diagnostics require complete, matching RootSE misstep and ranking files")
    root_hit = symptom_hit = accepted = proposed = 0
    misstep_steps = propagation_steps = 0
    facet_counts: dict[str, int] = {}
    single_top1 = single_top3 = 0
    per_trajectory = []
    for tid, item in input_by_id.items():
        rows = missteps[tid]["steps"]
        row_by_id = {str(x["source_step_id"]): x for x in rows}
        scores = {sid: float(row.get("misstep_score", 0)) for sid, row in row_by_id.items()}
        order = sorted(item.steps, key=lambda s: (-scores.get(s.source_step_id, 0), s.ordinal))
        roots = set(label_by_id[tid].root_cause_steps)
        single_top1 += bool(order and order[0].source_step_id in roots)
        single_top3 += any(s.source_step_id in roots for s in order[:3])
        root_step = next((row_by_id.get(x) for x in roots if x in row_by_id), None)
        root_hit += bool(root_step and root_step.get("misstep_score", 0) > 0)
        symptom_ids = {str(x["source_step_id"]) for x in rankings[tid].get("clue_hits", [])}
        symptom_hit += sum(row_by_id.get(sid, {}).get("misstep_score", 0) > 0 for sid in symptom_ids)
        misstep_steps += sum(x.get("verdict") == "misstep" for x in rows)
        propagation_steps += sum(x.get("verdict") == "propagation" for x in rows)
        for row in rows:
            proposed += len(row.get("facets", [])) + len(row.get("rejected_facets", []))
            accepted += len(row.get("facets", []))
            for facet in row.get("facets", []):
                facet_counts[facet["name"]] = facet_counts.get(facet["name"], 0) + 1
        per_trajectory.append({"trajectory_id": tid, "misstep_steps": sum(x.get("verdict") == "misstep" for x in rows), "propagation_steps": sum(x.get("verdict") == "propagation" for x in rows), "root_step_misstep": bool(root_step and root_step.get("misstep_score", 0) > 0)})
    n = len(input_by_id)
    result = {"trajectories": n, "root_cause_steps_with_misstep": root_hit, "root_cause_step_coverage": root_hit / n if n else 0, "misstep_only_top1_hits": single_top1, "misstep_only_top1": single_top1 / n if n else 0, "misstep_only_top3_hits": single_top3, "misstep_only_top3": single_top3 / n if n else 0, "mean_misstep_steps": misstep_steps / n if n else 0, "mean_propagation_steps": propagation_steps / n if n else 0, "symptom_steps_marked_misstep": symptom_hit, "accepted_facets": accepted, "all_facet_proposals": proposed, "facet_acceptance_rate": accepted / proposed if proposed else 0, "facet_counts": facet_counts, "per_trajectory": per_trajectory}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_trajectory"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
