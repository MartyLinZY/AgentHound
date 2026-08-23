#!/usr/bin/env python3
"""Run prepare -> audit -> rank in one deterministic command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.adapters.agentracer import load_agentracer
from loghound_agent_debug.features.specificity import corpus_statistics
from loghound_agent_debug.rankers.loghound_static import load_config, rank_trajectory
from loghound_agent_debug.evidence import extract_evidence, load_clue_rules
from loghound_agent_debug.evaluate import rows_from_results, summarize_rows
from loghound_agent_debug.reporting import error_categories, make_case_cards, render_case_report
from loghound_agent_debug.schema import input_to_dict, label_to_dict


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/agentracer/tracertraj-code-test.parquet")
    parser.add_argument("--processed", default="data/processed/agentracer")
    parser.add_argument("--rankings", default="outputs/rankings/rankings.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rules", default="configs/clue_rules.yaml")
    args = parser.parse_args()

    inputs, labels = load_agentracer(args.raw)
    processed = Path(args.processed)
    write_jsonl(processed / "inputs.jsonl", [input_to_dict(item) for item in inputs])
    write_jsonl(processed / "labels.jsonl", [label_to_dict(label) for label in labels])
    config = load_config(args.config)
    rules = load_clue_rules(args.rules)
    rankings = [rank_trajectory(item, config, rules) for item in inputs]
    ranking_path = Path(args.rankings)
    write_jsonl(ranking_path, rankings)
    split_rows = []
    for item, label in zip(inputs, labels):
        evidence = extract_evidence(item, rules)
        split_rows.append({
            "trajectory_id": item.trajectory_id,
            "evidence_tier": evidence.tier,
            "evidence_status": evidence.status,
            "seed_steps": [seed["source_step_id"] for seed in evidence.symptom_seeds],
            "failed_occurrences": evidence.failed_occurrences,
            "failed_clusters": evidence.failed_clusters,
            "step_count": len(item.steps),
            "root_cause_steps": label.root_cause_steps,
        })
    split_path = Path("data/splits/evidence_split.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({"schema_version": "0.1.0", "counts": {tier: sum(row["evidence_tier"] == tier for row in split_rows) for tier in ("A", "B", "C")}, "trajectories": split_rows}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    labels_by_id = {label.trajectory_id: label for label in labels}
    metric_rows = rows_from_results(rankings, labels_by_id)
    metric_dir = Path("outputs/metrics")
    metric_dir.mkdir(parents=True, exist_ok=True)
    summaries = {"all": summarize_rows(metric_rows)}
    for tier in ("A", "B", "C"):
        summaries[tier] = summarize_rows([row for row in metric_rows if row["evidence_tier"] == tier])
    (metric_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_by_id = {item.trajectory_id: item for item in inputs}
    cards = make_case_cards(input_by_id, labels_by_id, rankings)
    case_dir = Path("outputs/cases")
    case_dir.mkdir(parents=True, exist_ok=True)
    errors = error_categories(cards)
    (case_dir / "a_tier_cases.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (case_dir / "case_report.md").write_text(render_case_report(cards, errors), encoding="utf-8")
    (case_dir / "error_categories.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "trajectories": len(inputs),
        "specificity": corpus_statistics(inputs),
        "specificity_used_for_ranking": config.get("specificity", {}).get("mode", "singleton"),
        "rule_version": rules.get("version"),
        "ranking_path": str(ranking_path),
        "evidence_split_path": str(split_path),
        "metrics_path": str(metric_dir / "summary.json"),
        "case_report_path": str(case_dir / "case_report.md"),
    }
    ranking_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
