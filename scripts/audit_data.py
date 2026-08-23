#!/usr/bin/env python3
"""Audit schema preservation, evidence tiers, and label mapping."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.evidence import extract_evidence, load_clue_rules
from loghound_agent_debug.schema import input_from_dict


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="data/processed/agentracer/inputs.jsonl")
    parser.add_argument("--labels", default="data/processed/agentracer/labels.jsonl")
    parser.add_argument("--rules", default="configs/clue_rules.yaml")
    parser.add_argument("--split-out", default="data/splits/evidence_split.json")
    args = parser.parse_args()
    input_rows = read_jsonl(Path(args.inputs))
    label_rows = read_jsonl(Path(args.labels))
    if len(input_rows) != len(label_rows):
        raise SystemExit(f"input/label count mismatch: {len(input_rows)} != {len(label_rows)}")
    rules = load_clue_rules(args.rules)
    tiers: Counter[str] = Counter()
    details = []
    for row, label in zip(input_rows, label_rows):
        item = input_from_dict(row)
        if item.trajectory_id != str(label["trajectory_id"]):
            raise SystemExit(f"trajectory mismatch: {item.trajectory_id} != {label['trajectory_id']}")
        ids = [step.source_step_id for step in item.steps]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"duplicate source step id: {item.trajectory_id}")
        root = str(label["root_cause_steps"][0])
        if root not in ids:
            raise SystemExit(f"root step not present: {item.trajectory_id}:{root}")
        evidence = extract_evidence(item, rules)
        tiers[evidence.tier] += 1
        details.append({
            "trajectory_id": item.trajectory_id,
            "evidence_tier": evidence.tier,
            "evidence_status": evidence.status,
            "seed_steps": [seed["source_step_id"] for seed in evidence.symptom_seeds],
            "failed_occurrences": evidence.failed_occurrences,
            "failed_clusters": evidence.failed_clusters,
            "step_count": len(item.steps),
            "root_cause_steps": label["root_cause_steps"],
        })
    result = {"schema_version": "0.1.0", "counts": {key: tiers[key] for key in ("A", "B", "C")}, "trajectories": details}
    output = Path(args.split_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
