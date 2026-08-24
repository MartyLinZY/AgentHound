#!/usr/bin/env python3
"""Render A-tier case cards and error categories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.reporting import error_categories, make_case_cards, render_case_report
from loghound_agent_debug.schema import input_from_dict, label_from_dict


def read_jsonl(path: Path) -> list[dict]:
    """Load JSONL records and ignore blank lines produced by hand edits."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="data/processed/agentracer/inputs.jsonl")
    parser.add_argument("--labels", default="data/processed/agentracer/labels.jsonl")
    parser.add_argument("--rankings", default="outputs/rankings/rankings.jsonl")
    parser.add_argument("--out-dir", default="outputs/cases")
    args = parser.parse_args()
    # Dictionaries make the join explicit and avoid depending on file order.
    inputs = {row["trajectory_id"]: input_from_dict(row) for row in read_jsonl(Path(args.inputs))}
    labels = {row["trajectory_id"]: label_from_dict(row) for row in read_jsonl(Path(args.labels))}
    rankings = read_jsonl(Path(args.rankings))
    # Case reports are intentionally limited to hard-evidence A-tier cases.
    cards = make_case_cards(inputs, labels, rankings)
    errors = error_categories(cards)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "a_tier_cases.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "case_report.md").write_text(render_case_report(cards, errors), encoding="utf-8")
    (output / "error_categories.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"a_tier_cases": len(cards), "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
