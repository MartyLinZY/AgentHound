#!/usr/bin/env python3
"""Evaluate rankings against the physically separate labels JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.evaluate import rows_from_results, summarize_rows
from loghound_agent_debug.schema import label_from_dict


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", default="outputs/rankings/rankings.jsonl")
    parser.add_argument("--labels", default="data/processed/agentracer/labels.jsonl")
    parser.add_argument("--out-dir", default="outputs/metrics")
    args = parser.parse_args()
    rankings = read_jsonl(Path(args.rankings))
    labels = {row["trajectory_id"]: label_from_dict(row) for row in read_jsonl(Path(args.labels))}
    rows = rows_from_results(rankings, labels)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["trajectory_id"])
        writer.writeheader()
        writer.writerows(rows)
    summaries = {"all": summarize_rows(rows)}
    for tier in ("A", "B", "C"):
        summaries[tier] = summarize_rows([row for row in rows if row["evidence_tier"] == tier])
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
