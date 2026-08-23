#!/usr/bin/env python3
"""Convert AgenTracer parquet into physically separated inputs and labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.adapters.agentracer import load_agentracer
from loghound_agent_debug.schema import input_to_dict, label_to_dict


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/agentracer/tracertraj-code-test.parquet")
    parser.add_argument("--out-dir", default="data/processed/agentracer")
    args = parser.parse_args()
    inputs, labels = load_agentracer(args.raw)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "inputs.jsonl", [input_to_dict(item) for item in inputs])
    write_jsonl(out_dir / "labels.jsonl", [label_to_dict(label) for label in labels])
    print(json.dumps({"trajectories": len(inputs), "inputs": str(out_dir / "inputs.jsonl"), "labels": str(out_dir / "labels.jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
