#!/usr/bin/env python3
"""Audit all RootSE operations and compare fixed-graph rankings without API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rankers.anchor_path import VARIANTS, rank_trajectory
from loghound_agent_debug.rankers.loghound_static import load_config
from loghound_agent_debug.rootse_operations import (
    parse_operations,
    track_state_dependencies,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify(frozen_repo: Path, output: Path) -> dict:
    frozen = (
        frozen_repo / "outputs/rootse-migration-fullfields-deepseek-v41-flash-32k-v1"
    )
    manifest = read(frozen / "manifest.json")
    for rel, expected in manifest["protocol"]["code"].items():
        if hashlib.sha256((frozen_repo / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen code changed: {rel}")
    expected_hashes = read(frozen / "verification_before_hashes.json")
    for rel, expected in expected_hashes.items():
        if hashlib.sha256((frozen / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen prediction/state changed: {rel}")
    source = frozen_repo.parent / "TrajAudit"
    for rel, expected in manifest["protocol"]["dataset"].items():
        if hashlib.sha256((source / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset changed: {rel}")
    inputs, _ = load_rootse(source / "RootSE")
    cases = {row["trajectory_id"]: row for row in manifest["cases"]}
    operation_counts = Counter()
    statuses = Counter()
    rows = []
    for item in inputs:
        operations = [op for step in item.steps for op in parse_operations(step)]
        operation_counts.update(op.kind for op in operations)
        statuses.update(op.status for op in operations)
        edges = track_state_dependencies(item.steps)
        if not all(
            0 <= e.source_ordinal < e.target_ordinal < len(item.steps) for e in edges
        ):
            raise ValueError("Invalid state edge direction or step ID")
        old = read(
            frozen / "rule_edges" / f"{cases[item.trajectory_id]['case_id']}.json"
        )
        rows.append(
            {
                "trajectory_id": item.trajectory_id,
                "old_rule_edges": len(old),
                "new_rule_edges": len(edges),
                "dependencies": [asdict(e) for e in edges],
            }
        )
    config = load_config(frozen / "snapshot/configs/default.yaml")
    rules = load_clue_rules(frozen / "snapshot/configs/clue_rules.yaml")
    checked = 0
    for protocol in ["trace-only", "postmortem"]:
        items, _ = load_rootse(source / "RootSE", protocol=protocol)
        for variant in VARIANTS:
            previous = {
                r["trajectory_id"]: r
                for r in map(
                    json.loads,
                    (frozen / "ablations" / protocol / variant / "rankings.jsonl")
                    .read_text()
                    .splitlines(),
                )
            }
            for item in items:
                graph = read(
                    frozen / "graphs" / f"{cases[item.trajectory_id]['case_id']}.json"
                )
                result = rank_trajectory(
                    item, config, rules, graph["dependencies"], variant
                )
                if result["ranking"] != previous[item.trajectory_id]["ranking"]:
                    raise ValueError(
                        f"Ranking behavior changed on a fixed graph: {protocol}/{variant}/{item.trajectory_id}"
                    )
                checked += 1
            print(
                f"Checked {protocol}/{variant}: {len(items)} fixed-graph rankings",
                flush=True,
            )
    summary = {
        "api_requests": 0,
        "trajectories": len(inputs),
        "steps": sum(len(i.steps) for i in inputs),
        "state_tracker": "explicit_operations_v2",
        "operation_counts": dict(operation_counts),
        "operation_statuses": dict(statuses),
        "old_rule_edges": sum(r["old_rule_edges"] for r in rows),
        "new_rule_edges": sum(r["new_rule_edges"] for r in rows),
        "fixed_graph_rankings_unchanged": checked,
        "frozen_state_and_ranking_hashes_unchanged": len(expected_hashes),
        "caveat": "Rule-edge counts are coverage diagnostics, not semantic accuracy or localization results. No new LLM inference or new-graph localization evaluation.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "state_edges.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-repo", type=Path, default=ROOT.parent / "AgentHound")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ppt-code-audit")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify(args.frozen_repo.resolve(), args.out.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
