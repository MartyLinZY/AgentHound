#!/usr/bin/env python3
"""Offline probe for the candidate-set precision gap (no HTTP, no labels in scoring).

Reports four things used by docs/通道2定位与候选集精排改进方案_2026-10-26.md:

1. channel ablation under weighted RRF (semantic / path / anchor / action_type);
2. root-cause rank distribution inside the candidate pool;
3. what the top-1 confusion looks like when the root cause *is* in the pool;
4. whether candidate compression changes end-to-end Top-1.

Labels are read only at evaluation time, never inside a channel.

Usage:
    python scripts/probe_channel_gap.py \
        --root data/raw/rootse \
        --graphs outputs/baseline/graphs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.features import anchor as anchor_feature
from loghound_agent_debug.features.semantic import scores as semantic_scores
from loghound_agent_debug.graph import Edge, Graph, pagerank

# One-shot prior used for the probe only.  A production implementation should
# derive these from per-agent lift instead of hard-coding them.
PRIOR = {
    "swe-agent": {"edit": 1.0, "reasoning": 0.9, "shell": 0.25, "explore": 0.15, "other": 0.3},
    "swe-agent-live": {"edit": 1.0, "reasoning": 0.9, "shell": 0.25, "explore": 0.15, "other": 0.3},
    "openhands": {"edit": 1.0, "reasoning": 1.0, "shell": 0.4, "explore": 0.3, "other": 0.3},
    "auto-code-rover": {"edit": 1.0, "reasoning": 0.9, "shell": 0.6, "explore": 0.3, "other": 0.3},
}
SHELL_HINT = re.compile(
    r"^(?:python3?|pytest|pip|npm|npx|yarn|node|git|make|go|mvn|gradle|cd|bash|sh|source|conda|"
    r"docker|curl|wget|chmod|mkdir|rm|cp|mv|export|sudo|apt|apt-get|gcc|g\+\+|cargo|ruby|perl|"
    r"tox|coverage|ls\b.*&&|echo)\b", re.I)
EXPLORE_HINT = re.compile(r"^(?:ls|cat|head|tail|less|more|grep|rg|find|tree|file|wc|sed -n|awk)\b", re.I)


def classify_action(raw) -> str:
    """Map one raw RootSE action (dict for OpenHands, list for ACR, str for swe-agent)."""
    if isinstance(raw, dict):
        tool = str(raw.get("tool") or "")
        payload = raw.get("input")
        command = str(payload.get("command") or "") if isinstance(payload, dict) else ""
        if "str_replace_editor" in tool:
            if command == "view":
                return "explore"
            return "edit" if command in ("create", "str_replace", "insert") else "other"
        if tool in ("bash", "execute_bash", "run", "execute"):
            return "shell"
        return "other"
    if isinstance(raw, list):
        names = [str(item.get("func_name") or "") for item in raw if isinstance(item, dict)]
        if any(n.startswith(("write_patch", "create_file", "modify")) for n in names):
            return "edit"
        if any(n.startswith(("search_", "get_", "list_", "find_", "read_")) for n in names):
            return "explore"
        if any(n.startswith(("run_", "execute", "test")) for n in names):
            return "shell"
        return "other"
    text = (raw if isinstance(raw, str) else "").strip()
    if not text:
        return "reasoning"
    low = text.lower()
    if "str_replace_editor" in low:
        if " view" in low:
            return "explore"
        return "edit" if (" create" in low or " str_replace" in low or " insert" in low) else "other"
    if re.search(r"(?:^|\n)\s*(?:cat|tee)\s+[^\n]*<<", low) or "apply_patch" in low or re.search(r"sed\s+-i", low):
        return "edit"
    if re.search(r"(?:^|\s)(?:python3?\s+-\s*<<|pytest|go test|npm test|tox)\b", low):
        return "shell"
    if EXPLORE_HINT.search(low):
        # A shell redirect into a source file is a write, not a read.
        return "edit" if re.search(r">\s*\S+\.\w+$", low) else "explore"
    return "shell" if SHELL_HINT.search(low) else "other"


def ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    out = [0] * len(values)
    for rank, index in enumerate(order, 1):
        out[index] = rank
    return out


def rrf(channels: dict[str, tuple[list[float], float]], k: int = 60) -> list[float]:
    size = len(next(iter(channels.values()))[0])
    fused = [0.0] * size
    for values, weight in channels.values():
        if weight <= 0 or max(values) == min(values):
            continue
        for index, rank in enumerate(ranks(values)):
            fused[index] += weight / (k + rank)
    return fused


def load_graphs(path: Path) -> dict[str, Graph]:
    graphs = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            edges = [Edge(e["source_ordinal"], e["target_ordinal"], e.get("weight", 0.1),
                          "+".join(e["relations"]), e.get("evidence")) for e in row["edges"]]
            graphs[row["trajectory_id"]] = Graph(row["nodes"], edges)
    return graphs


def load_raw_actions(root: Path) -> dict[str, list]:
    actions = {}
    for path in sorted((root / "data").rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root).as_posix().removeprefix("data/").removesuffix(".json")
        actions["rootse/" + rel] = [step.get("action") for step in raw["original_traj"]]
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/rootse")
    parser.add_argument("--graphs", default="outputs/baseline/graphs.jsonl")
    parser.add_argument("--task-weight", type=float, default=0.4)
    args = parser.parse_args()

    inputs, labels = load_rootse(args.root)
    graphs = load_graphs(Path(args.graphs))
    actions = load_raw_actions(Path(args.root))
    label_by = {label.trajectory_id: label for label in labels}

    rows = []
    kind_counter = Counter()
    for item in inputs:
        graph = graphs[item.trajectory_id]
        pagerank_scores, _, _ = pagerank(graph, 0.85, 1e-6, 1000)
        kinds = [classify_action(raw) for raw in actions[item.trajectory_id]]
        kind_counter.update(kinds)
        rows.append({
            "id": item.trajectory_id,
            "agent": item.steps[0].agent,
            "n": len(item.steps),
            "root": int(label_by[item.trajectory_id].root_cause_steps[0]),
            "semantic": semantic_scores(item, " ".join(o.content for o in item.failure_observations), args.task_weight, False),
            "path": pagerank_scores,
            "anchor": anchor_feature.scores(item),
            "kinds": kinds,
        })

    total = sum(kind_counter.values())
    print("== action distribution ==")
    print({k: round(v / total, 3) for k, v in kind_counter.most_common()})

    def prior(row):
        table = PRIOR[row["agent"]]
        return [table[kind] for kind in row["kinds"]]

    configs = {
        "sem+path": lambda r: {"sem": (r["semantic"], 1.0), "path": (r["path"], 0.5)},
        "sem+path+anchor": lambda r: {"sem": (r["semantic"], 1.0), "path": (r["path"], 0.5), "anc": (r["anchor"], 0.5)},
        "sem+path+action": lambda r: {"sem": (r["semantic"], 1.0), "path": (r["path"], 0.5), "act": (prior(r), 1.0)},
        "sem+path+anchor+action": lambda r: {"sem": (r["semantic"], 1.0), "path": (r["path"], 0.5),
                                             "anc": (r["anchor"], 0.5), "act": (prior(r), 1.0)},
        "action only": lambda r: {"act": (prior(r), 1.0)},
    }

    print("\n== channel ablation (102 trajectories, weighted RRF k=60) ==")
    best_orders = {}
    for name, build in configs.items():
        hits = Counter()
        hit_at_12 = 0
        for row in rows:
            fused = rrf(build(row))
            order = sorted(range(row["n"]), key=lambda i: (-fused[i], i))
            position = order.index(row["root"]) + 1
            for k in (1, 3, 5, 10):
                hits[k] += position <= k
            hit_at_12 += position <= 12
            if name == "sem+path+anchor+action":
                best_orders[row["id"]] = (order, fused)
        print(f"{name:26s} H1={hits[1]:3d} H3={hits[3]:3d} H5={hits[5]:3d} H10={hits[10]:3d} "
              f"R@12={hit_at_12:3d} P(1|12)={hits[1]/hit_at_12:.3f}" if hit_at_12 else name)

    print("\n== where the pool precision is lost (K=10) ==")
    rank_hist = Counter()
    confusion = Counter()
    gaps = []
    if not best_orders:
        print("rank distribution skipped: best config not evaluated")
        return 0
    for row in rows:
        order, fused = best_orders[row["id"]]
        position = order.index(row["root"]) + 1
        rank_hist[position if position <= 10 else "miss"] += 1
        if position <= 10 and position > 1:
            confusion[(row["kinds"][order[0]], row["kinds"][row["root"]])] += 1
            gaps.append((fused[order[0]] - fused[row["root"]]) / (fused[order[0]] or 1))
    ordered = {k: rank_hist[k] for k in range(1, 11)}
    print({**ordered, "miss": rank_hist["miss"]})
    print("median relative score gap (#1 - root)/#1:", round(sorted(gaps)[len(gaps) // 2], 4) if gaps else None)
    same_type = sum(v for (a, b), v in confusion.items() if a == b)
    print(f"same-type confusion: {same_type}/{sum(confusion.values())}")
    for (a, b), count in confusion.most_common(6):
        print(f"  {a:9s} -> {b:9s} : {count}")

    print("\n== compression vs recall ==")
    keep_root = Counter()
    top1 = Counter()
    for row in rows:
        order, fused = best_orders[row["id"]]
        kinds, anchor = row["kinds"], row["anchor"]
        n = row["n"]
        r1 = {i for i in range(n)
              if kinds[i] in ("edit", "reasoning") or anchor[i] > 0 or row["agent"] == "auto-code-rover"
              or any(0 <= j < n and kinds[j] == "edit" and abs(j - i) <= 1 for j in (i - 1, i + 1))}
        r2 = {i for i in r1 if i / max(1, n - 1) <= 0.85 or anchor[i] > 0}
        top12 = set(order[:12])
        for name, pool in (("all", set(range(n))), ("R1", r1), ("R1+pos", r2),
                           ("top12", top12), ("top12&R1+pos", top12 & r2)):
            keep_root[name] += row["root"] in pool
            if row["root"] in pool:
                top1[name] += min(pool, key=lambda i: (-fused[i], i)) == row["root"]
    for name in ("all", "R1", "R1+pos", "top12", "top12&R1+pos"):
        print(f"{name:14s} root-kept={keep_root[name]:3d}/{len(rows)}  end-to-end Top-1={top1[name]:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
