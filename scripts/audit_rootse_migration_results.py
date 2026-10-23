#!/usr/bin/env python3
"""Offline independent arithmetic and same-graph audit; never calls a model."""

import argparse
from datetime import datetime, timezone
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
from loghound_agent_debug.adapters.rootse import load_rootse

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/rootse-migration-state-v2-deepseek-v41-flash-32k",
    )
    a = p.parse_args(argv)
    out = a.out.resolve()

    def read(path: Path) -> dict:
        return json.loads(path.read_text())

    m = read(out / "metrics.json")
    manifest = read(out / "manifest.json")
    inputs, labels = load_rootse(
        ROOT.parent / "TrajAudit/RootSE", protocol="postmortem"
    )
    gold = {x.trajectory_id: set(x.root_cause_steps) for x in labels}
    ordinals = {
        x.trajectory_id: {s.source_step_id: s.ordinal for s in x.steps} for x in inputs
    }
    shared = defaultdict(set)
    checked = {}
    for name, metric in m["variants"].items():
        rows = [
            json.loads(x)
            for x in (out / "ablations" / name / "rankings.jsonl")
            .read_text()
            .splitlines()
        ]
        if not (len(rows) == len({r["trajectory_id"] for r in rows}) == 102):
            raise ValueError(
                "Integrity check failed: len(rows)==len({r['trajectory_id'] for r in rows})==102"
            )
        tolerance = [0] * 4
        top = [0] * 3
        reciprocal = 0.0
        strict = [0] * 4
        for row in rows:
            tid = row["trajectory_id"]
            ranked = [str(r["source_step_id"]) for r in row["ranking"]]
            if not (
                len(ranked) == len(set(ranked)) == len(ordinals[tid])
                and set(ranked) == set(ordinals[tid])
            ):
                raise ValueError(
                    "Integrity check failed: len(ranked)==len(set(ranked))==len(ordinals[tid]) and set(ranked)==set(ordinals[tid])"
                )
            rank = min(ranked.index(g) + 1 for g in gold[tid])
            distance = min(
                abs(ordinals[tid][ranked[0]] - ordinals[tid][g]) for g in gold[tid]
            )
            reciprocal += 1 / rank
            for k in range(4):
                tolerance[k] += distance <= k
                strict[k] += distance <= k and row["extraction_status"] == "complete"
            for k, cutoff in enumerate([1, 3, 5]):
                top[k] += rank <= cutoff
            shared[tid].add(row["dependency_hash"])
        if not (tolerance == metric["tolerance_hits"]):
            raise ValueError(
                "Integrity check failed: tolerance==metric['tolerance_hits']"
            )
        if not (strict == metric["strict_complete_graph_tolerance_hits"]):
            raise ValueError(
                "Integrity check failed: strict==metric['strict_complete_graph_tolerance_hits']"
            )
        if not (top == [metric["top1_hits"], metric["top3_hits"], metric["top5_hits"]]):
            raise ValueError(
                "Integrity check failed: top==[metric['top1_hits'],metric['top3_hits'],metric['top5_hits']]"
            )
        if not (abs(reciprocal / 102 - metric["mrr"]) < 1e-12):
            raise ValueError(
                "Integrity check failed: abs(reciprocal/102-metric['mrr'])<1e-12"
            )
        checked[name] = {
            "tolerance_hits": tolerance,
            "top1_3_5": top,
            "mrr": reciprocal / 102,
            "strict_complete_hits": strict,
        }
    if not (len(shared) == 102 and all(len(v) == 1 for v in shared.values())):
        raise ValueError(
            "Integrity check failed: len(shared)==102 and all(len(v)==1 for v in shared.values())"
        )
    status = Counter()
    failed_tokens = 0
    unknown = 0
    http = Counter()
    attempts = 0
    known = 0
    for t in manifest["targets"]:
        folder = out / t["directory"]
        s = read(folder / "state.json")
        status[s["status"]] += 1
        for f in folder.glob("calls/*/state.json"):
            for attempt in read(f)["attempts"]:
                attempts += 1
                http[str(attempt.get("http_status", "transport_error"))] += 1
                usage = attempt.get("usage")
                # Preserve transport's normalized token accounting; no unknown->zero claim.
                if usage is None:
                    unknown += 1
                else:
                    total = usage.get("total_tokens", 0)
                    known += total
                    if s["status"] != "complete":
                        failed_tokens += total
    if not (dict(status) == m["extraction_status_counts"]):
        raise ValueError(
            "Integrity check failed: dict(status)==m['extraction_status_counts']"
        )
    # State transport structure is recorded below; aggregate accounting is checked independently.
    expected = read(out / "verification_before_hashes.json")
    if not (
        all(
            hashlib.sha256((out / rel).read_bytes()).hexdigest() == h
            for rel, h in expected.items()
        )
    ):
        raise ValueError(
            "Integrity check failed: all(hashlib.sha256((out/rel).read_bytes()).hexdigest()==h for rel,h in expected.items())"
        )
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "frozen_fingerprint": manifest["fingerprint"],
        "all_checks_passed": True,
        "n": 102,
        "checked_variants": checked,
        "same_dependency_hash_in_all_eight_variants": True,
        "state_and_ranking_files_unchanged": len(expected),
        "target_statuses": dict(status),
        "attempts": attempts,
        "known_tokens_from_raw_attempts": known,
        "unknown_usage_attempts": unknown,
        "terminal_failed_target_known_tokens": failed_tokens,
        "http_status_counts": dict(http),
    }
    if not (attempts == m["accounting"]["api_requests"]):
        raise ValueError(
            "Integrity check failed: attempts==m['accounting']['api_requests']"
        )
    if not (known == m["accounting"]["known_total_tokens"]):
        raise ValueError(
            "Integrity check failed: known==m['accounting']['known_total_tokens']"
        )
    if not (unknown == m["accounting"]["attempts_without_usage"]):
        raise ValueError(
            "Integrity check failed: unknown==m['accounting']['attempts_without_usage']"
        )
    (out / "analysis/independent_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "checked_variants"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
