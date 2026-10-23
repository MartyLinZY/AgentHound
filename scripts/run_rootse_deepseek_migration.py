#!/usr/bin/env python3
"""Freeze and run independent per-step dependency requests, with no gold evaluation."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import threading
from deepseek_migration_core import build_target, run_target, ROOT
from deepseek_suite_transport import save, digest, now
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.schema import input_to_dict, input_from_dict
from loghound_agent_debug.rootse_operations import track_state_dependencies
from dataclasses import asdict

DEFAULT = ROOT / "outputs/rootse-migration-state-v2-deepseek-official-flash-thinking-32k-v1"


def prepare(source: Path, out: Path, config: dict) -> dict:
    expected = json.loads(
        (ROOT / "experiments/rootse_full/dataset_manifest.json").read_text()
    )
    actual = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source / "RootSE").rglob("*.json")
    }
    if actual != expected or len(actual) != 102:
        raise ValueError("Frozen RootSE changed")
    code_paths = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "scripts" / n
        for n in ["deepseek_migration_core.py", "run_rootse_deepseek_migration.py", "deepseek_suite_transport.py"]
    ]
    code_paths += [ROOT / "configs/default.yaml", ROOT / "configs/clue_rules.yaml"]
    code_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_paths
    }
    protocol = {"config": config, "dataset": actual, "code": code_hashes}
    fp = digest(protocol)
    if (out / "manifest.json").exists():
        m = json.loads((out / "manifest.json").read_text())
        if m["fingerprint"] != fp:
            raise ValueError("Frozen graph protocol changed; use a new directory")
        return m
    items, _ = load_rootse(source / "RootSE", protocol="trace-only")
    targets = []
    cases = []
    groups = defaultdict(list)
    for item in items:
        cid = hashlib.sha256(item.trajectory_id.encode()).hexdigest()[:16]
        save(out / "inputs" / f"{cid}.json", input_to_dict(item))
        save(
            out / "rule_edges" / f"{cid}.json",
            [asdict(e) for e in track_state_dependencies(item.steps)],
        )
        case = {
            "case_id": cid,
            "trajectory_id": item.trajectory_id,
            "steps": len(item.steps),
            "targets": [],
            "skipped": [],
        }
        for ordinal in range(len(item.steps)):
            req, meta = build_target(item, ordinal, config)
            if req is None:
                if meta["status"] not in {"no_candidates", "no_target_content"}:
                    raise ValueError("Input guard triggered")
                case["skipped"].append(meta)
                continue
            payload = {
                "model": config["model"],
                "messages": req["model_messages"],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            }
            record = {
                "case_id": cid,
                "target_ordinal": ordinal,
                "directory": f"targets/{cid}/{ordinal:04d}",
                "request_sha256": digest(payload),
                "input_chars": meta["messages_chars"],
            }
            targets.append(record)
            case["targets"].append(record)
        cases.append(case)
        group = "/".join(item.trajectory_id.split("/")[1:3])
        groups[group].append(case)
    pilot = []
    for group, rows in sorted(groups.items()):
        candidates = sorted(rows, key=lambda c: (c["steps"], c["trajectory_id"]))
        case = candidates[len(candidates) // 2]
        pilot.append(
            max(case["targets"], key=lambda t: (t["input_chars"], t["directory"]))[
                "directory"
            ]
        )
    pilot.append(
        max(targets, key=lambda t: (t["input_chars"], t["directory"]))["directory"]
    )
    snapshot = out / "snapshot"
    for p in code_paths:
        dest = snapshot / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    save(snapshot / "config.json", config)
    m = {
        "status": "prepared",
        "fingerprint": fp,
        "protocol": protocol,
        "cases": cases,
        "targets": targets,
        "pilot_directories": sorted(set(pilot)),
        "created_at": now(),
        "n_cases": len(cases),
        "n_targets": len(targets),
        "max_input_chars": max(t["input_chars"] for t in targets),
    }
    save(out / "manifest.json", m)
    return m


def run(manifest: dict, out: Path, key_path: Path, pilot: bool) -> dict:
    config = manifest["protocol"]["config"]
    key = key_path.read_text().strip()
    if not key:
        raise ValueError("Empty credential file")
    selected = [
        t
        for t in manifest["targets"]
        if not pilot or t["directory"] in manifest["pilot_directories"]
    ]
    base = out / "pilot" if pilot else out
    items = {
        c["case_id"]: input_from_dict(
            json.loads((out / "inputs" / f"{c['case_id']}.json").read_text())
        )
        for c in manifest["cases"]
    }
    stop = threading.Event()
    counts = Counter()

    def job(target: dict) -> dict:
        if stop.is_set():
            return {"status": "not_started"}
        state = run_target(
            items[target["case_id"]],
            target["target_ordinal"],
            base / target["directory"],
            config,
            key,
        )
        if state["request_sha256"] != target["request_sha256"]:
            raise ValueError("Prepared request mismatch")
        # Persistent authentication/quota failures require external action, not thousands of retries.
        if state["status"] == "service_error":
            p = base / target["directory"] / "calls/0001/state.json"
            call = json.loads(p.read_text())
            if any(
                a.get("http_status") in {401, 402, 403, 429} for a in call["attempts"]
            ):
                stop.set()
        return state

    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [
            pool.submit(job, t)
            for t in sorted(selected, key=lambda t: (-t["input_chars"], t["directory"]))
        ]
        for n, future in enumerate(as_completed(futures), 1):
            state = future.result()
            counts[state["status"]] += 1
            if n % 10 == 0 or n == len(selected):
                print(
                    json.dumps(
                        {
                            "terminal_seen": n - counts["not_started"],
                            "planned": len(selected),
                            "counts": dict(counts),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                save(
                    base / "progress.json",
                    {
                        "updated_at": now(),
                        "planned": len(selected),
                        "seen": n,
                        "counts": dict(counts),
                    },
                )
    status = "blocked" if stop.is_set() else "complete"
    summary = {
        "status": status,
        "planned": len(selected),
        "status_counts": dict(counts),
        "finished_at": now(),
    }
    if pilot:
        completed = [
            json.loads((base / t["directory"] / "state.json").read_text())
            for t in selected
            if (base / t["directory"] / "state.json").exists()
        ]
        edges = sum(len(s.get("validated_dependencies", [])) for s in completed)
        gate = config["pilot_gate"]
        passed = (
            status == "complete"
            and counts["complete"] / len(selected) >= gate["minimum_complete_fraction"]
            and edges >= gate["minimum_validated_edges"]
            and not any(counts[x] for x in gate["forbidden_statuses"])
        )
        summary.update(technical_gate_passed=passed, validated_edges=edges)
    save(base / "run_status.json", summary)
    if not pilot:
        for case in manifest["cases"]:
            results = [
                json.loads((out / t["directory"] / "state.json").read_text())
                for t in case["targets"]
                if (out / t["directory"] / "state.json").exists()
            ]
            if len(results) != len(case["targets"]):
                continue
            llm = [e for s in results for e in s.get("validated_dependencies", [])]
            rules = json.loads(
                (out / "rule_edges" / f"{case['case_id']}.json").read_text()
            )
            save(
                out / "graphs" / f"{case['case_id']}.json",
                {
                    "trajectory_id": case["trajectory_id"],
                    "status": "complete"
                    if all(s["status"] == "complete" for s in results)
                    else "partial",
                    "targets": len(results),
                    "target_status_counts": dict(Counter(s["status"] for s in results)),
                    "rule_edge_count": len(rules),
                    "llm_edge_count": len(llm),
                    "dependencies": rules + llm,
                },
            )
        manifest.update(status=status, finished_at=now())
        save(out / "manifest.json", manifest)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT.parent / "TrajAudit")
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument(
        "--config", type=Path, default=ROOT / "experiments/rootse_migration/config_deepseek_official_flash.json"
    )
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--workers", type=int, help="Concurrent dependency requests")
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text())
    if a.workers is not None:
        config["workers"] = a.workers
    if config["workers"] < 1:
        p.error("--workers must be positive")
    if (
        config["model"] != "deepseek-flash"
        or config["endpoint"]
        != "https://api.deepseek.com/chat/completions"
    ):
        raise ValueError("Unapproved deployment")
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = prepare(a.source.resolve(), out, config)
        if a.prepare_only:
            print(
                json.dumps(
                    {
                        k: m[k]
                        for k in [
                            "fingerprint",
                            "n_cases",
                            "n_targets",
                            "max_input_chars",
                            "pilot_directories",
                        ]
                    },
                    indent=2,
                )
            )
            return 0
        if a.api_key_file is None:
            p.error("--api-key-file required")
        if not a.pilot:
            pilot_status = out / "pilot/run_status.json"
            if not pilot_status.exists() or not json.loads(
                pilot_status.read_text()
            ).get("technical_gate_passed"):
                raise ValueError("Technical pilot must finish before full run")
        result = run(m, out, a.api_key_file, a.pilot)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" and (not a.pilot or result["technical_gate_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
