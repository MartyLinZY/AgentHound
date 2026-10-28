#!/usr/bin/env python3
"""Freeze, validate, and execute five matched LLM context-view experiments."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.context_map import (VERSION, METHODS, safe_case, build_map,
                                             build_views, messages, parse_answer, tokens)
from suite_transport import RecordedClient, ExperimentAbort, save, digest, now

OUT = ROOT / "outputs/attentionmap-context-v1-20261028"
SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
GRAPH = ROOT / "outputs/rootse-ours-qwen38-nothinking-causal-v1"
TERMINAL = ROOT / "outputs/attentionmap-aa-input-audit-20261027/terminal-test-encoding-cache"
SUITE = ROOT / "outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> dict:
    old = read(ROOT / "experiments/rootse_suite/config_aliyun_qwen38_flash.json")
    keep = ("model", "endpoint", "temperature", "max_tokens", "timeout_seconds", "stream", "enable_thinking", "transport_attempts")
    return {**{k: old[k] for k in keep}, "max_calls_per_case": 1, "workers": 16,
            "context_budget": 16384, "budget_tokenizer": "tiktoken-0.12.0/cl100k_base (proxy; provider usage recorded separately)",
            "version": VERSION, "methods": list(METHODS), "runs": 1,
            "transport_retry_policy": "network/429/5xx only; no retry based on answer or accuracy"}


def guideline_text() -> str:
    path = SUITE / "snapshot/upstream/utils.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "annotation_guidelines" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing audited guidelines")


def verify_sources(manifest: dict) -> None:
    for path, expected in manifest["source_hashes"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Source changed: " + path)


def prepare(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cfg = config()
    own = [Path(__file__).resolve(), ROOT / "src/loghound_agent_debug/context_map.py", ROOT / "scripts/suite_transport.py",
           ROOT / "experiments/rootse_suite/config_aliyun_qwen38_flash.json", SUITE / "snapshot/upstream/utils.py"]
    if (out / "manifest.json").exists():
        manifest = read(out / "manifest.json")
        if manifest["config"] != cfg:
            raise ValueError("Configuration drift")
        verify_sources(manifest)
        for path, expected in manifest["prepared_hashes"].items():
            if sha(out / path) != expected:
                raise ValueError("Prepared artifact changed: " + path)
        return manifest
    hashes = {str(p): sha(p) for p in own}
    original = read(SOURCE / "manifest.json")
    guidelines = guideline_text()
    records, labels = [], {}
    started = time.monotonic()
    for record in original["cases"]:
        cid = record["case_id"]
        rawpath = Path(record["sources"]["raw"]["path"])
        graphpath = GRAPH / "graphs" / f"{cid}.json"
        featurepath = SOURCE / "features" / f"{cid}.npz"
        for p in (rawpath, graphpath, featurepath):
            hashes[str(p)] = sha(p)
        if sha(rawpath) != record["sources"]["raw"]["sha256"] or sha(featurepath) != record["feature_sha256"]:
            raise ValueError("Historical source mismatch")
        raw = read(rawpath)
        safe = safe_case(raw)
        mutated = {**raw, "failure_id": "ANSWER_SENTINEL", "failure_reason": "ANSWER_SENTINEL", "gold_patch": "ANSWER_SENTINEL"}
        if safe_case(mutated) != safe:
            raise ValueError("Label affects context")
        terminal_text = safe["test_code"] + "\n" + safe["error"]
        terminal_path = TERMINAL / (hashlib.sha256(terminal_text.encode()).hexdigest() + ".npz")
        hashes[str(terminal_path)] = sha(terminal_path)
        with np.load(featurepath, allow_pickle=False) as features, np.load(terminal_path, allow_pickle=False) as terminal:
            relation_map = build_map(safe, read(graphpath)["dependencies"], features["x"][:, :128],
                                     features["x"][:, 128:256], terminal["vector"], features["failure"][:128])
        views = build_views(safe, relation_map, cfg["context_budget"])
        save(out / "maps" / f"{cid}.json", relation_map)
        for method, view in views.items():
            request = messages(safe, view, guidelines)
            save(out / "prepared" / method / f"{cid}.json", {"case_id": cid, "method": method, "view": view,
                "messages": request, "message_tokens_proxy": sum(tokens(m["content"]) for m in request),
                "step_ids": [s["id"] for s in safe["steps"]], "messages_sha256": digest(request)})
        labels[cid] = [str(raw["failure_id"])]
        records.append({k: record[k] for k in ("case_id", "trajectory_id", "group", "repository", "n_steps")})
        print(json.dumps({"stage": "prepare", "cases": len(records), "total": len(original["cases"]),
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    save(out / "labels.json", labels)
    for path in own:
        target = out / "snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(out / "config.json", cfg)
    (out / "annotation_guidelines.txt").write_text(guidelines)
    prepared_hashes = {str(p.relative_to(out)): sha(p) for base in (out / "maps", out / "prepared") for p in base.rglob("*.json")}
    prepared_hashes["labels.json"] = sha(out / "labels.json")
    # Protect existing experiment predictions and API state records from accidental writes.
    protected = {}
    for directory in (SUITE, ROOT / "outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1", GRAPH,
                      ROOT / "outputs/rootse-influence-qwen38-nothinking-v3"):
        for pattern in ("**/state.json", "rankings/**/*.json"):
            for p in directory.glob(pattern):
                protected[str(p)] = sha(p)
    save(out / "historical_hashes.json", protected)
    manifest = {"version": VERSION, "config": cfg, "source_hashes": hashes, "prepared_hashes": prepared_hashes,
                "cases": records, "n": len(records), "groups": len({r["group"] for r in records}),
                "created_at": now(), "prepare_seconds": time.monotonic() - started,
                "historical_hashes_sha256": sha(out / "historical_hashes.json")}
    manifest["fingerprint"] = digest(manifest)
    save(out / "manifest.json", manifest)
    return manifest


def run_one(out: Path, manifest: dict, cid: str, method: str, key: str) -> dict:
    prepared = read(out / "prepared" / method / f"{cid}.json")
    case_dir = out / "runs" / method / cid
    signature = digest({"experiment": manifest["fingerprint"], "messages": prepared["messages_sha256"]})
    path = case_dir / "state.json"
    if path.exists():
        state = read(path)
        if state["signature"] != signature:
            raise ValueError("Cached result differs from frozen experiment")
        if state["status"] != "running":
            return state
    state = {"case_id": cid, "method": method, "signature": signature, "status": "running", "started_at": now()}
    save(path, state)
    try:
        answer = RecordedClient(case_dir, manifest["config"], key).create(
            model=manifest["config"]["model"], messages=prepared["messages"])
        state.update(status="valid", decision=parse_answer(answer.choices[0].message.content, prepared["step_ids"]))
    except ExperimentAbort as exc:
        if exc.status in {"model_mismatch", "replay_mismatch", "call_limit"}:
            raise RuntimeError(str(exc)) from exc
        state.update(status=exc.status, failure_reason=str(exc))
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        state.update(status="format_error", failure_reason=type(exc).__name__ + ": " + str(exc)[:300])
    state["finished_at"] = now()
    save(path, state)
    return state


def run(out: Path, manifest: dict, keypath: Path, pilot: bool = False) -> dict:
    key = keypath.read_text().strip()
    if not key or "\n" in key:
        raise ValueError("Invalid credential file")
    cases = manifest["cases"]
    # Select solely on observable size: median and largest full prompts, no labels.
    if pilot:
        ordered = sorted(cases, key=lambda c: read(out / "prepared/full" / f"{c['case_id']}.json")["message_tokens_proxy"])
        cases = [ordered[len(ordered) // 2], ordered[-1]]
    else:
        gate = read(out / "pilot_status.json")
        if not gate["technical_gate_passed"]:
            raise ValueError("Pilot did not pass")
    tasks = [(c["case_id"], m) for c in cases for m in METHODS]
    random.Random(20261028).shuffle(tasks)
    counts = Counter()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=manifest["config"]["workers"]) as pool:
        futures = [pool.submit(run_one, out, manifest, cid, method, key) for cid, method in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            state = future.result()
            counts[state["status"]] += 1
            progress = {"stage": "pilot" if pilot else "run", "completed": number, "planned": len(tasks),
                        "status_counts": dict(counts), "seconds": round(time.monotonic() - start, 1), "updated_at": now()}
            save(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    result = {**progress, "status": "complete", "case_ids": [c["case_id"] for c in cases]}
    if pilot:
        # An oversized full-reference input may overflow; bounded views must work.
        bounded = [read(out / "runs" / m / c["case_id"] / "state.json")["status"]
                   for c in cases for m in METHODS if m != "full"]
        result["technical_gate_passed"] = all(s == "valid" for s in bounded)
    save(out / ("pilot_status.json" if pilot else "run_status.json"), result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--stage", choices=("prepare", "pilot", "run", "all"), default="prepare")
    p.add_argument("--api-key-file", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.out)
        if args.stage == "prepare":
            print(json.dumps({"status": "prepared", "n": manifest["n"], "fingerprint": manifest["fingerprint"]}))
            return 0
        if not args.api_key_file:
            p.error("--api-key-file required")
        if args.stage in ("pilot", "all"):
            gate = run(args.out, manifest, args.api_key_file, pilot=True)
            if not gate["technical_gate_passed"]:
                return 2
        if args.stage in ("run", "all"):
            run(args.out, manifest, args.api_key_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
