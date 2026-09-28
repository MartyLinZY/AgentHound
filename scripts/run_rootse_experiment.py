#!/usr/bin/env python3
"""Run RootSE with resumable request caches and complete-set status reporting."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.graph import build_rootse_graph
from loghound_agent_debug.rankers.loghound_static import load_config, rank_trajectory
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rootse_dependencies import DependencyExtractor
from loghound_agent_debug.rootse_operations import track_state_dependencies


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/rootse")
    parser.add_argument("--cache-dir", default="outputs/rootse/llm_cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-requests-per-trajectory", type=int, default=300)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rules", default="configs/clue_rules.yaml")
    parser.add_argument("--out-dir", default="outputs/rootse")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs, labels = load_rootse(args.root)
    if {x.trajectory_id for x in inputs} != {x.trajectory_id for x in labels}:
        raise ValueError("inputs and labels do not cover the same trajectories")
    config = load_config(args.config)
    rules = load_clue_rules(args.rules)
    extractor = DependencyExtractor(Path.cwd(), args.cache_dir, cache_only=args.cache_only,
        max_requests=args.max_requests_per_trajectory,
        request_params=config.get("rootse_llm", {}).get("request_params", {}),
        output_token_overrides=config.get("rootse_llm", {}).get("output_token_overrides", {}))
    rankings, graphs, statuses = {}, {}, {}

    def process(trajectory):
        state = [asdict(x) for x in track_state_dependencies(trajectory.steps)]
        llm, status = extractor.extract_for_trajectory(trajectory)
        if status["status"] != "complete":
            return status, None, None
        dependencies = state + llm
        graph = build_rootse_graph(len(trajectory.steps), dependencies)
        ranked = rank_trajectory(trajectory, config, rules, dependencies)
        if not ranked["graph_stats"]["pagerank_converged"]:
            status["status"] = "incomplete"
            status["failures"].append({"error": "PageRank did not converge"})
            return status, None, None
        ranked["llm_request_params"] = extractor.request_params
        graph_record = {"trajectory_id": trajectory.trajectory_id, "nodes": graph.nodes,
            "edges": [{"source_ordinal": e.source, "target_ordinal": e.target,
                "source_step_id": trajectory.steps[e.source].source_step_id,
                "target_step_id": trajectory.steps[e.target].source_step_id,
                "weight": e.weight, "relations": e.relation.split("+"),
                "evidence": e.evidence or []} for e in graph.edges]}
        return status, ranked, graph_record

    def save(state):
        ordered_ids = [x.trajectory_id for x in inputs]
        _write_jsonl(out / "rankings.jsonl", [rankings[k] for k in ordered_ids if k in rankings])
        _write_jsonl(out / "graphs.jsonl", [graphs[k] for k in ordered_ids if k in graphs])
        snapshot = {"status": state, "trajectories_expected": len(inputs),
            "trajectories_processed": len(statuses), "trajectories_ranked": len(rankings),
            "targets_completed": sum(s["targets_completed"] for s in statuses.values()),
            "http_requests_this_run": sum(s["requests"] for s in statuses.values()),
            "cache_hits": sum(s["cache_hits"] for s in statuses.values()),
            "response_reuses": sum(s.get("response_reuses", 0) for s in statuses.values()),
            "failures": [{"trajectory_id": k, **f} for k, s in statuses.items() for f in s["failures"]],
            "cache_only": args.cache_only, "workers": args.workers,
            "request_params": extractor.request_params,
            "output_token_overrides": extractor.output_token_overrides}
        extractor._atomic_write(out / "run_status.json", snapshot)
        return snapshot

    save("running")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item): item.trajectory_id for item in inputs}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                status, ranked, graph = future.result()
            except Exception as exc:
                status = {"status": "incomplete", "targets_completed": 0, "requests": 0,
                    "cache_hits": 0, "failures": [{"error": f"{type(exc).__name__}: {exc}"}]}
                ranked = graph = None
            statuses[tid] = status
            if ranked is not None:
                rankings[tid], graphs[tid] = ranked, graph
            save("running")
            print(f"[{len(statuses)}/{len(inputs)}] {tid}: {status['status']}; "
                f"{status['requests']} HTTP requests, {status['cache_hits']} cache hits", flush=True)
    complete = len(rankings) == len(inputs)
    if complete:
        evaluator_path = Path(__file__).with_name("evaluate_rootse.py")
        spec = importlib.util.spec_from_file_location("evaluate_rootse", evaluator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load exact Top-1/Top-3 evaluator")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        metrics = evaluator.evaluate(inputs, labels, [rankings[x.trajectory_id] for x in inputs])
        extractor._atomic_write(out / "metrics.json", metrics)
    snapshot = save("complete" if complete else "incomplete")
    print(json.dumps(snapshot, ensure_ascii=False), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
