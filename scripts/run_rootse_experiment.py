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
from loghound_agent_debug.rootse_misstep import MisstepJudge, score_map
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
    parser.add_argument("--cache-dir", default="outputs/rootse/dependency_cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-requests-per-trajectory", type=int, default=300)
    parser.add_argument("--misstep-cache-dir", default="outputs/rootse/misstep_cache")
    parser.add_argument("--misstep-cache-only", action="store_true")
    parser.add_argument("--skip-misstep", action="store_true")
    parser.add_argument("--resume", action="store_true",
        help="reuse existing rankings/graphs/misstep records and process only trajectories without a ranking")
    parser.add_argument("--resume-failures", action="store_true",
        help="resume only prior failed trajectories that still have misstep request budget")
    parser.add_argument("--pilot-trajectories", type=int, default=0, help="run one trajectory per agent family for a small pilot")
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
    if args.pilot_trajectories:
        by_agent = {}
        for item in inputs:
            by_agent.setdefault(item.steps[0].agent, item)
        preferred = ["swe-agent", "openhands", "auto-code-rover", "swe-agent-live"]
        chosen = [by_agent[name] for name in preferred if name in by_agent][:args.pilot_trajectories]
        if len(chosen) < args.pilot_trajectories:
            chosen.extend(item for item in inputs if item not in chosen and len(chosen) < args.pilot_trajectories)
        chosen_ids = {x.trajectory_id for x in chosen}
        inputs = chosen
        labels = [x for x in labels if x.trajectory_id in chosen_ids]
    if {x.trajectory_id for x in inputs} != {x.trajectory_id for x in labels}:
        raise ValueError("inputs and labels do not cover the same trajectories")
    config = load_config(args.config)
    rules = load_clue_rules(args.rules)
    extractor = DependencyExtractor(Path.cwd(), args.cache_dir, cache_only=args.cache_only,
        max_requests=args.max_requests_per_trajectory,
        request_params=config.get("rootse_llm", {}).get("request_params", {}),
        output_token_overrides=config.get("rootse_llm", {}).get("output_token_overrides", {}))
    misstep_cfg = config.get("rootse_misstep", {})
    judge = None if args.skip_misstep else MisstepJudge(Path.cwd(), args.misstep_cache_dir, misstep_cfg,
        cache_only=args.misstep_cache_only)
    rankings, graphs, statuses = {}, {}, {}
    missteps = {}

    def read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if args.resume or args.resume_failures:
        if not (out / "rankings.jsonl").exists():
            parser.error("resume mode requires existing outputs in " + str(out))
        prior_status_path = out / "run_status.json"
        prior_status = json.loads(prior_status_path.read_text(encoding="utf-8")) if prior_status_path.exists() else {}
        rankings.update((row["trajectory_id"], row) for row in read_jsonl(out / "rankings.jsonl"))
        graphs.update((row["trajectory_id"], row) for row in read_jsonl(out / "graphs.jsonl"))
        if not args.skip_misstep:
            missteps.update((row["trajectory_id"], row) for row in read_jsonl(out / "misstep.jsonl"))
        # Existing rankings are complete work. Seed them into the status table so
        # the final status and metrics cover the full dataset while only missing
        # trajectories are submitted to the worker pool.
        for trajectory_id in rankings:
            statuses[trajectory_id] = {
                "status": "complete", "targets_completed": 0, "requests": 0,
                "cache_hits": 0, "failures": [],
                "misstep": {"status": "complete", "requests": 0,
                    "cache_hits": 0, "chunks_completed": 0, "failures": []},
            }
        if args.resume_failures:
            retryable_ids = set()
            for failure in prior_status.get("failures", []):
                trajectory_id = failure["trajectory_id"]
                messages = [str(row.get("error", "")) for row in failure.get("failures", [])]
                if trajectory_id not in rankings and judge is not None \
                        and judge.counts.get(trajectory_id, 0) < judge.max_requests \
                        and not any("max_requests_per_trajectory exhausted" in message for message in messages):
                    retryable_ids.add(trajectory_id)
                elif trajectory_id not in rankings:
                    statuses[trajectory_id] = {
                        "status": "incomplete", "targets_completed": 0, "requests": 0,
                        "cache_hits": 0, "failures": [{"stage": failure.get("stage", "resume"),
                            "failures": failure.get("failures", [])}],
                        "misstep": {"status": "incomplete", "requests": 0,
                            "cache_hits": 0, "chunks_completed": 0,
                            "failures": failure.get("failures", [])},
                    }
            pending_inputs = [item for item in inputs if item.trajectory_id in retryable_ids]
        else:
            pending_inputs = [item for item in inputs if item.trajectory_id not in rankings]
    else:
        pending_inputs = inputs

    def process(trajectory):
        state = [asdict(x) for x in track_state_dependencies(trajectory.steps)]
        llm, status = extractor.extract_for_trajectory(trajectory)
        if status["status"] != "complete":
            return status, None, None, None
        dependencies = state + llm
        misstep_record = None
        misstep_status = {"status": "skipped", "requests": 0, "cache_hits": 0, "chunks_completed": 0, "failures": []}
        misstep_scores = None
        if judge is not None:
            misstep_record, misstep_status = judge.judge(trajectory)
            if misstep_status["status"] != "complete":
                status["status"] = "incomplete"
                status["failures"].append({"stage": "misstep", "failures": misstep_status["failures"]})
                return status, None, None, misstep_record
            misstep_scores = score_map(misstep_record, trajectory.steps)
        graph = build_rootse_graph(len(trajectory.steps), dependencies)
        ranked = rank_trajectory(trajectory, config, rules, dependencies, misstep_scores=misstep_scores)
        if not ranked["graph_stats"]["pagerank_converged"]:
            status["status"] = "incomplete"
            status["failures"].append({"error": "PageRank did not converge"})
            return status, None, None, misstep_record
        ranked["llm_request_params"] = extractor.request_params
        graph_record = {"trajectory_id": trajectory.trajectory_id, "nodes": graph.nodes,
            "edges": [{"source_ordinal": e.source, "target_ordinal": e.target,
                "source_step_id": trajectory.steps[e.source].source_step_id,
                "target_step_id": trajectory.steps[e.target].source_step_id,
                "weight": e.weight, "relations": e.relation.split("+"),
                "evidence": e.evidence or []} for e in graph.edges]}
        status["misstep"] = misstep_status
        return status, ranked, graph_record, misstep_record

    def save(state):
        ordered_ids = [x.trajectory_id for x in inputs]
        _write_jsonl(out / "rankings.jsonl", [rankings[k] for k in ordered_ids if k in rankings])
        _write_jsonl(out / "graphs.jsonl", [graphs[k] for k in ordered_ids if k in graphs])
        if not args.skip_misstep:
            _write_jsonl(out / "misstep.jsonl", [missteps[k] for k in ordered_ids if k in missteps])
        snapshot = {"status": state, "trajectories_expected": len(inputs),
            "trajectories_processed": len(statuses), "trajectories_ranked": len(rankings),
            "targets_completed": sum(s["targets_completed"] for s in statuses.values()),
            "http_requests_this_run": sum(s["requests"] for s in statuses.values()),
            "cache_hits": sum(s["cache_hits"] for s in statuses.values()),
            "response_reuses": sum(s.get("response_reuses", 0) for s in statuses.values()),
            "failures": [{"trajectory_id": k, **f} for k, s in statuses.items() for f in s["failures"]],
            "cache_only": args.cache_only, "workers": args.workers,
            "misstep_cache_only": args.misstep_cache_only,
            "misstep_skipped": args.skip_misstep,
            "dependency_cache_dir": str(Path(args.cache_dir)),
            "misstep_cache_dir": None if args.skip_misstep else str(Path(args.misstep_cache_dir)),
            "request_params": extractor.request_params,
            "output_token_overrides": extractor.output_token_overrides}
        if not args.skip_misstep:
            snapshot["misstep_chunks_completed"] = sum(s.get("misstep", {}).get("chunks_completed", 0) for s in statuses.values())
            snapshot["misstep_http_requests_this_run"] = sum(s.get("misstep", {}).get("requests", 0) for s in statuses.values())
            snapshot["misstep_cache_hits"] = sum(s.get("misstep", {}).get("cache_hits", 0) for s in statuses.values())
        extractor._atomic_write(out / "run_status.json", snapshot)
        return snapshot

    save("running")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item): item.trajectory_id for item in pending_inputs}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                status, ranked, graph, misstep_record = future.result()
            except Exception as exc:
                status = {"status": "incomplete", "targets_completed": 0, "requests": 0,
                    "cache_hits": 0, "failures": [{"error": f"{type(exc).__name__}: {exc}"}]}
                ranked = graph = misstep_record = None
            statuses[tid] = status
            if ranked is not None:
                rankings[tid], graphs[tid] = ranked, graph
            if misstep_record is not None:
                missteps[tid] = misstep_record
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
