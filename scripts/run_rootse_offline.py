#!/usr/bin/env python3
"""Run fixed RootSE ablations using rules or validated caches, without API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rankers.anchor_path import VARIANTS, rank_trajectory
from loghound_agent_debug.rankers.loghound_static import load_config
from loghound_agent_debug.rootse_evaluation import evaluate
from loghound_agent_debug.rootse_operations import track_state_dependencies
from loghound_agent_debug.schema import input_to_dict


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="HF clone with data/, or official RootSE directory")
    parser.add_argument("--out-dir", required=True, help="new output directory; existing runs are not overwritten")
    parser.add_argument("--protocols", nargs="+", choices=("trace-only", "postmortem"), default=["trace-only", "postmortem"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--rules", default=str(ROOT / "configs/clue_rules.yaml"))
    parser.add_argument("--graph-source", choices=("rules", "cache"), default="rules")
    parser.add_argument("--cache-dir", default=str(ROOT / "outputs/rootse/llm_cache"))
    args = parser.parse_args(argv)
    if len(set(args.protocols)) != len(args.protocols) or len(set(args.variants)) != len(args.variants):
        parser.error("protocols and variants must be unique")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error("output directory is nonempty; use a new directory to preserve previous results")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "python": platform.python_version(), "api_requests_this_run": 0,
                "graph_source": args.graph_source, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_files": {str(p.relative_to(ROOT)): digest(p) for folder in ("src", "scripts", "configs") for p in sorted((ROOT / folder).rglob("*")) if p.suffix in {".py", ".yaml"}},
                "config_sha256": digest(Path(args.config)), "rules_sha256": digest(Path(args.rules)),
                "packages": {name: importlib.metadata.version(name) for name in ("PyYAML", "numpy", "requests")},
                "historical_llm_cost": "not_applicable" if args.graph_source == "rules" else "not_reconstructed_from_cache; excluded_from_this_run_time"}
    write_json(out / "manifest.json", manifest)
    try:
        config, rules = load_config(args.config), load_clue_rules(args.rules)
        trace_inputs, _ = load_rootse(args.root)
        files = sorted(Path(args.root).rglob("*.json"))
        manifest["raw_files"] = {str(p.relative_to(Path(args.root))): digest(p) for p in files}
        manifest["trajectories"] = len(trace_inputs)
        manifest["steps"] = sum(len(item.steps) for item in trace_inputs)
        dependencies, graph_rows, extraction = {}, [], []
        extractor = None
        if args.graph_source == "cache":
            from loghound_agent_debug.rootse_dependencies import DependencyExtractor
            extractor = DependencyExtractor(ROOT, args.cache_dir, cache_only=True,
                request_params=config.get("rootse_llm", {}).get("request_params", {}),
                output_token_overrides=config.get("rootse_llm", {}).get("output_token_overrides", {}))
        graph_start = time.perf_counter()
        for item in trace_inputs:
            edges = [asdict(edge) for edge in track_state_dependencies(item.steps)]
            if extractor is not None:
                llm_edges, status = extractor.extract_for_trajectory(item)
                extraction.append(status)
                if status["status"] != "complete":
                    write_json(out / "cache_status.json", extraction)
                    raise ValueError(f"required validated cache incomplete: {item.trajectory_id}")
                edges += llm_edges
            dependencies[item.trajectory_id] = edges
            graph_rows.append({"trajectory_id": item.trajectory_id, "input_hash": hashlib.sha256(json.dumps(input_to_dict(item), sort_keys=True).encode()).hexdigest(), "dependencies": edges})
        manifest["graph_seconds"] = time.perf_counter() - graph_start
        write_jsonl(out / "dependencies.jsonl", graph_rows)
        if extraction:
            write_json(out / "cache_status.json", extraction)
        summaries = []
        for protocol in args.protocols:
            inputs, labels = load_rootse(args.root, protocol=protocol)
            for variant in args.variants:
                run_start = time.perf_counter()
                results = [rank_trajectory(item, config, rules, dependencies[item.trajectory_id], variant=variant) for item in inputs]
                ranking_seconds = time.perf_counter() - run_start
                metrics = evaluate(inputs, labels, results)
                folder = out / protocol / variant
                write_jsonl(folder / "rankings.jsonl", results)
                write_json(folder / "metrics.json", metrics)
                summary = {"protocol": protocol, "variant": variant,
                           **{key: value for key, value in metrics.items() if key not in {"per_trajectory", "by_group"}},
                           "ranking_seconds": ranking_seconds,
                           "graph_plus_ranking_seconds": manifest["graph_seconds"] + ranking_seconds,
                           "no_linked_anchor": sum(r.get("evidence_status") == "no_linked_anchor" for r in results) if variant != "baseline" else None,
                           "trajectories_with_upstream_propagation": sum((r.get("anchor_stats", {}).get("propagation_reaches_unseeded_steps") or 0) > 0 for r in results) if variant.endswith("backtrace") else None,
                           "trajectories_with_supported_dependency_edges": sum((r.get("anchor_stats", {}).get("supported_dependency_edges") or 0) > 0 for r in results) if variant.endswith("backtrace") else None,
                           "graph_edges": sum(r["graph_stats"]["edges"] for r in results),
                           "graph_modes": dict(Counter(r["graph_stats"].get("ranking_mode", "global") for r in results))}
                summaries.append(summary)
                print(f"{protocol}/{variant}: Top-1 {metrics['top1_hits']}/{len(inputs)}, Top-3 {metrics['top3_hits']}/{len(inputs)}, MRR {metrics['mrr']:.4f}", flush=True)
        write_json(out / "summary.json", summaries)
        # Paired changes identify exactly which trajectories fusion gains/loses.
        paired = {}
        for protocol in args.protocols:
            if "baseline" not in args.variants:
                continue
            base = json.loads((out / protocol / "baseline/metrics.json").read_text())["per_trajectory"]
            for variant in args.variants:
                rows = json.loads((out / protocol / variant / "metrics.json").read_text())["per_trajectory"]
                paired[f"{protocol}/{variant}"] = {metric: {
                    "gained": [r["trajectory_id"] for b, r in zip(base, rows) if r[metric] and not b[metric]],
                    "lost": [r["trajectory_id"] for b, r in zip(base, rows) if b[metric] and not r[metric]]}
                    for metric in ("top1_hit", "top3_hit")}
        write_json(out / "paired_changes.json", paired)
        manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        write_json(out / "manifest.json", manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
