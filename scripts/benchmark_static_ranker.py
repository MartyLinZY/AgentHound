#!/usr/bin/env python3
"""Benchmark the static LogHound ranker and save per-trajectory timings.

The benchmark starts after a prepared trajectory has been deserialized.  It
therefore measures the static analysis/ranking method rather than parquet
adaptation or downstream evaluation.  A separate batch measurement includes
prepared JSONL input and ranking JSONL output.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loghound_agent_debug.evidence import extract_evidence, load_clue_rules
from loghound_agent_debug.features.semantic import minmax, scores as semantic_scores
from loghound_agent_debug.graph import build_graph, pagerank
from loghound_agent_debug.rankers.loghound_static import (
    _evidence_for_step,
    _failure_text,
    load_config,
    rank_file,
    rank_trajectory,
)
from loghound_agent_debug.schema import TrajectoryInput, input_from_dict


def _quantile(values: list[float], quantile: float) -> float:
    """Return a deterministic nearest-rank quantile for a small sample."""
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _summary(values: list[float]) -> dict[str, float]:
    """Summarize a non-empty numeric series in milliseconds or counts."""
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(_quantile(values, 0.95), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _phase_timings(item: TrajectoryInput, config: dict, rules: dict) -> dict[str, float]:
    """Time ranker phases while rebuilding the same intermediate values.

    The benchmark input is the AgenTracer JSONL, which never passes
    ``rootse_dependencies``, so the linear fusion branch here mirrors
    ``rank_trajectory`` exactly.  RootSE RRF timing comes from
    ``official_rank_trajectory_ms``, which calls the real ranker.
    """
    start = time.perf_counter_ns()
    evidence = extract_evidence(item, rules)
    evidence_end = time.perf_counter_ns()

    semantic = semantic_scores(
        item,
        _failure_text(item),
        task_weight=float(config.get("semantic", {}).get("query_task_weight", 0.4)),
        include_observation=bool(config.get("semantic", {}).get("include_observation", False)),
    )
    graph = build_graph(item)
    pagerank_scores, _converged, iterations = pagerank(
        graph,
        alpha=float(config.get("graph", {}).get("alpha", 0.85)),
        tolerance=float(config.get("graph", {}).get("tolerance", 1e-6)),
        max_iterations=int(config.get("graph", {}).get("max_iterations", 1000)),
    )
    analysis_end = time.perf_counter_ns()

    semantic_norm = minmax(semantic)
    path_norm = minmax(pagerank_scores)
    fusion = config.get("fusion", {})
    semantic_weight = float(fusion.get("semantic", 0.12))
    clue_weight = float(fusion.get("clue", 1.0))
    path_weight = float(fusion.get("path", 1.0))
    denominator = float(fusion.get("denominator", semantic_weight + clue_weight + path_weight))
    ranking = []
    for ordinal, step in enumerate(item.steps):
        raw = (
            semantic_weight * semantic_norm[ordinal]
            + clue_weight * evidence.clue_scores[ordinal]
            + path_weight * path_norm[ordinal]
        )
        ranking.append(
            {
                "rank": 0,
                "source_step_id": step.source_step_id,
                "agent": step.agent,
                "score_raw": round(raw, 9),
                "suspiciousness": round(raw / denominator if denominator else 0.0, 9),
                "components": {
                    "semantic_normalized": round(semantic_norm[ordinal], 9),
                    "clue": round(evidence.clue_scores[ordinal], 9),
                    "pagerank": round(pagerank_scores[ordinal], 9),
                    "path_normalized": round(path_norm[ordinal], 9),
                },
                "evidence": _evidence_for_step(ordinal, evidence, graph.edges, path_norm),
            }
        )
    ordinal_by_source = {step.source_step_id: step.ordinal for step in item.steps}
    ranking.sort(key=lambda row: (-row["score_raw"], int(ordinal_by_source[row["source_step_id"]])))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    finish = time.perf_counter_ns()

    return {
        "evidence_ms": (evidence_end - start) / 1e6,
        "semantic_graph_pagerank_ms": (analysis_end - evidence_end) / 1e6,
        "fusion_sort_ms": (finish - analysis_end) / 1e6,
        "algorithm_ms": (finish - start) / 1e6,
        "steps": len(item.steps),
        "edges": len(graph.edges),
        "pagerank_iterations": iterations,
    }


def _official_timing(item: TrajectoryInput, config: dict, rules: dict) -> float:
    """Time the public API that returns the complete ranking result."""
    start = time.perf_counter_ns()
    rank_trajectory(item, config, rules)
    return (time.perf_counter_ns() - start) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="data/processed/agentracer/inputs.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rules", default="configs/clue_rules.yaml")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/efficiency")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    inputs_path = Path(args.inputs)
    items = [
        input_from_dict(json.loads(line))
        for line in inputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = load_config(args.config)
    rules = load_clue_rules(args.rules)

    # One unrecorded pass warms imports, regexes, and allocator state.
    for item in items:
        _phase_timings(item, config, rules)
        _official_timing(item, config, rules)

    phase_runs = []
    official_runs = []
    for _ in range(args.repeats):
        phase_runs.append([_phase_timings(item, config, rules) for item in items])
        official_runs.append([_official_timing(item, config, rules) for item in items])

    rows = []
    metric_names = (
        "evidence_ms",
        "semantic_graph_pagerank_ms",
        "fusion_sort_ms",
        "algorithm_ms",
        "steps",
        "edges",
        "pagerank_iterations",
    )
    for index, item in enumerate(items):
        row = {name: statistics.median([run[index][name] for run in phase_runs]) for name in metric_names}
        row["official_rank_trajectory_ms"] = statistics.median([run[index] for run in official_runs])
        row["trajectory_id"] = item.trajectory_id
        rows.append(row)

    # This separate measure answers how long the normal JSONL CLI path takes.
    with tempfile.TemporaryDirectory(prefix="loghound-static-benchmark-") as temporary_directory:
        output_path = str(Path(temporary_directory) / "rankings.jsonl")
        rank_file(inputs_path, args.config, args.rules, output_path)
        io_batches = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            rank_file(inputs_path, args.config, args.rules, output_path)
            io_batches.append((time.perf_counter_ns() - started) / 1e6)

    summary = {
        "method": {
            "clock": "time.perf_counter_ns wall-clock",
            "warmup_passes": 1,
            "measured_repeats": args.repeats,
            "per_trajectory_value": "median across measured repeats",
            "algorithm_scope": "prepared, deserialized input through complete rank_trajectory output; excludes parquet adaptation and evaluation",
            "jsonl_scope": "includes prepared JSONL input, config/rules loading, ranking JSONL serialization and output",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "dataset": {
            "input_path": str(inputs_path),
            "trajectories": len(items),
            "input_jsonl_mib": round(inputs_path.stat().st_size / 1024 / 1024, 2),
        },
        "per_trajectory_distribution_ms": {
            name: _summary([row[name] for row in rows])
            for name in (*metric_names[:4], "official_rank_trajectory_ms")
        },
        "structural_distribution": {
            name: _summary([row[name] for row in rows]) for name in metric_names[4:]
        },
        "batch_jsonl_end_to_end_ms": _summary(io_batches),
        "batch_jsonl_per_trajectory_ms": {
            "mean": round(statistics.fmean(io_batches) / len(items), 4),
            "median": round(statistics.median(io_batches) / len(items), 4),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "static_ranker_per_trajectory_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("trajectory_id", *metric_names, "official_rank_trajectory_ms"))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "static_ranker_timing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "per_trajectory_path": str(csv_path), **summary["dataset"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
