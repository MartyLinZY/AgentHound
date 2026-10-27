#!/usr/bin/env python3
"""Offline Top-1 diagnosis; never update a checkpoint or historical prediction."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker
from analyze_attentionmap_semantic import measure, require
from run_attentionmap import read, save, sha, order

OUT = ROOT / "outputs/attentionmap-top1-diagnosis-20261027"
SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
SUITES = {
    "qwen38_thinking": ROOT / "outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1",
    "deepseek_official_thinking": ROOT / "outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1",
    "moma_deepseek41": ROOT.parent / "AgentHound/outputs/rootse-suite-deepseek-v41-flash-32k-v1",
}
METHODS = ("all_at_once", "step_by_step", "binary_search", "TRAIL", "FAMAS", "agent_locate")


def summary(rows: list[dict], key: str) -> dict:
    values = [r[key] for r in rows]
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def cluster_stats(difference: dict[str, float], cases: dict) -> dict:
    groups = sorted({c["group"] for c in cases.values()})
    totals = np.array([sum(difference[cid] for cid, c in cases.items() if c["group"] == g) for g in groups])
    sizes = np.array([sum(c["group"] == g for c in cases.values()) for g in groups])
    rng = np.random.default_rng(20261027)
    indices = rng.integers(len(groups), size=(5000, len(groups)))
    samples = totals[indices].sum(1) / sizes[indices].sum(1)
    # Exchange method labels jointly within a repository/task group.
    signs = rng.choice([-1, 1], size=(20000, len(groups)))
    null = signs @ totals / sizes.sum()
    observed = float(totals.sum() / sizes.sum())
    p = (int(np.sum(np.abs(null) >= abs(observed) - 1e-12)) + 1) / (len(null) + 1)
    return {"delta": observed, "group_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist(),
            "group_label_swap_p": p}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read(SOURCE / "manifest.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    by_id = {c["trajectory_id"]: c for c in cases.values()}
    labels = read(SOURCE / "labels.json")
    seeds = manifest["config"]["seeds"]
    provenance = {}

    def record(path: Path) -> None:
        provenance[str(path.resolve())] = sha(path)

    for name in ("manifest.json", "labels.json", "baselines.json"):
        record(SOURCE / name)
    data = {}
    for cid, c in cases.items():
        require(c["step_ids"] == [str(i) for i in range(c["n_steps"])], "Step IDs are not contiguous")
        record(Path(c["sources"]["raw"]["path"]))
        require(sha(Path(c["sources"]["raw"]["path"])) == c["sources"]["raw"]["sha256"], "Raw source drift")
        p = SOURCE / "features" / f"{cid}.npz"
        record(p)
        require(sha(p) == c["feature_sha256"], "Feature drift")
        with np.load(p) as z:
            data[cid] = tuple(torch.from_numpy(z[k]) for k in ("x", "failure", "prior"))
    baseline = {}
    baseline_info = {}
    for suite, path in SUITES.items():
        for name in ("manifest.json", "metrics.json", "predictions.csv", "snapshot/config.json", "snapshot/dataset_manifest.json"):
            record(path / name)
        m = read(path / "metrics.json")
        require(read(path / "manifest.json")["status"] == "complete", "Incomplete baseline suite")
        dataset = read(path / "snapshot/dataset_manifest.json")
        for c in cases.values():
            name = "RootSE/" + c["trajectory_id"].removeprefix("rootse/") + ".json"
            require(dataset[name] == c["sources"]["raw"]["sha256"], "Baseline data identity differs")
        with (path / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            require(len(selected) == len(cases), "Incomplete baseline predictions")
            key = f"{suite}/{method}"
            records = {}
            for row in selected:
                c = by_id[row["trajectory_id"]]
                cid = c["case_id"]
                require(labels[cid] == [int(row["gold_step"])], "Baseline gold mismatch")
                state_path = path / row["record_directory"] / "state.json"
                record(state_path)
                state = read(state_path)
                require(state["status"] == row["status"], "Baseline state status differs")
                prediction = int(row["prediction"]) if row["prediction"] else None
                if row["status"] == "valid":
                    require(str(prediction) == str(state["failure_id"]), "Baseline state prediction differs")
                correct = int(row["status"] == "valid" and prediction in labels[cid])
                require(correct == int(row["tol0"]), "Baseline metric differs")
                records[cid] = {"correct": correct, "prediction": prediction, "status": row["status"],
                                "state_path": str(state_path), "reason": state.get("failure_reason", "")}
            require(set(records) == set(cases), "Baseline IDs mismatch")
            require(sum(r["correct"] for r in records.values()) == m["metrics"][method]["tolerance_hits"][0], "Baseline summary mismatch")
            baseline[key] = records
            baseline_info[key] = {"model": m["model"], "n": len(records),
                                  "hits": sum(r["correct"] for r in records.values()),
                                  "valid": m["metrics"][method]["valid_predictions"],
                                  "top1": sum(r["correct"] for r in records.values()) / len(records),
                                  "statuses": dict(Counter(r["status"] for r in records.values())),
                                  "config": read(path / "snapshot/config.json")}
    save(OUT / "baseline_cases.json", baseline)
    save(OUT / "baseline_summary.json", baseline_info)

    prediction_rows = []
    fit_rows = []
    all_metrics = defaultdict(dict)
    intervention = {"key_max_attention_difference": 0.0, "key_max_score_difference": 0.0,
                    "key_top1_changed": 0, "key_zero_hits": 0, "query_zero_hits": 0, "n": 0}
    for variant in manifest["config"]["variants"]:
        for seed in seeds:
            metrics = {}
            for fold in range(manifest["config"]["folds"]):
                run = SOURCE / "runs" / variant / f"seed-{seed}" / f"fold-{fold}"
                for name in ("predictions.json", "model.pt", "status.json"):
                    record(run / name)
                status = read(run / "status.json")
                require(sha(run / "model.pt") == status["model_sha256"], "Model drift")
                require(sha(run / "predictions.json") == status["predictions_sha256"], "Prediction drift")
                stored = read(run / "predictions.json")
                checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
                model = AttentionMapRanker(checkpoint["input_dim"], hidden=checkpoint["hidden"], variant=variant)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                train_metrics = []
                with torch.no_grad():
                    for cid in status["train_ids"]:
                        r = model(*data[cid])
                        train_metrics.append(measure(r["scores"].tolist(), labels[cid]))
                    test_metrics = []
                    for cid, row in stored.items():
                        result = model(*data[cid])
                        require(np.array_equal(result["scores"].numpy(), row["scores"]), "Replay drift")
                        value = measure(row["scores"], labels[cid])
                        metrics[cid] = value
                        test_metrics.append(value)
                        if variant != "dense":
                            continue
                        local = measure(row["local"], labels[cid])
                        rel = measure(row["relation"], labels[cid])
                        root = min(labels[cid], key=lambda i: order(row["scores"]).index(i))
                        pred = order(row["scores"])[0]
                        probs = torch.softmax(result["scores"], dim=0)
                        x, f, e = data[cid]
                        entry = {"case_id": cid, "seed": seed, "fold": fold, **value,
                                 "root": root, "pred": pred, "n_steps": len(x),
                                 "local_rank": local["rank"], "relation_rank": rel["rank"],
                                 "local_pred": order(row["local"])[0],
                                 "offset": pred - root, "root_relative_position": root / max(len(x)-1, 1),
                                 "pred_relative_position": pred / max(len(x)-1, 1),
                                 "local_margin": row["local"][pred] - row["local"][root],
                                 "relation_margin": row["relation"][pred] - row["relation"][root],
                                 "top1_probability": float(probs[pred]), "root_probability": float(probs[root]),
                                 "decision_cosine_root_pred": float(x[root, :128] @ x[pred, :128]),
                                 "root_q": float(x[root, 515]), "pred_q": float(x[pred, 515]),
                                 "root_q_missing": bool(x[root, 516]), "pred_q_missing": bool(x[pred, 516]),
                                 "root_seed_rank": order(row["seeds"]).index(root) + 1,
                                 "root_mass_rank": order(row["mass"]).index(root) + 1,
                                 "root_prior_indegree": int(e[:, root].sum()),
                                 "attention_drop100_rank": measure(row["drop100_scores"], labels[cid])["rank"],
                                 "attention_shuffled_rank": measure(row["shuffled_scores"], labels[cid])["rank"]}
                        prediction_rows.append(entry)
                        key_weight = model.failure_key.weight.clone()
                        model.failure_key.weight.zero_()
                        key_zero = model(x, f, e)
                        model.failure_key.weight.copy_(key_weight)
                        query_weight = model.failure_query.weight.clone()
                        model.failure_query.weight.zero_()
                        query_zero = model(x, f, e)
                        model.failure_query.weight.copy_(query_weight)
                        intervention["key_max_attention_difference"] = max(intervention["key_max_attention_difference"], float((result["attention"]-key_zero["attention"]).abs().max()))
                        intervention["key_max_score_difference"] = max(intervention["key_max_score_difference"], float((result["scores"]-key_zero["scores"]).abs().max()))
                        intervention["key_top1_changed"] += int(order(key_zero["scores"].tolist())[0] != pred)
                        intervention["key_zero_hits"] += measure(key_zero["scores"].tolist(), labels[cid])["top1"]
                        intervention["query_zero_hits"] += measure(query_zero["scores"].tolist(), labels[cid])["top1"]
                        intervention["n"] += 1
                fit_rows.append({"variant": variant, "seed": seed, "fold": fold,
                                 "train_n": len(train_metrics), "test_n": len(test_metrics),
                                 "train_hits": sum(r["top1"] for r in train_metrics),
                                 "test_hits": sum(r["top1"] for r in test_metrics),
                                 "train_mrr_sum": sum(r["mrr"] for r in train_metrics),
                                 "test_mrr_sum": sum(r["mrr"] for r in test_metrics),
                                 "loss_first": status["losses"][0], "loss_last": status["losses"][-1],
                                 "graph_bias": float(model.graph_bias.detach()),
                                 "parameters": status["parameters_with_gradient"]})
            all_metrics[variant][seed] = metrics
        print(json.dumps({"diagnosed_variant": variant}), flush=True)
    save(OUT / "case_seed_diagnostics.json", prediction_rows)
    save(OUT / "fit_diagnostics.json", fit_rows)
    save(OUT / "failure_injection_check.json", intervention)
    histograms = {}
    for seed in seeds:
        rows = [r for r in prediction_rows if r["seed"] == seed]
        wrong = [r for r in rows if r["rank"] > 1]
        histograms[seed] = {
            "rank_bins": {"1": sum(r["rank"] == 1 for r in rows), "2": sum(r["rank"] == 2 for r in rows),
                          "3-5": sum(3 <= r["rank"] <= 5 for r in rows), "6-10": sum(6 <= r["rank"] <= 10 for r in rows),
                          ">10": sum(r["rank"] > 10 for r in rows)},
            "wrong_later": sum(r["offset"] > 0 for r in wrong), "wrong_earlier": sum(r["offset"] < 0 for r in wrong),
            "wrong_adjacent": sum(abs(r["offset"]) == 1 for r in wrong),
            "local_hits": sum(r["local_rank"] == 1 for r in rows),
            "relation_gained": sum(r["local_rank"] > 1 and r["rank"] == 1 for r in rows),
            "relation_lost": sum(r["local_rank"] == 1 and r["rank"] > 1 for r in rows),
            "wrong_local_already_wrong": sum(r["local_rank"] > 1 for r in wrong),
            "wrong_probability_ge90": sum(r["top1_probability"] >= .9 for r in wrong),
            "wrong_probability": summary(wrong, "top1_probability"),
        }
    training = {}
    for variant in manifest["config"]["variants"]:
        per_seed = []
        for seed in seeds:
            fits = [r for r in fit_rows if r["variant"] == variant and r["seed"] == seed]
            per_seed.append({"seed": seed,
                             "train_top1": sum(r["train_hits"] for r in fits) / sum(r["train_n"] for r in fits),
                             "test_top1": sum(r["test_hits"] for r in fits) / sum(r["test_n"] for r in fits),
                             "train_mrr": sum(r["train_mrr_sum"] for r in fits) / sum(r["train_n"] for r in fits),
                             "test_mrr": sum(r["test_mrr_sum"] for r in fits) / sum(r["test_n"] for r in fits),
                             "loss_first": float(np.mean([r["loss_first"] for r in fits])),
                             "loss_last": float(np.mean([r["loss_last"] for r in fits]))})
        training[variant] = per_seed
    per_case = {cid: [r for r in prediction_rows if r["case_id"] == cid] for cid in cases}
    groups = {}
    for name, predicate in {
        "all_seeds_wrong": lambda r: all(x["rank"] > 1 for x in r),
        "all_seeds_rank2to5": lambda r: all(2 <= x["rank"] <= 5 for x in r),
        "all_seeds_wrong_later": lambda r: all(x["offset"] > 0 for x in r),
        "any_relation_lost": lambda r: any(x["local_rank"] == 1 and x["rank"] > 1 for x in r),
        "all_relation_lost": lambda r: all(x["local_rank"] == 1 and x["rank"] > 1 for x in r),
        "all_seeds_beyond5": lambda r: all(x["rank"] > 5 for x in r),
    }.items():
        groups[name] = [cid for cid, rows in per_case.items() if predicate(rows)]
    comparisons = {}
    dense_mean = {cid: float(np.mean([all_metrics["dense"][s][cid]["top1"] for s in seeds])) for cid in cases}
    references = {"semantic/" + v: {cid: float(np.mean([all_metrics[v][s][cid]["top1"] for s in seeds])) for cid in cases}
                  for v in ("linear", "mlp", "fixed_graph", "dense_unconditioned")}
    # Primary external comparisons: complete latest Qwen and official DeepSeek suites.
    references.update({name: {cid: r["correct"] for cid, r in rows.items()} for name, rows in baseline.items()
                       if not name.startswith("moma_")})
    for name, scores in references.items():
        row = cluster_stats({cid: dense_mean[cid] - scores[cid] for cid in cases}, cases)
        row["reference_top1"] = float(np.mean(list(scores.values())))
        row["dense_top1"] = float(np.mean(list(dense_mean.values())))
        if name in baseline:
            row["per_seed_overlap"] = []
            for seed in seeds:
                dense = all_metrics["dense"][seed]
                row["per_seed_overlap"].append({"seed": seed,
                    "both_correct": sum(dense[c]["top1"] and scores[c] for c in cases),
                    "attention_only": sum(dense[c]["top1"] and not scores[c] for c in cases),
                    "baseline_only": sum(not dense[c]["top1"] and scores[c] for c in cases),
                    "both_wrong": sum(not dense[c]["top1"] and not scores[c] for c in cases),
                    "baseline_correct_root_in_attention_top5": sum(scores[c] and dense[c]["rank"] <= 5 for c in cases)})
        comparisons[name] = row
    # Holm correction over the complete 16-contrast exploratory family.
    ordered = sorted(comparisons, key=lambda k: comparisons[k]["group_label_swap_p"])
    previous = 0.0
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered)-index) * comparisons[name]["group_label_swap_p"]))
        comparisons[name]["holm_adjusted_p"] = previous
    save(OUT / "comparisons.json", comparisons)
    subgroup_baselines = {name: {method: {"n": len(ids), "hits": sum(rows[c]["correct"] for c in ids),
                                         "valid": sum(rows[c]["status"] == "valid" for c in ids)}
                                for method, rows in baseline.items()} for name, ids in groups.items()}
    result = {"n": len(cases), "seeds": seeds, "rank_and_branch": histograms,
              "training": training, "failure_injection": intervention,
              "case_groups": groups, "subgroup_baselines": subgroup_baselines,
              "hit_consistency": dict(Counter(sum(r["rank"] == 1 for r in rows) for rows in per_case.values())),
              "parameters_dense": 43653, "new_training_fits": 0, "new_api_calls": 0}
    save(OUT / "diagnosis.json", result)
    # Detailed cases are chosen by explicit groups, then shorter length and ID.
    selection = {}
    for name in ("all_relation_lost", "all_seeds_rank2to5", "all_seeds_wrong_later", "all_seeds_beyond5"):
        candidates = groups[name]
        if name != "all_relation_lost":
            candidates = [cid for cid in candidates if baseline["qwen38_thinking/all_at_once"][cid]["correct"]]
        selection[name] = sorted(candidates, key=lambda cid: (cases[cid]["n_steps"], cid))[:3]
    packets = {}
    for cid in sorted({cid for ids in selection.values() for cid in ids}):
        c = cases[cid]
        raw = read(Path(c["sources"]["raw"]["path"]))
        selected_steps = {i for r in per_case[cid] for i in (r["root"], r["pred"], r["local_pred"])}
        packets[cid] = {"trajectory_id": c["trajectory_id"], "case": c,
                        "mission": raw["original_mission_prompt"], "gold_reason": raw["failure_reason"],
                        "failed_tests": raw["failed_tests"], "rows": per_case[cid],
                        "steps": {str(i): raw["original_traj"][i] for i in sorted(selected_steps)},
                        "baseline": {k: v[cid] for k, v in baseline.items()
                                     if k in ["qwen38_thinking/all_at_once", "qwen38_thinking/agent_locate", "deepseek_official_thinking/all_at_once"]}}
    save(OUT / "selected_cases.json", {"selection": selection, "cases": packets})
    for name, expected in provenance.items():
        require(sha(Path(name)) == expected, "Historical asset modified during diagnosis")
    save(OUT / "source_hashes.json", provenance)
    save(OUT / "verification.json", {"files_unchanged": len(provenance), "baseline_predictions_verified": len(baseline)*len(cases),
                                      "oof_predictions_replayed": 6*3*len(cases), "new_api_calls": 0, "new_training_fits": 0,
                                      "script_sha256": sha(Path(__file__))})
    print(json.dumps({"rank_and_branch": histograms, "training": training, "injection": intervention,
                      "case_group_sizes": {k: len(v) for k, v in groups.items()}, "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
