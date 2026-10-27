#!/usr/bin/env python3
"""Measure failure-context dilution and probe terminal-test-only encoding."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse_file
from loghound_agent_debug.features.frozen_semantic import FrozenTextEncoder
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker
from analyze_attentionmap_semantic import measure, require
from diagnose_attentionmap_top1 import cluster_stats
from run_attentionmap import read, save, sha, order

OUT = ROOT / "outputs/attentionmap-aa-input-audit-20261027"
SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
MODEL = ROOT / "outputs/models/all-MiniLM-L6-v2-1110a243"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read(SOURCE / "manifest.json")
    labels = read(SOURCE / "labels.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows, texts, data, hashes = [], [], {}, {}
    for cid, c in cases.items():
        raw = Path(c["sources"]["raw"]["path"])
        require(sha(raw) == c["sources"]["raw"]["sha256"], "Raw data drift")
        hashes[str(raw)] = sha(raw)
        item, _ = load_rootse_file(raw, protocol="postmortem")
        final = "\n".join(f"{t.code}\n{t.error_description}" for t in item.diagnostic_tests)
        observations = [o.content for o in item.failure_observations]
        counts = [len(x) for x in tokenizer([final]+observations, add_special_tokens=False,
                                           truncation=False, verbose=False)["input_ids"]]
        rows.append({"case_id": cid, "trajectory_id": c["trajectory_id"], "test_tokens": counts[0],
                     "trace_error_tokens": sum(counts[1:]), "test_share": counts[0]/max(1, sum(counts)),
                     "trace_error_steps": len(observations)})
        texts.append(final)
        featurepath = SOURCE / "features" / f"{cid}.npz"
        hashes[str(featurepath)] = sha(featurepath)
        require(sha(featurepath) == c["feature_sha256"], "Feature drift")
        with np.load(featurepath) as z:
            data[cid] = tuple(torch.from_numpy(z[k]) for k in ("x", "failure", "prior"))
    context_summary = {"n": len(rows), "median_test_share": float(np.median([r["test_share"] for r in rows])),
        "cases_test_share_lt_half": sum(r["test_share"] < .5 for r in rows),
        "cases_test_share_lt_tenth": sum(r["test_share"] < .1 for r in rows),
        "total_test_tokens": sum(r["test_tokens"] for r in rows),
        "total_trace_error_tokens": sum(r["trace_error_tokens"] for r in rows)}
    save(OUT / "failure-context-audit.json", {"summary": context_summary, "cases": rows})
    protocol = {"variant": "terminal_test_only", "source_fingerprint": manifest["fingerprint"],
        "definition": "Encode admitted final test code+error alone; replace failure[128:] and x[:,256:384]=decision*failure. Keep task, decision, observations, numeric features, q, prior and all weights frozen.",
        "evaluation": "OOF inference-time intervention, not retraining", "new_api_calls": 0, "new_training_fits": 0}
    if (OUT / "terminal-test-protocol.json").exists():
        require(read(OUT / "terminal-test-protocol.json") == protocol, "Protocol changed")
    else:
        save(OUT / "terminal-test-protocol.json", protocol)
    encoder = FrozenTextEncoder(MODEL)
    vectors, encoding = encoder.encode(texts, OUT / "terminal-test-encoding-cache", batch_size=32)
    vector_map = dict(zip(cases, vectors))
    predicted = []
    with torch.no_grad():
        for seed in manifest["config"]["seeds"]:
            for fold in range(5):
                run = SOURCE / "runs/dense" / f"seed-{seed}" / f"fold-{fold}"
                for filename in ("model.pt", "predictions.json", "status.json"):
                    hashes[str(run / filename)] = sha(run / filename)
                status = read(run / "status.json")
                require(sha(run / "model.pt") == status["model_sha256"], "Model drift")
                require(sha(run / "predictions.json") == status["predictions_sha256"], "Prediction drift")
                cp = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
                model = AttentionMapRanker(cp["input_dim"], hidden=cp["hidden"], variant="dense")
                model.load_state_dict(cp["state_dict"]);model.eval()
                for cid, saved in read(run / "predictions.json").items():
                    require(cases[cid]["fold"] == fold and cid not in status["train_ids"], "OOF mismatch")
                    x, failure, prior = data[cid]
                    original = model(x, failure, prior)
                    require(np.array_equal(original["scores"].numpy(), saved["scores"]), "Original prediction drift")
                    updated_x, updated_failure = x.clone(), failure.clone()
                    updated_failure[128:] = torch.from_numpy(vector_map[cid])
                    updated_x[:, 256:384] = x[:, :128] * updated_failure[128:]
                    result = model(updated_x, updated_failure, prior)
                    predicted.append({"case_id": cid, "seed": seed, "fold": fold,
                        "original": measure(saved["scores"], labels[cid]),
                        "terminal_test_only": measure(result["scores"].tolist(), labels[cid]),
                        "original_pred": order(saved["scores"])[0], "pred": order(result["scores"].tolist())[0],
                        "scores": result["scores"].tolist(),
                        "failure_embedding_cosine": float(failure[128:] @ updated_failure[128:])})
    summary = {}
    for mode in ("original", "terminal_test_only"):
        per_seed = [{"seed": s, "hits": sum(r[mode]["top1"] for r in predicted if r["seed"] == s),
                     **{k: float(np.mean([r[mode][k] for r in predicted if r["seed"] == s])) for k in ("top1", "top5", "mrr")}}
                    for s in manifest["config"]["seeds"]]
        summary[mode] = {"per_seed": per_seed, **{k: float(np.mean([r[mode][k] for r in predicted])) for k in ("top1", "top5", "mrr")}}
    delta = {cid: float(np.mean([r["terminal_test_only"]["top1"]-r["original"]["top1"] for r in predicted if r["case_id"] == cid])) for cid in cases}
    summary["pairing"] = cluster_stats(delta, cases)
    summary["gained_case_seed"] = sum(not r["original"]["top1"] and r["terminal_test_only"]["top1"] for r in predicted)
    summary["lost_case_seed"] = sum(r["original"]["top1"] and not r["terminal_test_only"]["top1"] for r in predicted)
    for path, expected in hashes.items():
        require(sha(Path(path)) == expected, "Frozen asset modified")
    save(OUT / "terminal-test-predictions.json", predicted)
    save(OUT / "terminal-test-summary.json", summary)
    save(OUT / "terminal-test-verification.json", {"files_unchanged": len(hashes), "original_replays": len(predicted),
        "encoding": encoding, "new_api_calls": 0, "new_training_fits": 0, "script_sha256": sha(Path(__file__))})
    save(OUT / "terminal-test-source-hashes.json", hashes)
    print(json.dumps({"context": context_summary, "intervention": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
