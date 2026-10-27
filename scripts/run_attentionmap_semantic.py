#!/usr/bin/env python3
"""Matched lexical/semantic experiments; the v1 trainer and model stay unchanged."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.features.frozen_semantic import (
    FrozenTextEncoder, MODEL_FILES, MODEL_ID, REVISION, item_texts, replace_text_features,
)
from run_attentionmap import digest, read, report, save, sha, train


def check_source(source: Path) -> dict:
    manifest = read(source / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Completed v1 experiment required")
    for case in manifest["cases"]:
        for entry in case["sources"].values():
            if sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Historical input changed")
        if sha(source / "features" / f"{case['case_id']}.npz") != case["feature_sha256"]:
            raise ValueError("Historical features changed")
    for name in ("labels", "baselines"):
        if sha(source / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Historical {name} changed")
    return manifest


def prepare(out: Path, source: Path, model_dir: Path, device: str) -> dict:
    original = check_source(source)
    files = list(original["config"]["source_hashes"]) + [
        "scripts/run_attentionmap_semantic.py",
        "src/loghound_agent_debug/features/frozen_semantic.py",
        "experiments/attentionmap/SEMANTIC_PROTOCOL.md",
        "experiments/attentionmap/semantic-requirements.lock.txt",
    ]
    hashes = {name: sha(ROOT / name) for name in files}
    if any(hashes[name] != value for name, value in original["config"]["source_hashes"].items()):
        raise ValueError("Original pipeline code changed")
    identity = {"source": str(source.resolve()), "source_fingerprint": original["fingerprint"],
                "source_manifest_sha256": sha(source / "manifest.json"),
                "source_hashes": hashes, "model_id": MODEL_ID,
                "revision": REVISION, "encoding_device": device, "batch_size": 32}
    fingerprint = digest(identity)
    if (out / "experiment.json").exists():
        experiment = read(out / "experiment.json")
        if experiment["fingerprint"] != fingerprint:
            raise ValueError("Frozen semantic experiment changed; use a new directory")
        if experiment["status"] == "prepared":
            for name in ("lexical", "semantic"):
                m = read(out / name / "manifest.json")
                for c in m["cases"]:
                    if sha(out / name / "features" / f"{c['case_id']}.npz") != c["feature_sha256"]:
                        raise ValueError("Prepared feature drift")
            return experiment
    else:
        experiment = {**identity, "fingerprint": fingerprint, "status": "encoding",
                      "created_at": time.time()}
        save(out / "experiment.json", experiment)
        for name in files:
            dest = out / "snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        # Preserve a complete inventory before any new encoding/training.
        save(out / "historical_files.json", {str(p.relative_to(source)): sha(p)
             for p in source.rglob("*") if p.is_file()})
    if any(not (model_dir / name).is_file() for name in MODEL_FILES):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=list(MODEL_FILES),
                          local_dir=model_dir, token=False)
    torch.set_num_threads(1)
    encoder = FrozenTextEncoder(model_dir, device=device)
    inputs, _ = load_rootse(ROOT.parent / "TrajAudit/RootSE", protocol="postmortem")
    by_id = {item.trajectory_id: item for item in inputs}
    ordered = []
    text_lists = []
    for case in original["cases"]:
        item = by_id[case["trajectory_id"]]
        if [s.source_step_id for s in item.steps] != case["step_ids"]:
            raise ValueError("Step identity changed")
        ordered.append(item)
        text_lists.append(item_texts(item))
    vectors, stats = encoder.encode(
        [text for texts in text_lists for text in texts], out / "embeddings",
        batch_size=32, progress=lambda status: print(json.dumps(status), flush=True))
    save(out / "encoding.json", stats)
    for representation in ("lexical", "semantic"):
        dest = out / representation
        dest.mkdir(exist_ok=True)
        config = copy.deepcopy(original["config"])
        config.update({"representation": representation, "source_hashes": hashes,
                       "semantic_experiment": fingerprint})
        manifest = {**copy.deepcopy(original), "config": config, "status": "prepared",
                    "created_at": time.time()}
        offset = 0
        for case, texts in zip(manifest["cases"], text_lists):
            feature_path = dest / "features" / f"{case['case_id']}.npz"
            feature_path.parent.mkdir(exist_ok=True)
            origin = source / "features" / feature_path.name
            if representation == "lexical":
                shutil.copy2(origin, feature_path)
            else:
                with np.load(origin, allow_pickle=False) as z:
                    x, failure = replace_text_features(z["x"], vectors[offset:offset + len(texts)])
                    np.savez_compressed(feature_path, x=x, failure=failure, prior=z["prior"])
            offset += len(texts)
            case["feature_sha256"] = sha(feature_path)
        for name in ("labels.json", "baselines.json"):
            shutil.copy2(source / name, dest / name)
        manifest["fingerprint"] = digest({"config": config, "cases": manifest["cases"],
                                           "labels": manifest["labels_sha256"]})
        save(dest / "manifest.json", manifest)
    experiment["status"] = "prepared"
    experiment["encoding_identity"] = stats["identity"]
    experiment["encoding_sha256"] = sha(out / "encoding.json")
    save(out / "experiment.json", experiment)
    return experiment


def validate_completed_artifacts(out: Path, manifest: dict) -> None:
    """The historical trainer checks predictions; also protect models and maps."""
    for path in (out / "runs").rglob("status.json"):
        status = read(path)
        if status["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Run fingerprint mismatch")
        for name in ("model.pt", "predictions.json"):
            key = "model_sha256" if name == "model.pt" else "predictions_sha256"
            if sha(path.parent / name) != status[key]:
                raise ValueError(f"Completed artifact changed: {name}")
    seal = out / "artifacts.json"
    if seal.exists():
        for name, expected in read(seal).items():
            if sha(out / name) != expected:
                raise ValueError(f"Completed artifact changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/attentionmap-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/attentionmap-semantic-v2")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs/models/all-MiniLM-L6-v2-1110a243")
    parser.add_argument("--encoding-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--stage", choices=("prepare", "train", "report", "all"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.out, args.source, args.model_dir, args.encoding_device)
        for name in ("lexical", "semantic"):
            dest = args.out / name
            manifest = read(dest / "manifest.json")
            validate_completed_artifacts(dest, manifest)
            if args.stage in ("train", "all"):
                print(json.dumps({"representation": name}), flush=True)
                train(manifest, dest)
                save(dest / "artifacts.json", {
                    str(p.relative_to(dest)): sha(p) for directory in ("runs", "maps")
                    for p in (dest / directory).rglob("*") if p.is_file()})
            if args.stage in ("report", "all"):
                report(manifest, dest)
                (dest / "REPORT.md").write_text(
                    f"# {name} 分组实验\n\n统一比较见上级 analysis/report.md；逐方法数据见 summary.json。\n")


if __name__ == "__main__":
    main()
