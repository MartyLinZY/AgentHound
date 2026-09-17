#!/usr/bin/env python3
"""Convert mini-SWE-agent trajectories into official SWE-bench Pro inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--prefix", default="qwen3.7-flash-mini-swe-agent")
    return parser.parse_args()


def load_trajectories(trajectory_dir: Path, instance_ids: list[str]) -> list[dict[str, Any]]:
    patches = []
    missing = []
    invalid = []
    for instance_id in instance_ids:
        path = trajectory_dir / instance_id / f"{instance_id}.traj.json"
        if not path.exists():
            missing.append(instance_id)
            continue
        trajectory = json.loads(path.read_text())
        submission = trajectory.get("info", {}).get("submission")
        if trajectory.get("info", {}).get("exit_status") != "Submitted" or not isinstance(submission, str):
            invalid.append({"instance_id": instance_id, "status": trajectory.get("info", {}).get("exit_status")})
            continue
        patches.append({"instance_id": instance_id, "patch": submission, "prefix": ""})

    if missing or invalid:
        raise SystemExit(json.dumps({"missing": missing, "invalid": invalid}, indent=2))
    return patches


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    instance_ids = [str(instance_id) for instance_id in selection["instance_ids"]]
    patches = load_trajectories(args.trajectory_dir, instance_ids)
    for patch in patches:
        patch["prefix"] = args.prefix

    dataset = load_dataset("parquet", data_files=str(args.data), split="train")
    rows = {str(row["instance_id"]): dict(row) for row in dataset}
    missing_rows = [instance_id for instance_id in instance_ids if instance_id not in rows]
    if missing_rows:
        raise SystemExit(f"Missing rows in parquet: {missing_rows}")

    args.patch_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.patch_output.write_text(json.dumps(patches, indent=2, ensure_ascii=False))
    with args.sample_output.open("w") as output:
        for instance_id in instance_ids:
            output.write(json.dumps(rows[instance_id], ensure_ascii=False) + "\n")

    print(f"Prepared {len(patches)} patches")
    print(f"Patch JSON: {args.patch_output}")
    print(f"Sample JSONL: {args.sample_output}")


if __name__ == "__main__":
    main()
