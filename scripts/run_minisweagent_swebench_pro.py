#!/usr/bin/env python3
"""Run mini-SWE-agent on the ScaleAI SWE-bench Pro parquet dataset."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj


DEFAULT_DATA = Path("data/raw/swebench_pro/SWE-bench_Pro-test.parquet")
DEFAULT_CONFIG = Path(
    "project/SWE-bench_Pro-os/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml"
)
DEFAULT_LANGUAGE_QUOTA = "go=12,python=11,js=6,ts=1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--instance-id", default="")
    parser.add_argument(
        "--retry-status",
        default="",
        help="Only rerun instances with this status from the existing results.json",
    )
    parser.add_argument(
        "--language-quota",
        default=DEFAULT_LANGUAGE_QUOTA,
        help="Comma-separated repo_language quotas, e.g. go=12,python=11,js=6,ts=1",
    )
    parser.add_argument("--redo-existing", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    rows = [dict(row) for row in dataset]
    return sorted(rows, key=lambda row: str(row["instance_id"]))


def parse_language_quota(value: str) -> dict[str, int]:
    aliases = {
        "go": "go",
        "golang": "go",
        "python": "python",
        "py": "python",
        "javascript": "js",
        "js": "js",
        "typescript": "ts",
        "ts": "ts",
    }
    quota: dict[str, int] = {}
    for item in value.split(","):
        language, separator, count = item.partition("=")
        language = aliases.get(language.strip().lower())
        if not separator or language is None or not count.strip().isdigit():
            raise ValueError(f"Invalid --language-quota item: {item!r}")
        quota[language] = int(count)
    if not quota or any(count < 0 for count in quota.values()):
        raise ValueError(f"Invalid --language-quota: {value!r}")
    return quota


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if args.instance_id:
        selected = [row for row in rows if str(row["instance_id"]) == args.instance_id]
        if not selected:
            raise SystemExit(f"instance_id not found: {args.instance_id}")
        return selected, {str(selected[0].get("repo_language") or "unknown"): 1}

    quota = parse_language_quota(args.language_quota)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("repo_language") or "unknown").lower(), []).append(row)

    selected: list[dict[str, Any]] = []
    language_counts: dict[str, int] = {}
    for language, count in quota.items():
        candidates = grouped.get(language, [])
        if len(candidates) < count:
            raise SystemExit(
                f"Not enough {language} rows: requested {count}, available {len(candidates)}"
            )
        selected.extend(candidates[:count])
        language_counts[language] = count

    selected.sort(key=lambda row: str(row["instance_id"]))
    return selected, language_counts


def make_task(row: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            str(row.get("problem_statement") or ""),
            "Requirements:\n" + str(row.get("requirements") or ""),
            "New interfaces introduced:\n" + str(row.get("interface") or ""),
        ]
    ).strip()


def make_image(row: dict[str, Any]) -> str:
    tag = str(row.get("dockerhub_tag") or "").strip()
    if not tag:
        raise ValueError(f"Missing dockerhub_tag for {row.get('instance_id')}")
    if tag.startswith("jefzda/sweap-images:"):
        return tag
    return f"jefzda/sweap-images:{tag}"


def redact_trajectory(path: Path) -> None:
    """Remove credentials that mini-swe-agent includes in its saved config."""
    data = json.loads(path.read_text())
    secret_values = {
        value
        for key in ("MSWEA_MODEL_API_KEY", "OPENAI_API_KEY")
        if (value := os.getenv(key))
    }
    sensitive_keys = {"api_key", "authorization", "password", "secret", "token"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if key.lower() in sensitive_keys else scrub(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            for secret in secret_values:
                value = value.replace(secret, "[REDACTED]")
        return value

    path.write_text(json.dumps(scrub(data), indent=2))


def build_config(config_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if model_name := os.getenv("MSWEA_MODEL_NAME"):
        # LiteLLM needs an explicit provider for OpenAI-compatible endpoints.
        config.setdefault("model", {})["model_name"] = (
            model_name if "/" in model_name else f"openai/{model_name}"
        )
    config.setdefault("agent", {})["step_limit"] = 130
    config["agent"]["instance_template"] = config["agent"]["instance_template"].replace(
        "/testbed", "/app"
    )
    config.setdefault("environment", {})
    run_args = list(config["environment"].get("run_args", ["--rm"]))
    if "--platform" not in run_args:
        run_args.extend(["--platform", "linux/amd64"])
    # SWE-bench Pro images set /bin/bash as an ENTRYPOINT. mini-swe-agent
    # supplies `sleep 2h` as the container command, so clear that entrypoint
    # or Docker interprets `sleep` as a bash script path and exits immediately.
    if "--entrypoint" not in run_args and "--entrypoint=" not in run_args:
        run_args.extend(["--entrypoint", ""])
    config["environment"].update(
        {
            "environment_class": "docker",
            "image": make_image(row),
            "cwd": "/app",
            "run_args": run_args,
            "pull_timeout": int(os.getenv("MSWEA_DOCKER_PULL_TIMEOUT", "600")),
        }
    )
    return config


def run_one(row: dict[str, Any], config_path: Path, output_dir: Path, redo_existing: bool) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    instance_dir = output_dir / instance_id
    trajectory_path = instance_dir / f"{instance_id}.traj.json"
    if trajectory_path.exists() and not redo_existing:
        return {"instance_id": instance_id, "status": "skipped", "trajectory": str(trajectory_path)}

    config = build_config(config_path, row)
    model = get_model(config=config.get("model", {}))
    agent = None
    env = None
    exit_status: str | None = None
    result: str | None = None
    extra_info: dict[str, Any] | None = None

    try:
        env = get_environment(config["environment"])
        agent = DefaultAgent(model, env, **config.get("agent", {}))
        exit_status, result = agent.run(make_task(row))
    except Exception as exc:  # keep one failed task from losing the pilot manifest
        exit_status = type(exc).__name__
        result = str(exc)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            trajectory_path,
            print_path=False,
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            repo=row.get("repo"),
            dockerhub_tag=row.get("dockerhub_tag"),
            base_commit=row.get("base_commit"),
        )
        redact_trajectory(trajectory_path)
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()

    return {
        "instance_id": instance_id,
        "status": exit_status,
        "trajectory": str(trajectory_path),
        "submission": result,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data)
    selected, language_counts = select_rows(rows, args)
    existing_results: list[dict[str, Any]] = []
    if args.retry_status:
        results_path = args.output_dir / "results.json"
        if not results_path.exists():
            raise SystemExit(f"Cannot retry without existing results: {results_path}")
        existing_results = json.loads(results_path.read_text())
        retry_ids = {
            str(item["instance_id"])
            for item in existing_results
            if str(item.get("status")) == args.retry_status
        }
        selected = [row for row in selected if str(row["instance_id"]) in retry_ids]
        language_counts = {}
        for row in selected:
            language = str(row.get("repo_language") or "unknown").lower()
            language_counts[language] = language_counts.get(language, 0) + 1
    if not selected:
        raise SystemExit(f"No instances selected for retry status {args.retry_status!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / (
        f"retry-selection-{args.retry_status}.json" if args.retry_status else "selection.json"
    )
    selection_path.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "config": str(args.config),
                "start": args.start,
                "limit": args.limit,
                "count": len(selected),
                "language_quota": parse_language_quota(args.language_quota),
                "language_counts": language_counts,
                "selection_strategy": "per-language quota, then instance_id ascending",
                "retry_status": args.retry_status,
                "instance_ids": [row["instance_id"] for row in selected],
            },
            indent=2,
        )
    )

    result_by_id = {str(item["instance_id"]): item for item in existing_results}
    output_order = [str(item["instance_id"]) for item in existing_results]
    if not output_order:
        output_order = [str(row["instance_id"]) for row in selected]
    for index, row in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {row['instance_id']}", flush=True)
        result = run_one(row, args.config, args.output_dir, args.redo_existing or bool(args.retry_status))
        result_by_id[str(row["instance_id"])] = result
        if str(row["instance_id"]) not in output_order:
            output_order.append(str(row["instance_id"]))
        results = [result_by_id[instance_id] for instance_id in output_order if instance_id in result_by_id]
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[{index}/{len(selected)}] {result['status']}", flush=True)


if __name__ == "__main__":
    main()
