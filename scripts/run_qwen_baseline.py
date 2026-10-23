#!/usr/bin/env python3
"""Qwen-only RootSE All-at-Once pilot using the official TrajAudit prompt."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.adapters.rootse import load_rootse_file


def literal_assignment(path: Path, name: str) -> str:
    """Read prompt constants without importing/executing upstream config/code."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"missing literal prompt constant: {name}")


def messages_for(raw: dict, source: Path) -> list[dict[str, str]]:
    baseline = source / "baseline/who_and_when/all_at_once.py"
    user = literal_assignment(baseline, "user_prompt")
    system = literal_assignment(baseline, "system_prompt")
    guidelines = literal_assignment(source / "utils.py", "annotation_guidelines")
    # Match official prompt and first-test policy. Only explicit step fields
    # are forwarded, never top-level annotations or the reference patch.
    fields = ("index", "thought", "response", "action", "observation", "execution_time", "state", "extra_info")
    steps = [json.dumps({k: value for k, value in step.items() if k in fields}, indent=2) for step in raw["original_traj"]]
    test = next(iter(raw["failed_tests"].values()))
    user = user.format(problem_statement=raw["original_mission_prompt"],
        trajectories="<trajectory>\n" + "\n\n".join(steps) + "\n</trajectory>",
        test_code=test["code"], error_message=test["error_description"],
        annotation_guidelines=guidelines, ground_truth="\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def select_files(root: Path, limit: int) -> list[Path]:
    files = sorted((root / "data" if (root / "data").is_dir() else root).rglob("*.json"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        # Public RootSE layout is agent/model/file; selection uses no labels.
        groups["/".join(path.parts[-3:-1])].append(path)
    selected = []
    while groups:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if limit and len(selected) == limit:
                return selected
    return selected


def validate_model(model: str) -> None:
    if "qwen" not in model.lower() or "claude" in model.lower() or "YOUR_" in model:
        raise ValueError("this runner requires an explicit Qwen model; other models are disabled")


def endpoint_for(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query or fragment")
    if not parsed.hostname or not (parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("use an HTTPS API endpoint, or HTTP on localhost")
    return base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/chat/completions") else "/chat/completions")


def parse_prediction(response: dict, step_ids: set[str], recover_reason_quotes: bool = False) -> dict:
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("model response is truncated or unfinished")
    content = choice["message"].get("content") or ""
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("```"):
        content = content[8:-3].strip()
    recovered = False
    try:
        prediction = json.loads(content)
    except json.JSONDecodeError:
        if not recover_reason_quotes:
            raise
        # Only recover an unambiguous two-field envelope. The step ID remains
        # valid JSON; malformed quotation marks in the explanation are kept
        # verbatim. Never infer a step ID from prose or consult the gold label.
        match = re.fullmatch(r'\{\s*"failure_id"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)\s*,\s*"failure_reason"\s*:\s*"(.*)"\s*\}', content, re.DOTALL)
        if not match or re.search(r'"\s*:', match.group(2)):
            raise ValueError("ambiguous malformed prediction")
        prediction = {"failure_id": json.loads(match.group(1)), "failure_reason": match.group(2)}
        recovered = True
    step_id = str(prediction["failure_id"])
    if step_id not in step_ids or not isinstance(prediction.get("failure_reason"), str):
        raise ValueError("prediction must contain an original step ID and a reason")
    return {"failure_id": step_id, "failure_reason": prediction["failure_reason"], "format_recovered": recovered}


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--trajaudit-source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL_NAME", "Qwen3.5-27B"))
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", ""))
    key_source = parser.add_mutually_exclusive_group()
    key_source.add_argument("--api-key-env", default="QWEN_API_KEY")
    key_source.add_argument("--api-key-file", type=Path, help="read a plain-text key locally; never log its value")
    parser.add_argument("--limit", type=int, default=8, help="0 for all; otherwise round-robin by agent/model")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--response-format", choices=("text", "json_object"), default="text")
    parser.add_argument("--recover-reason-quotes", action="store_true")
    parser.add_argument("--selection", choices=("round-robin", "shortest-input"), default="round-robin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    validate_model(args.model)
    if args.limit < 0 or args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("limit must be nonnegative; tokens/timeout must be positive")
    api_key = args.api_key_file.read_text().strip() if args.api_key_file else os.getenv(args.api_key_env, "")
    if not args.dry_run and (not args.base_url or not api_key):
        parser.error(f"missing Qwen configuration: set QWEN_BASE_URL and {args.api_key_env}")
    endpoint = endpoint_for(args.base_url) if args.base_url else None
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        parser.error("use a new output directory; previous runs are preserved")
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.trajaudit_source)
    files = select_files(Path(args.root), 0 if args.selection == "shortest-input" else args.limit)
    if args.selection == "shortest-input":
        # Selection uses only fields sent to the model, never annotation size
        # or correctness. This is a feasibility subset, not a full benchmark.
        files.sort(key=lambda p: (len(json.dumps(messages_for(json.loads(p.read_text()), source))), str(p)))
        if args.limit:
            files = files[:args.limit]
    if not files:
        parser.error("no RootSE JSON files found")
    manifest = {"status": "preparing", "baseline": "official_all_at_once_prompt", "model": args.model,
                "protocol": "postmortem_first_test_without_gold_patch", "selected": len(files),
                "scope": "full" if args.limit == 0 else "pilot", "api_requests": 0, "dry_run": args.dry_run,
                "endpoint": endpoint, "max_tokens": args.max_tokens, "temperature": 0,
                "response_format": args.response_format,
                "selection": args.selection, "recover_reason_quotes": args.recover_reason_quotes,
                "prompt_source_hashes": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source / "utils.py", source / "baseline/who_and_when/all_at_once.py")}}
    save(out / "manifest.json", manifest)
    records = []
    prepared = []
    # Validate all payloads before sending the first request. No truncation:
    # context-limit rejections are explicit failures, not silent evidence loss.
    for path in files:
        raw = json.loads(path.read_text())
        item, label = load_rootse_file(path, protocol="postmortem")
        payload = {"model": args.model, "messages": messages_for(raw, source), "temperature": 0, "max_tokens": args.max_tokens}
        if args.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        prepared.append((item, label, payload))
        records.append({"trajectory_id": item.trajectory_id, "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_chars": len(encoded), "status": "prepared"})
    save(out / "records.json", records)
    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        save(out / "manifest.json", manifest)
        print(f"Qwen dry-run: {len(files)} prepared; API requests=0; model={args.model}")
        return 0
    manifest["status"] = "running"
    save(out / "manifest.json", manifest)
    for i, (item, label, payload) in enumerate(prepared):
        started = time.perf_counter()
        # One request per case; avoid implicit expensive retries. Persist the
        # counter before sending and never log the Authorization header.
        manifest["api_requests"] += 1
        save(out / "manifest.json", manifest)
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=args.timeout, allow_redirects=False)
            records[i]["http_status"] = response.status_code
            if not response.ok:
                # Retain bounded provider diagnostics, with credential redaction.
                records[i]["provider_error"] = response.text.replace(api_key, "[REDACTED]")[:2000]
                raise ValueError(f"HTTP {response.status_code}")
            data = response.json()
            # Preserve model output for offline parser diagnosis; credentials
            # and request headers are never included in these artifacts.
            response_dir = out / "responses"
            response_dir.mkdir(exist_ok=True)
            safe_data = json.loads(json.dumps(data).replace(api_key, "[REDACTED]"))
            save(response_dir / f"{i:03d}.json", safe_data)
            records[i]["usage"] = data.get("usage", {})
            records[i]["response_model"] = data.get("model")
            records[i]["finish_reason"] = data.get("choices", [{}])[0].get("finish_reason")
            if "claude" in str(data.get("model", "")).lower():
                raise ValueError("endpoint returned a disallowed model")
            prediction = parse_prediction(data, {step.source_step_id for step in item.steps}, args.recover_reason_quotes)
            records[i].update(prediction, status="complete", exact_hit=prediction["failure_id"] in label.root_cause_steps)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            # Exception text may include a proxy URL/credentials; retain only
            # the safe error category (HTTP status is separately available).
            records[i].update(status="failed", error_type=type(exc).__name__, exact_hit=False)
        records[i]["elapsed_seconds"] = time.perf_counter() - started
        save(out / "records.json", records)
        print(f"[{i+1}/{len(files)}] {item.trajectory_id}: {records[i]['status']}", flush=True)
        if records[i]["status"] == "failed":
            # Stop on failure for inspection instead of charging for a run with
            # a bad model/endpoint/context configuration.
            break
    complete = all(row["status"] == "complete" for row in records)
    manifest["status"] = "complete" if complete else "incomplete"
    save(out / "manifest.json", manifest)
    if complete:
        save(out / "metrics.json", {"trajectories": len(records), "exact_hits": sum(r["exact_hit"] for r in records),
             "exact_accuracy": sum(r["exact_hit"] for r in records) / len(records), "scope": manifest["scope"],
             "selection": args.selection, "format_recovered_cases": sum(r["format_recovered"] for r in records),
             "reported_total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in records),
             "usage_missing_cases": sum(not r.get("usage") for r in records)})
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
