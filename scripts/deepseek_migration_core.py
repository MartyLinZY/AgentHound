"""Label-free graph extraction with frozen, target-local audited transport."""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loghound_agent_debug.rootse_dependencies import (
    select_candidates,
    _build_request,
    _parse_model_json,
    validate_response,
)
from loghound_agent_debug.schema import TrajectoryInput
from deepseek_suite_transport import RecordedClient, ExperimentAbort, digest, save, now


def build_target(
    item: TrajectoryInput, target: int, config: dict
) -> tuple[dict | None, dict]:
    candidates = select_candidates(item.steps, target)
    if not candidates:
        return None, {"status": "no_candidates", "target_ordinal": target}
    req, meta = _build_request(
        item.steps,
        target,
        candidates,
        max_field_chars=2**31,
        max_input_chars=config["max_input_chars"],
        max_output_tokens=config["max_tokens"],
    )
    if req is None and meta.get("status") == "budget_insufficient":
        raise ValueError("Full-field protocol would truncate input; refuse before API")
    if req is not None:
        if meta["candidate_ordinals"] != candidates or any(
            v["truncated"]
            for fields in meta["truncation"].values()
            for v in fields.values()
        ):
            raise ValueError(
                "Full-field protocol would truncate input; refuse before API"
            )
        meta["truncation_version"] = "selected_full_fields_v2"
    return req, meta


def parse_answer(content: str) -> tuple[dict, list[int]]:
    text = content.strip()
    if text.startswith("```"):
        matched = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if not matched:
            raise ValueError("Malformed JSON fence")
        text = matched.group(1)
    parsed, repairs = _parse_model_json(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("dependencies"), list):
        raise ValueError("Response must contain a dependencies list")
    return parsed, repairs


def run_target(
    item: TrajectoryInput,
    target: int,
    out: Path,
    config: dict,
    key: str,
    client_factory=RecordedClient,
) -> dict:
    req, meta = build_target(item, target, config)
    if req is None:
        raise ValueError("Only planned requests may be executed")
    signature = digest(
        {
            "messages": req["model_messages"],
            "model": config["model"],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
        }
    )
    state_path = out / "state.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old["request_sha256"] != signature:
            raise ValueError("Target replay request mismatch")
        if old["status"] != "running":
            return old
    state = {
        "trajectory_id": item.trajectory_id,
        "target_ordinal": target,
        "target_step_id": item.steps[target].source_step_id,
        "request_sha256": signature,
        "status": "running",
        "started_at": now(),
        "metadata": meta,
    }
    save(out / "input_metadata.json", meta)
    save(state_path, state)
    try:
        response = client_factory(out, config, key).create(
            model=config["model"], messages=req["model_messages"]
        )
        content = response.choices[0].message.content
        parsed, repairs = parse_answer(content)
        valid, rejected = validate_response(
            parsed,
            target,
            item.steps[target].source_step_id,
            item.steps,
            meta["candidate_ordinals"],
            req,
        )
        state.update(
            status="complete",
            validated_dependencies=valid,
            rejected_dependencies=rejected,
            json_repairs=repairs,
        )
    except ExperimentAbort as exc:
        if exc.status in {"model_mismatch", "replay_mismatch", "call_limit"}:
            raise RuntimeError(str(exc)) from exc
        state.update(
            status=exc.status, validated_dependencies=[], failure_reason=str(exc)
        )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        state.update(
            status="format_error",
            validated_dependencies=[],
            failure_reason=f"{type(exc).__name__}: {str(exc)[:400]}",
        )
    state["finished_at"] = now()
    save(state_path, state)
    return state
