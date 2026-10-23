"""Label-isolated adapter for the public RootSE JSON release."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loghound_agent_debug.schema import DiagnosticTest, FailureObservation, Label, Step, TrajectoryInput


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


_ANCHOR_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_ANCHOR_STOP = {
    "the", "this", "that", "with", "from", "have", "will", "your", "they",
    "them", "then", "than", "into", "file", "files", "code", "test", "tests",
    "error", "line", "true", "false", "none", "self", "cls", "args", "kwargs",
    "return", "value", "values", "import", "class", "def", "result",
    "expected", "actual", "assert", "assertion", "failed", "failure",
}


def extract_failure_anchors(raw: dict[str, Any]) -> list[str]:
    """Extract code symbols from failure-test error descriptions only.

    This function must not read failure_id, failure_reason, or gold_patch: they
    are annotations, and using them would leak the answer into the ranking
    input.  The order of first appearance is kept so runs stay reproducible.
    """
    texts: list[str] = []
    failed_tests = raw.get("failed_tests")
    if isinstance(failed_tests, dict):
        for value in failed_tests.values():
            if isinstance(value, dict):
                texts.append(_text(value.get("error_description")))
    blob = " ".join(texts)
    seen: dict[str, None] = {}
    for token in _ANCHOR_TOKEN.findall(blob):
        if token.lower() in _ANCHOR_STOP:
            continue
        seen.setdefault(token, None)
    return list(seen)


def _trajectory_id(raw: dict[str, Any], path: Path) -> str:
    # Both official layouts must address the same run/cache key: HF data/ and
    # TrajAudit RootSE/. Do not include the download directory in the identity.
    for parent in path.parents:
        if parent.name in {"data", "RootSE"}:
            return f"rootse/{path.relative_to(parent).as_posix()}".removesuffix(".json")
    rel = path.relative_to(path.parents[3]).as_posix() if len(path.parents) > 3 else path.as_posix()
    return f"rootse/{rel.removeprefix('data/')}".removesuffix(".json")


def load_rootse_file(path: str | Path, protocol: str = "trace-only") -> tuple[TrajectoryInput, Label]:
    """Read one raw instance, keeping annotations out of the model input."""
    if protocol not in {"trace-only", "postmortem"}:
        raise ValueError(f"unknown diagnosis protocol: {protocol}")
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_steps = raw.get("original_traj")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{path}: original_traj must be a non-empty list")
    trajectory_id = _trajectory_id(raw, path)
    steps: list[Step] = []
    observations: list[FailureObservation] = []
    for ordinal, item in enumerate(raw_steps):
        source_id = _text(item.get("index", ordinal))
        thought, response = _text(item.get("thought")), _text(item.get("response"))
        decision = "\n".join(x for x in (thought, response) if x)
        action = _text(item.get("action"))
        observation = _text(item.get("observation"))
        state = item.get("state")
        cwd = ""
        if isinstance(state, dict):
            cwd = _text(state.get("working_dir") or state.get("cwd") or state.get("current_dir"))
        elif isinstance(state, str):
            match = re.search(r"(?:working[_ ]dir|cwd)\s*[:=]\s*([^\s]+)", state, re.I)
            cwd = match.group(1) if match else ""
        tool_name = None
        if isinstance(item.get("action"), list) and item["action"]:
            first = item["action"][0]
            if isinstance(first, dict):
                tool_name = _text(first.get("func_name") or first.get("name") or first.get("command_name")) or None
        steps.append(Step(
            ordinal=ordinal,
            source_step_id=source_id,
            source_event_ids=[source_id],
            agent=_text(raw.get("agent")) or "unknown",
            role="assistant",
            raw_content="",
            decision_text=decision,
            action_text=action,
            observation_text=observation,
            tool_name=tool_name,
            rootse_fields={"thought": thought, "response": response, "action": action, "observation": observation},
            working_dir=cwd,
        ))
        if re.search(r"Traceback|AssertionError|FAILED|Error:|No such file|command not found", observation, re.I):
            observations.append(FailureObservation(source_id, "trace_failure", observation, "trace"))
    source_ids = [step.source_step_id for step in steps]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"{path}: duplicate original step IDs")
    diagnostic_tests = []
    if protocol == "postmortem":
        # Match TrajAudit's first-test policy. Explicit allow-list: gold_patch,
        # failure_id/reason and other annotation fields never enter the input.
        tests = raw.get("failed_tests")
        if not isinstance(tests, dict) or not tests:
            raise ValueError(f"{path}: postmortem protocol requires failed_tests")
        test_id, test = next(iter(tests.items()))
        if not isinstance(test, dict):
            raise ValueError(f"{path}: failed_tests entry must be an object")
        diagnostic_tests.append(DiagnosticTest(
            test_id=str(test_id), code=_text(test.get("code")),
            error_description=_text(test.get("error_description")),
        ))
    item = TrajectoryInput(
        schema_version="0.1.0", dataset="rootse", trajectory_id=trajectory_id,
        task=_text(raw.get("original_mission_prompt")), domain="coding", outcome="failure",
        failure_observations=observations, steps=steps,
        protocol=protocol, diagnostic_tests=diagnostic_tests,
        task_id=_text(raw.get("instance_id")), repository=_text(raw.get("repo")),
        base_commit=_text(raw.get("base_commit")),
        # Follow the same first-test input policy as diagnostic_tests.
        failure_anchors=extract_failure_anchors({"failed_tests": {
            t.test_id: {"error_description": t.error_description} for t in diagnostic_tests
        }}),
    )
    # Read the annotation only after constructing the label-free input.
    label = Label(
        trajectory_id=trajectory_id,
        root_cause_steps=[_text(raw.get("failure_id"))],
        reason=_text(raw.get("failure_reason")),
        annotation_method="human_annotated_earliest_decisive_error",
        provenance="dengdan1999/RootSE",
    )
    return item, label


def load_rootse(root: str | Path, protocol: str = "trace-only") -> tuple[list[TrajectoryInput], list[Label]]:
    root = Path(root)
    files = sorted((root / "data").rglob("*.json"))
    if not files and root.name == "RootSE":
        files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"no RootSE instance JSON files below {root / 'data'}")
    pairs = [load_rootse_file(path, protocol=protocol) for path in files]
    ids = [item.trajectory_id for item, _ in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("RootSE adapter produced duplicate trajectory IDs")
    return [x for x, _ in pairs], [y for _, y in pairs]
