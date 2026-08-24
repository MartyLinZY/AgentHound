"""Adapter for the local AgenTracer coding-test parquet export."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from loghound_agent_debug.schema import (
    FailureObservation,
    Label,
    Step,
    TrajectoryInput,
)

_SOLUTION_FILE_RE = re.compile(r"[A-Za-z0-9_.-]+_attacked_solution\.py", re.I)
_PATH_RE = re.compile(
    r"(?:^|[\s'\"`])((?:/[^\s'\"`]+)|(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)|(?:[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml)))"
)
_OBSERVATION_PAIR_RE = re.compile(
    r"\[command\]:\s*(.*?)\n\[command output\]\s*:\s*(.*?)(?=\n\[command\]:|\Z)",
    re.S | re.I,
)
_TERMINAL_RE = re.compile(r"Terminal output:\s*(.*)", re.S | re.I)
_CONFESSION_RE = re.compile(
    r"(?:previous attempt.{0,120}\bfailed\b|failed due to (?:a )?(?:syntax|indentation|compile|runtime) error|test cases?\s+.*?\bfailed\b)",
    re.I | re.S,
)
_NEGATIVE_OBSERVATION_RE = re.compile(
    r"(?:Traceback|AssertionError|NameError|IndentationError|\bFAILED\b|command not found|No such file|Permission denied|ModuleNotFoundError)",
    re.I,
)


def _plain(value: Any) -> str:
    """Convert scalar parquet values to text without exposing null literals."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _parse_history(value: Any) -> list[dict[str, Any]]:
    """Read ndarray/list history and common stringified representations."""
    if hasattr(value, "tolist"):
        # Parquet exports may materialize a history column as a NumPy array.
        value = value.tolist()
    if isinstance(value, str):
        try:
            # literal_eval handles Python-style stringified lists without code
            # execution; JSON is the fallback for strict JSON exports.
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("history is neither a list nor a parseable list string") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"unsupported history type: {type(value)!r}")
    result = []
    for item in value:
        if hasattr(item, "item"):
            item = item.item()
        if not isinstance(item, dict):
            raise ValueError(f"history item is not a mapping: {type(item)!r}")
        result.append(dict(item))
    return result


def _json_commands(content: str) -> list[dict[str, Any]]:
    """Extract command arrays from fenced JSON/Python blocks in one history item."""
    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.S | re.I)
    commands: list[dict[str, Any]] = []
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            # Older exports may use single quotes, which literal_eval can read.
            try:
                value = ast.literal_eval(block)
            except (ValueError, SyntaxError):
                continue
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            commands.extend(x for x in value if isinstance(x, dict))
    return commands


def _command_text(commands: Iterable[dict[str, Any]]) -> str:
    """Flatten structured commands into deterministic action text."""
    parts = []
    for command in commands:
        name = command.get("command_name") or command.get("name") or command.get("command")
        args = command.get("args", command.get("arguments", ""))
        if name is None:
            continue
        if isinstance(args, (dict, list)):
            # Stable key ordering makes downstream hashes and tests reproducible.
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        parts.append(f"{name} {_plain(args)}".strip())
    return " ; ".join(parts)


def _observation_text(content: str) -> str:
    """Extract only explicit Terminal output, excluding task/prompt text."""
    terminal = _TERMINAL_RE.search(content)
    if not terminal:
        return ""
    tail = terminal.group(1)
    pairs = _OBSERVATION_PAIR_RE.findall(tail)
    if pairs:
        return "\n".join(f"[command] {cmd.strip()}\n[command output] {out.strip()}" for cmd, out in pairs)
    # Preserve an unpaired terminal return for auditability; this is still
    # only text inside the explicit Terminal output section.
    return tail.strip()


def _decision_text(content: str) -> str:
    """Remove command/output scaffolding while retaining auditable reasoning text."""
    text = re.sub(r"```(?:json)?\s*\[.*?\]\s*```", " ", content, flags=re.S | re.I)
    text = _TERMINAL_RE.sub(" ", text)
    text = re.sub(r"^\s*(?:[^\n:]+\s+)?thinking:\s*", "", text, flags=re.I)
    # Keep the task repetition in the raw-derived field for auditability, but
    # remove the known orchestration scaffolding headings and empty tags.
    text = re.sub(r"^\s*#\s*(?:Past Experience|Tool State|Current Plan|Current Task|Response Language)\s*$", " ", text, flags=re.I | re.M)
    text = re.sub(r"<experience>.*?</experience>", " ", text, flags=re.I | re.S)
    text = re.sub(r"\n\s*No Plan\s*\n", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_solution_filename(text: str) -> str:
    """Mask injected solution filenames that otherwise act as lexical shortcuts."""
    return _SOLUTION_FILE_RE.sub("<SOLUTION_FILE>", text)


def _artifacts(text: str) -> list[str]:
    """Extract unique file-like artifacts from parsed step text."""
    found: list[str] = []
    for match in _PATH_RE.finditer(text):
        path = match.group(1).rstrip(".,:;)]}")
        path = _normalise_solution_filename(path)
        if path not in found:
            found.append(path)
    return found


def parse_history_entry(item: dict[str, Any], ordinal: int) -> Step:
    """Convert one raw history entry into one canonical decision step."""
    content = _plain(item.get("content"))
    commands = _json_commands(content)
    action = _command_text(commands)
    observation = _observation_text(content)
    decision = _normalise_solution_filename(_decision_text(content))
    action = _normalise_solution_filename(action)
    observation = _normalise_solution_filename(observation)
    # Keep the source ID for audits; ordinal is only the internal graph index.
    raw_step_id = item.get("step", ordinal)
    source_step_id = _plain(raw_step_id)
    agent = _plain(item.get("name")) or "unknown"
    role = _plain(item.get("role")) or "assistant"
    artifacts = _artifacts(decision + " " + action + " " + observation)
    tool_name = None
    if commands:
        tool_name = _plain(commands[0].get("command_name") or commands[0].get("name") or commands[0].get("command")) or None
    action_lower = action.lower()
    # Artifact edges are heuristic: only classify a file as read/written when
    # the action text contains an explicit read/write verb.
    read_action = bool(re.search(r"\b(read|open|view|cat|grep|find|list|inspect)\b", action_lower))
    write_action = bool(re.search(r"\b(write|insert|edit|create|replace|append|save|overwrite)\b|[>]", action_lower))
    file_artifacts = [x for x in artifacts if x.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml"))]
    return Step(
        ordinal=ordinal,
        source_step_id=source_step_id,
        source_event_ids=[source_step_id],
        agent=agent,
        role=role,
        raw_content=content,
        decision_text=decision,
        action_text=action,
        observation_text=observation,
        tool_name=tool_name,
        artifacts_read=file_artifacts if read_action else [],
        artifacts_written=file_artifacts if write_action else [],
    )


def _row_to_input(row: Any, dataset: str = "agentracer") -> tuple[TrajectoryInput, Label]:
    """Convert a parquet row into label-free input plus a separate annotation."""
    history = _parse_history(row["history"])
    steps = [parse_history_entry(item, ordinal) for ordinal, item in enumerate(history)]
    observations: list[FailureObservation] = []
    for step in steps:
        # Scan structured observation first; only use confession text when no
        # explicit terminal failure was extracted from that same step.
        confession = _CONFESSION_RE.search(step.decision_text)
        if step.observation_text and _NEGATIVE_OBSERVATION_RE.search(step.observation_text):
            observations.append(FailureObservation(step.source_step_id, "trace_failure", step.observation_text, "trace"))
        elif confession:
            observations.append(FailureObservation(step.source_step_id, "agent_confession", confession.group(0), "trace"))
    trajectory_id = _plain(row["question_ID"])
    input_item = TrajectoryInput(
        schema_version="0.1.0",
        dataset=dataset,
        trajectory_id=trajectory_id,
        task=_normalise_solution_filename(_plain(row["question"])),
        domain="coding",
        outcome="failure",
        failure_observations=observations,
        steps=steps,
    )
    # These fields are intentionally created here but written to labels.jsonl,
    # never embedded in TrajectoryInput.
    label = Label(
        trajectory_id=trajectory_id,
        root_cause_steps=[_plain(row["mistake_step"])],
        root_cause_agents=[_plain(row["mistake_agent"])],
        reason=_plain(row["mistake_reason"]),
        annotation_method="fault_injection",
        provenance="agentracer",
    )
    return input_item, label


def load_agentracer(path: str | Path, dataset: str = "agentracer") -> tuple[list[TrajectoryInput], list[Label]]:
    """Load and validate the local AgenTracer Parquet export."""
    frame = pd.read_parquet(path)
    required = {"question_ID", "question", "history", "mistake_agent", "mistake_step", "mistake_reason"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing AgenTracer columns: {sorted(missing)}")
    inputs, labels = [], []
    for _, row in frame.iterrows():
        # Preserve row order so input and label JSONL files remain aligned.
        item, label = _row_to_input(row, dataset=dataset)
        inputs.append(item)
        labels.append(label)
    return inputs, labels
