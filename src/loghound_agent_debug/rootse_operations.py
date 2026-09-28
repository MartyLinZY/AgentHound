"""Conservative RootSE action parsing and file-version dependency tracking."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Operation:
    kind: str
    path: str = ""
    source_path: str = ""
    destination_path: str = ""
    status: str = "unknown"
    evidence: str = ""
    mode: str = ""


@dataclass
class Dependency:
    source_ordinal: int
    target_ordinal: int
    relation: str
    evidence: dict[str, Any]


@dataclass
class FileState:
    producer: int | None = None
    exists: bool | None = True
    uncertain: bool = False
    evidence: str = ""


_READ_TOOLS = {"read_file", "open_file", "view", "cat", "read", "search_file", "grep"}
_WRITE_TOOLS = {"write_file", "edit_file", "str_replace", "replace", "append_file", "create_file", "delete_file", "copy_file", "move_file"}
_PATH_KEYS = {"path", "file", "filename", "file_path", "target", "target_path"}


def _norm(path: str, cwd: str) -> str | None:
    path = path.strip().strip("'\"` ")
    if not path or any(ch in path for ch in "*$(){}"):
        return None
    if not path.startswith("/"):
        if not cwd:
            return None
        path = posixpath.join(cwd, path)
    return posixpath.normpath(path)


def _execution_status(action: str, observation: str, ok: Any = None) -> str:
    if ok is False:
        return "not_applied"
    if ok is True:
        return "applied" if not re.search(r"(?:failed|error|exception|not found)", observation, re.I) else "unknown"
    if re.search(r"(?:exit code|exit status)\s*[:=]?\s*[1-9]\d*|command not found|Traceback|Permission denied", observation, re.I):
        return "unknown"
    if re.search(r"(?:successfully|updated|written|created|deleted|saved|copied|moved|has been edited|has been created|here's the result of running `cat -n`|files and directories up to)", observation, re.I):
        return "applied"
    return "unknown"


def parse_operations(step: Any) -> list[Operation]:
    """Parse only recognizable tool/file actions; unsupported actions stay absent."""
    try:
        action = json.loads(step.action_text)
    except (json.JSONDecodeError, TypeError):
        action = step.action_text
    items = action if isinstance(action, list) else [action]
    operations: list[Operation] = []
    step_cwd = getattr(step, "working_dir", "")
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("func_name") or item.get("name") or item.get("command_name") or item.get("tool") or "").lower()
            args = item.get("arguments", item.get("args", item.get("input", item)))
            if not isinstance(args, dict):
                args = {"command": args}
            values = {str(k).lower(): v for k, v in args.items()}
            cwd = str(values.get("working_dir") or values.get("cwd") or step_cwd)
            evidence = json.dumps(item, ensure_ascii=False, sort_keys=True)
            ok = item.get("call_ok", item.get("success"))
            names = name.replace(".", "_").split("_")
            lowered = "_".join(names)
            path_value = next((values[k] for k in _PATH_KEYS if k in values and isinstance(values[k], str)), None)
            if path_value is None:
                # Code/search tools only become file operations with an explicit file path.
                path_value = None
            path = _norm(str(path_value or ""), cwd) if path_value else None
            if name in _READ_TOOLS or any(x in name for x in ("read_file", "open_file", "view_file")):
                if path:
                    operations.append(Operation("read", path, status=_execution_status(evidence, step.observation_text, ok), evidence=evidence))
            elif name in _WRITE_TOOLS or any(x in name for x in ("write_file", "edit_file", "replace_file", "append_file", "create_file", "delete_file")):
                if path:
                    kind = "delete" if "delete" in name else "modify"
                    mode = "overwrite" if any(x in name for x in ("write_file", "create_file")) else "partial"
                    operations.append(Operation(kind, path, status=_execution_status(evidence, step.observation_text, ok), evidence=evidence, mode=mode))
            elif any(x in name for x in ("copy", "move", "rename")):
                src = next((values[k] for k in ("source", "src", "source_path", "old_path") if isinstance(values.get(k), str)), None)
                dst = next((values[k] for k in ("destination", "dst", "destination_path", "new_path", "target") if isinstance(values.get(k), str)), None)
                if src and dst:
                    operations.append(Operation("move" if "move" in name or "rename" in name else "copy", status=_execution_status(evidence, step.observation_text, ok), source_path=_norm(src, cwd) or "", destination_path=_norm(dst, cwd) or "", evidence=evidence))
            elif name in {"write_patch", "apply_patch", "edit_code", "modify_code", "run_patch"}:
                operations.append(Operation("unknown", status=_execution_status(evidence, step.observation_text, ok), evidence=evidence))
            command = values.get("command")
            # OpenHands stores str_replace_editor arguments under `input`.
            # Resolve its explicit command and path instead of dropping it.
            if name == "str_replace_editor":
                editor_command = str(values.get("command", "")).lower()
                editor_path = values.get("path")
                path = _norm(str(editor_path or ""), cwd) if editor_path else None
                if path and editor_command == "view":
                    operations.append(Operation("read", path, status=_execution_status(evidence, step.observation_text), evidence=evidence))
                elif path and editor_command in {"str_replace", "insert", "edit"}:
                    operations.append(Operation("modify", path, status=_execution_status(evidence, step.observation_text), evidence=evidence, mode="partial"))
                elif path and editor_command == "create":
                    operations.append(Operation("modify", path, status=_execution_status(evidence, step.observation_text), evidence=evidence, mode="overwrite"))
                # Undo may restore an unknown prior file version; do not use stale state.
                elif path and editor_command == "undo_edit":
                    operations.append(Operation("modify", path, status="unknown", evidence=evidence, mode="partial"))
                continue
            if name in {"bash", "terminal", "execute_bash", "run_command"} and isinstance(command, str):
                action = command
            elif name in {"bash", "terminal", "execute_bash", "run_command"}:
                continue
            elif isinstance(command, str):
                action = command
            else:
                continue
        if isinstance(action, str):
            # Supported shell subset. Complex or composed commands are ignored conservatively.
            cwd = step_cwd
            for fragment in re.split(r"\s*(?:&&|\|\||;)\s*", action):
                try:
                    words = shlex.split(fragment)
                except ValueError:
                    continue
                if not words:
                    continue
                cmd = posixpath.basename(words[0])
                if cmd == "cd" and len(words) > 1:
                    cwd = _norm(words[1], cwd) or ""
                    continue
                args = [w for w in words[1:] if not w.startswith("-")]
                fileargs = [x for x in args if re.search(r"(?:/|\.[A-Za-z0-9]{1,8}$)", x)]
                status = _execution_status(fragment, step.observation_text)
                read_status = "applied" if re.search(r"<returncode>\s*0\s*</returncode>", step.observation_text, re.I) else status
                redirects = [i for i, word in enumerate(words[:-1]) if word in {">", ">>"}]
                if redirects:
                    dest = words[redirects[-1] + 1]
                    norm = _norm(dest, cwd)
                    if norm:
                        redirect_status = status
                        if cmd in {"echo", "printf"} and re.search(r"<returncode>\s*0\s*</returncode>", step.observation_text, re.I):
                            redirect_status = "applied"
                        operations.append(Operation("modify", norm, status=redirect_status, evidence=fragment, mode="partial" if words[redirects[-1]] == ">>" else "overwrite"))
                elif cmd == "sed" and "-i" in words and fileargs:
                    for p in fileargs:
                        norm = _norm(p, cwd)
                        if norm:
                            operations.append(Operation("modify", norm, status="applied" if re.search(r"\b(updated|replaced|changed)\b", step.observation_text, re.I) else status, evidence=fragment, mode="partial"))
                elif cmd in {"cat", "less", "head", "tail", "sed", "grep", "rg"} and fileargs:
                    for p in fileargs:
                        norm = _norm(p, cwd)
                        if norm:
                            operations.append(Operation("read", norm, status=read_status, evidence=fragment))
                elif cmd in {"rm"} and fileargs:
                    for p in fileargs:
                        norm = _norm(p, cwd)
                        if norm:
                            operations.append(Operation("delete", norm, status=status, evidence=fragment))
                elif cmd in {"cp", "mv"} and len(fileargs) >= 2:
                    a, b = (_norm(x, cwd) for x in fileargs[-2:])
                    if a and b:
                        operations.append(Operation("move" if cmd == "mv" else "copy", status=status, source_path=a, destination_path=b, evidence=fragment))
                elif cmd in {"tee"} and fileargs:
                    norm = _norm(fileargs[-1], cwd)
                    if norm:
                        operations.append(Operation("modify", norm, status=status, evidence=fragment, mode="overwrite"))
    return operations


def track_state_dependencies(steps: list[Any]) -> list[Dependency]:
    states: dict[str, FileState] = {}
    edges: list[Dependency] = []
    for step in steps:
        for op in parse_operations(step):
            if op.status == "not_applied":
                continue
            if op.kind == "read":
                state = states.get(op.path)
                if state and not state.uncertain and state.producer is not None:
                    missing_result = state.exists is False and bool(re.search(r"no such file|not found|does not exist", step.observation_text, re.I))
                    successful_read = state.exists is not False and op.status == "applied"
                    if missing_result or successful_read:
                        edges.append(Dependency(state.producer, step.ordinal, "state_dependency", {"path": op.path, "operation": "read_after_delete" if missing_result else "read", "source_evidence": state.evidence, "target_evidence": {"action": op.evidence, "observation": step.observation_text[:500]}}))
            elif op.kind in {"modify", "delete"}:
                state = states.get(op.path)
                if op.status == "unknown":
                    states[op.path] = FileState(None, None, True)
                    continue
                if op.kind == "modify" and op.mode != "overwrite" and state and not state.uncertain and state.producer is not None:
                    edges.append(Dependency(state.producer, step.ordinal, "state_dependency", {"path": op.path, "operation": op.mode or "partial", "source_evidence": state.evidence, "target_evidence": op.evidence}))
                states[op.path] = FileState(step.ordinal, op.kind != "delete", False, op.evidence)
            elif op.kind in {"copy", "move"}:
                source = states.get(op.source_path)
                if op.status == "unknown":
                    states[op.destination_path] = FileState(None, None, True)
                    if op.kind == "move":
                        states[op.source_path] = FileState(None, None, True)
                else:
                    if source and not source.uncertain and source.producer is not None:
                        edges.append(Dependency(source.producer, step.ordinal, "state_dependency", {"path": op.source_path, "operation": op.kind, "source_evidence": source.evidence, "target_evidence": op.evidence}))
                    states[op.destination_path] = FileState(step.ordinal, True, False, op.evidence)
                    if op.kind == "move":
                        states[op.source_path] = FileState(step.ordinal, False, False, op.evidence)
            elif op.kind == "unknown":
                for state in states.values():
                    state.uncertain = True
    return edges
