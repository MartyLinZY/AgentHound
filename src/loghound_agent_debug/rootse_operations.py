"""Conservative RootSE action parsing and file-version dependency tracking."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass
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


STATE_TRACKER_VERSION = "explicit_operations_v2"
_READ_TOOLS = {"read_file", "open_file", "view", "cat", "read", "search_file", "grep"}
_WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "str_replace",
    "replace",
    "append_file",
    "create_file",
    "delete_file",
}
_SHELL_TOOLS = {"bash", "terminal", "execute_bash", "run_command"}
_PATCH_TOOLS = {"write_patch", "apply_patch", "edit_code", "modify_code", "run_patch"}
# Ordered aliases: hash randomization must not decide which file is affected.
_PATH_KEYS = (
    "path",
    "file",
    "filename",
    "file_path",
    "file_name",
    "target",
    "target_path",
)
_EXIT = re.compile(
    r"(?:exit\s+(?:code|status)\s*[:=]?\s*|<returncode>\s*)(-?\d+)\b", re.I
)
_SUCCESS = re.compile(
    r"successfully|updated|written|created|deleted|saved|copied|moved|has been edited|here's the result of running `cat -n`|files and directories up to",
    re.I,
)
_SHELL_OPERATORS = {"&&", "||", ";", "|", "&", "<", "<<", "<<<", ">&", "&>", ";;"}


def _norm(path: str, cwd: str) -> str | None:
    path = path.strip().strip("'\"` ")
    if not path or any(ch in path for ch in "*$(){}"):
        return None
    if not path.startswith("/"):
        if not cwd:
            return None
        path = posixpath.join(cwd, path)
    return posixpath.normpath(path)


def _execution_status(observation: str, ok: Any = None) -> str:
    if ok is False:
        return "not_applied"
    exits = [int(m.group(1)) for m in _EXIT.finditer(observation)]
    if any(code != 0 for code in exits):
        return "unknown"  # A failed edit may have partially changed a file.
    if exits:
        return "applied"
    if re.search(
        r"command not found|Traceback|Permission denied|Error:|not found",
        observation,
        re.I,
    ):
        return "unknown"
    return "applied" if ok is True or _SUCCESS.search(observation) else "unknown"


def _unknown(evidence: str) -> list[Operation]:
    """Unknown mutation invalidates state; it does not invent a producer."""
    return [Operation("unknown", status="unknown", evidence=evidence)]


def _first_string(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    return next((values[key] for key in keys if isinstance(values.get(key), str)), "")


def _editor(
    command: str, raw_path: str, cwd: str, status: str, evidence: str
) -> list[Operation]:
    path = _norm(raw_path, cwd)
    if command == "view":
        return (
            [Operation("read", path, status=status, evidence=evidence)] if path else []
        )
    if command in {"create", "str_replace", "insert", "edit", "undo_edit"}:
        if not path:
            return _unknown(evidence)
        return [
            Operation(
                "modify",
                path,
                status="unknown" if command == "undo_edit" else status,
                evidence=evidence,
                mode="overwrite" if command == "create" else "partial",
            )
        ]
    return _unknown(evidence)


def _positionals(words: list[str], value_options: set[str]) -> list[str]:
    """Skip known option arguments; unknown shell syntax is not interpreted."""
    result = []
    skip = False
    options = True
    for word in words:
        if skip:
            skip = False
        elif options and word == "--":
            options = False
        elif options and word in value_options:
            skip = True
        elif not options or not word.startswith("-"):
            result.append(word)
    return result


def _shell(command: str, cwd: str, observation: str, ok: Any = None) -> list[Operation]:
    try:
        # Tokenize quoted payloads intact; never split a quoted ';' or '&&'.
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return _unknown(command)
    if not words:
        return []
    # Only this compound form has an unambiguous successful child execution.
    if len(words) >= 4 and words[0] == "cd" and words[2] == "&&":
        cwd = _norm(words[1], cwd) or ""
        words = words[3:]
    if any(word in _SHELL_OPERATORS for word in words):
        return _unknown(command)
    cmd = posixpath.basename(words[0])
    status = _execution_status(observation, ok)
    if cmd == "str_replace_editor":
        return (
            _editor(words[1], words[2], cwd, status, command)
            if len(words) >= 3
            else _unknown(command)
        )
    redirects = [i for i, word in enumerate(words) if word in {">", ">>"}]
    if redirects:
        if (
            len(redirects) != 1
            or redirects[0] != len(words) - 2
            or cmd not in {"echo", "printf"}
        ):
            return _unknown(command)
        pos = redirects[0]
        path = _norm(words[pos + 1], cwd)
        return (
            [
                Operation(
                    "modify",
                    path,
                    status=status,
                    evidence=command,
                    mode="partial" if words[pos] == ">>" else "overwrite",
                )
            ]
            if path
            else _unknown(command)
        )
    value_options = (
        {"-n", "-c", "--lines", "--bytes"}
        if cmd in {"head", "tail"}
        else {"-A", "-B", "-C", "-e", "-f"}
    )
    args = _positionals(words[1:], value_options)
    if cmd in {"grep", "rg", "sed"}:
        # The script/pattern is not a file, even when it contains slashes.
        if not any(x in words for x in {"-e", "-f"}):
            args = args[1:]
    paths = [_norm(arg, cwd) for arg in args]
    if cmd in {"cat", "less", "head", "tail", "sed", "grep", "rg"}:
        kind = (
            "modify"
            if cmd == "sed" and any(x == "-i" or x.startswith("-i.") for x in words)
            else "read"
        )
        if kind == "modify" and (not paths or any(path is None for path in paths)):
            return _unknown(command)
        return [
            Operation(
                kind,
                path,
                status=status,
                evidence=command,
                mode="partial" if kind == "modify" else "",
            )
            for path in paths
            if path
        ]
    if cmd == "tee":
        if not paths or any(path is None for path in paths):
            return _unknown(command)
        mode = "partial" if "-a" in words or "--append" in words else "overwrite"
        return [
            Operation("modify", path, status=status, evidence=command, mode=mode)
            for path in paths
        ]
    if cmd == "rm":
        return [
            Operation("delete", path, status=status, evidence=command)
            for path in paths
            if path
        ] or _unknown(command)
    if cmd in {"cp", "mv"}:
        # Directory destinations/recursive copies need filesystem snapshots.
        if (
            len(args) != 2
            or not all(paths)
            or args[-1].endswith("/")
            or any(x in words for x in {"-r", "-R", "-a", "--recursive"})
        ):
            return _unknown(command)
        return [
            Operation(
                "move" if cmd == "mv" else "copy",
                status=status,
                source_path=paths[0],
                destination_path=paths[1],
                evidence=command,
            )
        ]
    if cmd == "find" and any(
        x in words for x in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
    ):
        return _unknown(command)
    if cmd in {"cd", "pwd", "ls", "find", "wc", "echo", "printf", "true", "false"}:
        return []
    # Arbitrary scripts, patches and executables may mutate files. Do not
    # silently preserve an old producer through an unparsed action.
    return _unknown(command)


def _tool(item: dict[str, Any], step: Any) -> list[Operation]:
    name = str(
        item.get("func_name")
        or item.get("name")
        or item.get("command_name")
        or item.get("tool")
        or ""
    ).lower()
    args = item.get("arguments", item.get("args", item.get("input", item)))
    values = (
        {str(k).lower(): v for k, v in args.items()}
        if isinstance(args, dict)
        else {"command": args}
    )
    cwd = str(values.get("working_dir") or values.get("cwd") or step.working_dir)
    evidence = json.dumps(item, ensure_ascii=False, sort_keys=True)
    ok = item.get("call_ok", item.get("success"))
    status = _execution_status(step.observation_text, ok)
    raw_path = _first_string(values, _PATH_KEYS)
    path = _norm(raw_path, cwd) if raw_path else None
    if name == "str_replace_editor":
        return _editor(
            str(values.get("command", "")).lower(), raw_path, cwd, status, evidence
        )
    if any(x in name for x in ("copy", "move", "rename")):
        src = _norm(
            _first_string(values, ("source", "src", "source_path", "old_path")), cwd
        )
        dst = _norm(
            _first_string(
                values, ("destination", "dst", "destination_path", "new_path", "target")
            ),
            cwd,
        )
        if not src or not dst:
            return _unknown(evidence)
        return [
            Operation(
                "move" if "move" in name or "rename" in name else "copy",
                status=status,
                source_path=src,
                destination_path=dst,
                evidence=evidence,
            )
        ]
    if name in _READ_TOOLS or any(
        x in name for x in ("read_file", "open_file", "view_file")
    ):
        return (
            [Operation("read", path, status=status, evidence=evidence)] if path else []
        )
    if name in _WRITE_TOOLS or any(
        x in name
        for x in (
            "write_file",
            "edit_file",
            "replace_file",
            "append_file",
            "create_file",
            "delete_file",
        )
    ):
        if not path:
            return _unknown(evidence)
        kind = "delete" if "delete" in name else "modify"
        mode = (
            "overwrite"
            if any(x in name for x in ("write_file", "create_file"))
            else "partial"
        )
        return [Operation(kind, path, status=status, evidence=evidence, mode=mode)]
    if name in _PATCH_TOOLS:
        return _unknown(evidence)
    if name in _SHELL_TOOLS and isinstance(values.get("command"), str):
        return _shell(values["command"], cwd, step.observation_text, ok)
    return []  # Non-file tools (e.g. task tracking) do not imply file writes.


def parse_operations(step: Any) -> list[Operation]:
    """Parse explicit tool actions and a bounded shell subset without execution."""
    try:
        action = json.loads(step.action_text)
    except (json.JSONDecodeError, TypeError):
        action = step.action_text
    items = action if isinstance(action, list) else [action]
    operations = []
    for item in items:
        if isinstance(item, dict):
            parsed = _tool(item, step)
        elif isinstance(item, str):
            parsed = _shell(
                item, getattr(step, "working_dir", ""), step.observation_text
            )
        else:
            continue
        # Shared batch output cannot certify each individual mutation.
        if len(items) > 1 and not (
            isinstance(item, dict) and any(k in item for k in ("call_ok", "success"))
        ):
            for op in parsed:
                op.status = "unknown"
        operations.extend(parsed)
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
                    missing_result = state.exists is False and bool(
                        re.search(
                            r"no such file|not found|does not exist",
                            step.observation_text,
                            re.I,
                        )
                    )
                    successful_read = (
                        state.exists is not False and op.status == "applied"
                    )
                    if missing_result or successful_read:
                        edges.append(
                            Dependency(
                                state.producer,
                                step.ordinal,
                                "state_dependency",
                                {
                                    "path": op.path,
                                    "operation": "read_after_delete"
                                    if missing_result
                                    else "read",
                                    "source_evidence": state.evidence,
                                    "target_evidence": {
                                        "action": op.evidence,
                                        "observation": step.observation_text[:500],
                                    },
                                },
                            )
                        )
            elif op.kind in {"modify", "delete"}:
                state = states.get(op.path)
                if op.status == "unknown":
                    states[op.path] = FileState(None, None, True)
                    continue
                if (
                    op.kind == "modify"
                    and op.mode != "overwrite"
                    and state
                    and not state.uncertain
                    and state.producer is not None
                ):
                    edges.append(
                        Dependency(
                            state.producer,
                            step.ordinal,
                            "state_dependency",
                            {
                                "path": op.path,
                                "operation": op.mode or "partial",
                                "source_evidence": state.evidence,
                                "target_evidence": op.evidence,
                            },
                        )
                    )
                states[op.path] = FileState(
                    step.ordinal, op.kind != "delete", False, op.evidence
                )
            elif op.kind in {"copy", "move"}:
                source = states.get(op.source_path)
                if op.status == "unknown":
                    states[op.destination_path] = FileState(None, None, True)
                    if op.kind == "move":
                        states[op.source_path] = FileState(None, None, True)
                else:
                    if source and not source.uncertain and source.producer is not None:
                        edges.append(
                            Dependency(
                                source.producer,
                                step.ordinal,
                                "state_dependency",
                                {
                                    "path": op.source_path,
                                    "operation": op.kind,
                                    "source_evidence": source.evidence,
                                    "target_evidence": op.evidence,
                                },
                            )
                        )
                    states[op.destination_path] = FileState(
                        step.ordinal, True, False, op.evidence
                    )
                    if op.kind == "move":
                        states[op.source_path] = FileState(
                            step.ordinal, False, False, op.evidence
                        )
            elif op.kind == "unknown":
                for state in states.values():
                    state.uncertain = True
    return [edge for edge in edges if edge.source_ordinal < edge.target_ordinal]
