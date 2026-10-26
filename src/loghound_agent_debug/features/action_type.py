"""Action-type prior channel.

Root causes concentrate on steps that make a decision (edits and pure
reasoning) rather than steps that gather evidence (file views, shell runs).
This module turns that one sentence into a fourth, label-free ranking channel:

1. :func:`classify` maps one raw ``action`` value to a coarse type.  The three
   agent families encode actions differently — OpenHands uses a dict, ACR a
   list of function calls, swe-agent a shell string — so each gets its own
   parser.
2. :func:`scores` gives decision steps 1 and evidence steps 0.  Two levels, no
   fitted weights, no per-agent table: RRF only consumes ranks, so a monotone
   two-level prior is all the channel needs.

Compressed trajectories (ACR collapses a whole run into a few stages) are
excluded by ``min_steps`` in the ranker: there "explore" calls are the decision
point, and the taxonomy no longer means the same thing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loghound_agent_debug.schema import Step

ACTION_TYPES = ("edit", "reasoning", "shell", "explore", "other")
DEFAULT_DECISION = ("edit", "reasoning")

EDIT_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}
SHELL_TOOLS = {"execute_bash", "bash", "run", "execute"}
EDIT_FUNCTIONS = ("write_patch", "create_file", "modify")
EXPLORE_FUNCTIONS = ("search_", "get_", "list_", "find_", "read_")
SHELL_FUNCTIONS = ("run_", "execute", "test")

SHELL_HINT = re.compile(
    r"^(?:python3?|pytest|pip|npm|npx|yarn|node|git|make|go|mvn|gradle|cd|bash|sh|source|conda|"
    r"docker|curl|wget|chmod|mkdir|rm|cp|mv|export|sudo|apt|apt-get|gcc|g\+\+|cargo|ruby|perl|"
    r"tox|coverage|ls\b.*&&|echo)\b", re.I)
EXPLORE_HINT = re.compile(r"^(?:ls|cat|head|tail|less|more|grep|rg|find|tree|file|wc|sed -n|awk)\b", re.I)
HEREDOC_WRITE = re.compile(r"(?:^|\n)\s*(?:cat|tee)\s+[^\n]*<<")
SHELL_RUN = re.compile(r"(?:^|\s)(?:python3?\s+-\s*<<|pytest|go test|npm test|tox)\b")
REDIRECT = re.compile(r">\s*\S+\.\w+$")


def classify(raw: Any) -> str:
    """Map one raw RootSE ``action`` value to ``edit/reasoning/shell/explore/other``."""
    if raw is None or raw == "" or raw == [] or raw == {}:
        # OpenHands records a pure-reasoning step as an empty action; that is
        # real semantics, not missing data (60% of its root causes land there).
        return "reasoning"
    if isinstance(raw, dict):
        return _classify_openhands(raw)
    if isinstance(raw, list):
        return _classify_acr(raw)
    return _classify_shell_text(str(raw))


def _classify_openhands(raw: dict[str, Any]) -> str:
    tool = str(raw.get("tool") or "")
    payload = raw.get("input")
    command = str(payload.get("command") or "") if isinstance(payload, dict) else ""
    if tool == "str_replace_editor":
        if command == "view":
            return "explore"
        return "edit" if command in EDIT_COMMANDS else "other"
    if tool in SHELL_TOOLS:
        return "shell"
    return "other"


def _classify_acr(raw: list[Any]) -> str:
    names = [str(item.get("func_name") or "") for item in raw if isinstance(item, dict)]
    if any(name.startswith(EDIT_FUNCTIONS) for name in names):
        return "edit"
    if any(name.startswith(EXPLORE_FUNCTIONS) for name in names):
        return "explore"
    if any(name.startswith(SHELL_FUNCTIONS) for name in names):
        return "shell"
    return "other"


def _classify_shell_text(text: str) -> str:
    low = text.strip().lower()
    if not low:
        return "reasoning"
    if "str_replace_editor" in low:
        if " view" in low:
            return "explore"
        if " create" in low or " str_replace" in low or " insert" in low:
            return "edit"
        return "other"
    # Order matters: a write is checked before a read, because
    # `grep ... > patch.py` and `cat > f.py << EOF` are writes.
    if "apply_patch" in low or re.search(r"sed\s+-i", low) or HEREDOC_WRITE.search(low):
        return "edit"
    if SHELL_RUN.search(low):
        return "shell"
    if EXPLORE_HINT.search(low):
        return "edit" if REDIRECT.search(low) else "explore"
    return "shell" if SHELL_HINT.search(low) else "other"


def _raw_action(step: Step) -> Any:
    """Read the raw action, falling back to parsing the stringified one."""
    raw = step.rootse_fields.get("raw_action")
    if raw is not None:
        return raw
    text = step.action_text
    if text[:1] in "[{":
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            pass
    return text


def kinds(steps: list[Step]) -> list[str]:
    """Classify every step of a trajectory."""
    return [classify(_raw_action(step)) for step in steps]


def scores(action_kinds: list[str], settings: dict[str, Any]) -> list[float]:
    """Two-level prior: 1 for decision steps, 0 for evidence steps."""
    decision = set(settings.get("decision") or DEFAULT_DECISION)
    return [1.0 if kind in decision else 0.0 for kind in action_kinds]
