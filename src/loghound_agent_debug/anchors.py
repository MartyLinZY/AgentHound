"""Auditable failure anchors. An anchor is a symptom, not a root-cause label."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re
from typing import Any

from loghound_agent_debug.evidence import extract_evidence
from loghound_agent_debug.schema import Step, TrajectoryInput

ANCHOR_VERSION = "observed_failure_v1"

# Match observed output only. Do not scan instructions, proposed code or plans
# for error vocabulary. These rules are fixed before the offline experiment.
_RULES = [
    ("test_failure", 1.0, r"^\s*(?:FAILED\s+\S+|FAIL\s+\S+|--- FAIL:.*|FAILURES!?\b.*|[1-9]\d*\s+failing\b.*)|\b[1-9]\d*\s+failed\b|\bAssertionError\b"),
    ("tool_environment", 0.6, r"(?:command not found|No such file or directory|Permission denied|ModuleNotFoundError|No module named)"),
    ("runtime_error", 0.8, r"^\s*(?:Traceback \(most recent call last\):|(?:[\w.]+Error|Exception):[^\n]*|panic:[^\n]*|npm ERR![^\n]*|[^\n]*syntax error near unexpected token[^\n]*)"),
    ("nonzero_exit", 0.6, r"(?:exit\s+(?:code|status)\s*[:=]?\s*|<returncode>\s*)(?:-[1-9]\d*|[1-9]\d*)\b"),
]
_COMPILED = [(kind, weight, re.compile(pattern, re.I | re.M)) for kind, weight, pattern in _RULES]
_EXIT = re.compile(r"(?:exit\s+(?:code|status)\s*[:=]?\s*|<returncode>\s*)(-?\d+)\b", re.I)
_ENTITY = re.compile(r"(?<![\w])(?:/?[\w.@+-]+(?:/[\w.@+-]+)*\.(?:py|go|js|ts|tsx|jsx|java|c|cc|cpp|h|tmpl|yaml|yml|toml)|test_[A-Za-z0-9_]+)(?![\w])")


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    source_step_id: str | None
    step_ordinal: int | None
    category: str
    strength: float
    field: str
    start: int
    end: int
    quote: str
    entity_refs: list[str]
    execution_status: str
    provenance: str = "trace"
    recovery_status: str = "unknown"
    recovery_step_id: str | None = None
    test_id: str | None = None
    # Diagnostic grouping only: repetition does not multiply local scores.
    signature: str = ""


@dataclass(frozen=True)
class AnchorLink:
    anchor_id: str
    step_ordinal: int
    relation: str
    field: str
    start: int
    end: int
    quote: str
    entity: str = ""


@dataclass
class AnchorEvidence:
    anchors: list[Anchor] = field(default_factory=list)
    links: list[AnchorLink] = field(default_factory=list)
    local_scores: list[float] = field(default_factory=list)
    seeds: list[float] = field(default_factory=list)


def entity_refs(text: str) -> list[str]:
    """Exact path/test identifiers; no guessed function or coverage edges."""
    return sorted(set(match.group(0) for match in _ENTITY.finditer(text)))


def _status(text: str) -> str:
    exits = [int(m.group(1)) for m in _EXIT.finditer(text)]
    if any(value != 0 for value in exits):
        return "failed"
    if exits:
        return "reported_zero_exit"
    return "unknown"


def _anchor(step: Step, kind: str, weight: float, field_name: str, match: re.Match[str]) -> Anchor:
    text = getattr(step, field_name)
    # Entity context is the matched line, so a path printed elsewhere in a long
    # observation is not automatically described as the failed entity.
    start_line = text.rfind("\n", 0, match.start()) + 1
    end_line = text.find("\n", match.end())
    if end_line < 0:
        end_line = len(text)
    quote = match.group(0)
    signature = hashlib.sha256(f"{kind}:{quote.strip()}".encode()).hexdigest()
    return Anchor(
        f"trace:{step.source_step_id}:{kind}:{match.start()}", step.source_step_id,
        step.ordinal, kind, weight, field_name, match.start(), match.end(), quote,
        entity_refs(text[start_line:end_line]), _status(step.observation_text), signature=signature,
    )


def _legacy_anchors(item: TrajectoryInput, rules: dict[str, Any]) -> list[Anchor]:
    result = []
    for hit in extract_evidence(item, rules).hits:
        step = item.steps[hit.step_ordinal]
        for rule in rules.get("rules", []):
            if rule["name"] not in hit.rule_names or float(rule["weight"]) != hit.strength:
                continue
            found = False
            for field_name in rule.get("fields", []):
                for pattern in rule.get("compiled", []):
                    match = pattern.search(getattr(step, field_name, ""))
                    if match:
                        result.append(_anchor(step, hit.category, hit.strength, field_name, match))
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return result


def extract_anchors(item: TrajectoryInput, rules: dict[str, Any], mode: str = "structured") -> AnchorEvidence:
    """Extract anchored spans and explicit external-evidence retrieval links.

    Later success of the exact same command is recorded, but does not erase
    failure mass: its scope/version may differ. No repair causality is assumed.
    """
    if mode not in {"legacy", "structured"}:
        raise ValueError(f"unknown anchor mode: {mode}")
    anchors = _legacy_anchors(item, rules) if mode == "legacy" else []
    if mode == "structured":
        for step in item.steps:
            for kind, weight, pattern in _COMPILED:
                match = pattern.search(step.observation_text)
                if match:
                    anchors.append(_anchor(step, kind, weight, "observation_text", match))
    # A later zero exit alone is not a verified recovery (pipelines can mask
    # failure). Keep status qualified, require no recognized failure output.
    for i, anchor in enumerate(anchors):
        source = item.steps[anchor.step_ordinal]
        if not source.action_text.strip():
            continue
        for later in item.steps[source.ordinal + 1:]:
            if later.action_text != source.action_text or later.working_dir != source.working_dir:
                continue
            if _status(later.observation_text) != "reported_zero_exit":
                continue
            if any(pattern.search(later.observation_text) for _, _, pattern in _COMPILED):
                continue
            anchors[i] = replace(anchor, recovery_status="later_same_command_success_unverified", recovery_step_id=later.source_step_id)
            break
    for i, test in enumerate(item.diagnostic_tests):
        if not test.error_description.strip():
            continue
        anchors.append(Anchor(
            f"postmortem:{i}", None, None, "external_test_failure", 1.0,
            f"diagnostic_tests[{i}].error_description", 0, len(test.error_description),
            test.error_description, entity_refs(test.error_description + "\n" + test.code + "\n" + test.test_id),
            "failed", provenance=test.provenance, test_id=test.test_id,
            signature=hashlib.sha256(test.error_description.encode()).hexdigest(),
        ))

    result = AnchorEvidence(anchors=anchors, local_scores=[0.0] * len(item.steps), seeds=[0.0] * len(item.steps))
    for anchor in anchors:
        if anchor.step_ordinal is not None:
            ordinal = anchor.step_ordinal
            result.local_scores[ordinal] = max(result.local_scores[ordinal], anchor.strength)
            result.seeds[ordinal] = max(result.seeds[ordinal], anchor.strength)
            result.links.append(AnchorLink(anchor.anchor_id, ordinal, "observed_at", anchor.field, anchor.start, anchor.end, anchor.quote))
            continue
        # External test evidence has no historical step. Literal entity matches
        # are weak retrieval links, never state/causal/coverage dependencies.
        linked: list[AnchorLink] = []
        for step in item.steps:
            for field_name in ("action_text", "observation_text", "decision_text"):
                text = getattr(step, field_name)
                matches = [m for m in _ENTITY.finditer(text) if m.group(0) in anchor.entity_refs]
                if matches:
                    match = matches[0]
                    linked.append(AnchorLink(anchor.anchor_id, step.ordinal, "entity_match", field_name, match.start(), match.end(), match.group(0), match.group(0)))
                    break
        result.links.extend(linked)
        # Divide each external anchor's weight across all its candidates. A
        # frequently mentioned filename must not multiply the total seed mass.
        for link in linked:
            result.seeds[link.step_ordinal] += anchor.strength / len(linked)
    return result
