"""Evidence-aware failure and clue extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from loghound_agent_debug.schema import Step, TrajectoryInput


@dataclass(frozen=True)
class ClueHit:
    """One rule match on a step, including its clustering signature."""
    step_ordinal: int
    rule_names: tuple[str, ...]
    category: str
    strength: float
    signature: str


@dataclass(frozen=True)
class EvidenceResult:
    """All evidence features needed by the ranker and audit reports."""
    tier: str
    status: str
    clue_scores: list[float]
    hits: list[ClueHit]
    symptom_seeds: list[dict[str, Any]]
    failed_occurrences: int
    failed_clusters: int


def load_clue_rules(path: str | Path) -> dict[str, Any]:
    """Load clue rules and compile their regexes once for the whole run."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for rule in data.get("rules", []):
        # Compiled patterns are stored beside the declarative rule so the
        # extraction loop does not repeatedly compile the same expressions.
        rule["compiled"] = [re.compile(pattern, re.I | re.S) for pattern in rule.get("patterns", [])]
    return data


def _error_signature(step: Step) -> str:
    """Create a coarse signature used to collapse repeated failures."""
    observation = step.observation_text.lower()
    error_type = re.search(r"\b(traceback|[a-z]+error|failed|command not found|no such file|permission denied|modulenotfounderror)\b", observation, re.I)
    line = ""
    for raw in observation.splitlines():
        if re.search(r"error|failed|not found|permission", raw, re.I):
            line = re.sub(r"\d+", "<N>", raw)
            line = re.sub(r"\s+", " ", line).strip()
            break
    command = re.sub(r"\s+", " ", step.action_text.lower()).strip()
    return f"{command}|{error_type.group(1).lower() if error_type else ''}|{line}"


def _hash_signature(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _rule_matches(rule: dict[str, Any], step: Step) -> bool:
    """Match only the fields explicitly allowed by the rule configuration."""
    texts = []
    for field in rule.get("fields", []):
        value = getattr(step, field, "")
        if value:
            texts.append(value)
    text = "\n".join(texts)
    # Restricting the scanned text prevents task instructions or raw prompts
    # from being mistaken for observed failures.
    return bool(text) and any(pattern.search(text) for pattern in rule.get("compiled", []))


def _tier(input_item: TrajectoryInput, hits: list[ClueHit]) -> str:
    """Assign A/B/C according to the strongest observable failure evidence."""
    if any(hit.category in {"terminal_failure", "tool_env_failure"} for hit in hits):
        return "A"
    if any(hit.category == "negative_confession" for hit in hits):
        return "B"
    return "C"


def extract_evidence(input_item: TrajectoryInput, rules: dict[str, Any]) -> EvidenceResult:
    """Extract local clue scores and deduplicated symptom seeds per trajectory."""
    clue_scores = [0.0] * len(input_item.steps)
    hits: list[ClueHit] = []
    for step in input_item.steps:
        matched = [rule for rule in rules.get("rules", []) if _rule_matches(rule, step)]
        if not matched:
            continue
        # Multiple rules may describe the same step; use the strongest fixed
        # weight rather than amplifying long or repetitive error messages.
        strongest = max(matched, key=lambda rule: float(rule.get("weight", 0.0)))
        hits.append(
            ClueHit(
                step_ordinal=step.ordinal,
                rule_names=tuple(rule["name"] for rule in matched),
                category=strongest["category"],
                strength=float(strongest["weight"]),
                signature=_error_signature(step),
            )
        )
        clue_scores[step.ordinal] = float(strongest["weight"])

    tier = _tier(input_item, hits)
    # A repeated observation is one symptom for propagation purposes. Keep the
    # first occurrence, while preserving the number of raw occurrences for the
    # graph audit.
    seed_by_cluster: dict[str, ClueHit] = {}
    for hit in hits:
        if hit.category == "weak_termination":
            continue
        cluster = _hash_signature(hit.signature)
        if cluster not in seed_by_cluster or hit.strength > seed_by_cluster[cluster].strength:
            seed_by_cluster[cluster] = hit
    seeds = []
    for hit in sorted(seed_by_cluster.values(), key=lambda x: x.step_ordinal):
        step = input_item.steps[hit.step_ordinal]
        evidence = "terminal_failure" if hit.category == "terminal_failure" else "tool_failure" if hit.category == "tool_env_failure" else "negative_confession"
        seeds.append({"source_step_id": step.source_step_id, "strength": hit.strength, "evidence": evidence})
    # failed_occurrences counts raw hits, while failed_clusters reflects the
    # number of independent symptoms that can seed graph propagation.
    return EvidenceResult(
        tier=tier,
        status="ok" if tier != "C" else "no_evidence",
        clue_scores=clue_scores,
        hits=hits,
        symptom_seeds=seeds,
        failed_occurrences=len(hits),
        failed_clusters=len(seed_by_cluster),
    )
