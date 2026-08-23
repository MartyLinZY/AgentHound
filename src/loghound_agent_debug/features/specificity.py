"""Cross-trajectory template specificity, disabled for the main protocol."""

from __future__ import annotations

import math
import re
from collections import Counter

from loghound_agent_debug.schema import TrajectoryInput, Step


def template(step: Step) -> str:
    raw = " ".join(filter(None, [step.role, step.agent, step.tool_name or "", step.action_text]))
    raw = raw.lower()
    raw = re.sub(r"https?://\S+|/[^\s]+", "<path>", raw)
    raw = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", raw)
    raw = re.sub(r"\b\d+\b", "<n>", raw)
    raw = re.sub(r"[^a-z0-9_<>]+", " ", raw)
    stop = {"the", "and", "for", "from", "with", "to", "a", "an", "of", "is", "in"}
    return " ".join(token for token in raw.split() if token not in stop)


def corpus_statistics(inputs: list[TrajectoryInput]) -> dict:
    document_frequency = Counter()
    all_templates = []
    for item in inputs:
        current = {template(step) for step in item.steps}
        document_frequency.update(current)
        all_templates.extend(current)
    trajectory_count = len(inputs)
    singleton_count = sum(1 for value in document_frequency.values() if value == 1)
    total = len(document_frequency)
    scores = [math.log((trajectory_count + 1) / (frequency + 1)) for frequency in document_frequency.values()]
    mean = sum(scores) / len(scores) if scores else 0.0
    variance = sum((value - mean) ** 2 for value in scores) / len(scores) if scores else 0.0
    return {
        "trajectories": trajectory_count,
        "templates": total,
        "singleton_templates": singleton_count,
        "singleton_ratio": singleton_count / total if total else 0.0,
        "specificity_mean": mean,
        "specificity_variance": variance,
        "mode": "singleton",
    }


def scores_for(item: TrajectoryInput, inputs: list[TrajectoryInput] | None = None, mode: str = "singleton") -> list[float]:
    if mode == "singleton" or not inputs:
        return [1.0] * len(item.steps)
    stats = corpus_statistics(inputs)
    del stats
    df = Counter()
    for other in inputs:
        df.update({template(step) for step in other.steps})
    n = len(inputs)
    return [math.log((n + 1) / (df[template(step)] + 1)) for step in item.steps]
