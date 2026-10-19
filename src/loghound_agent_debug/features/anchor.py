"""Failure-anchor matching channel.

Anchors are code symbols extracted from the failing test's error description.
A step scores higher when its decision/action mentions more of those symbols.
Observation is intentionally excluded so verbose tool output cannot dominate the
channel the way it dominated the semantic document before Day 1.
"""

from __future__ import annotations

import math
import re

from loghound_agent_debug.schema import TrajectoryInput

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def scores(input_item: TrajectoryInput, anchors: list[str] | None = None) -> list[float]:
    """Return one anchor score per step, normalized by sqrt(|anchors|).

    The square-root denominator keeps trajectories with many anchors from being
    flattened out.  No within-trajectory min-max is applied: rank fusion only
    consumes the ordering, so the raw scale does not change the result.
    """
    anchor_set = {str(x) for x in (anchors if anchors is not None else input_item.failure_anchors) if x}
    if not anchor_set:
        return [0.0] * len(input_item.steps)
    denominator = math.sqrt(len(anchor_set))
    result: list[float] = []
    for step in input_item.steps:
        text = f"{step.decision_text} {step.action_text}"
        hits = len(anchor_set & set(_TOKEN.findall(text)))
        result.append(hits / denominator)
    return result
