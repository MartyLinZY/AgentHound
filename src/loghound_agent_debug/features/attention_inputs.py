"""Deterministic, label-free inputs for the small attention ranker."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Sequence

import numpy as np

from loghound_agent_debug.features.semantic import semantic_document
from loghound_agent_debug.schema import TrajectoryInput

HASH_DIM = 128
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=")


def text_vector(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Signed log-TF hashing with no vocabulary fitted on other trajectories."""
    if dim < 1:
        raise ValueError("dim must be positive")
    tokens = TOKEN.findall(text.lower())
    counts = Counter(tokens)
    counts.update(a + "::" + b for a, b in zip(tokens, tokens[1:]))
    result = np.zeros(dim, dtype=np.float32)
    for token, count in sorted(counts.items()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign * (1.0 + math.log(count))
    norm = np.linalg.norm(result)
    return result / norm if norm else result


@dataclass(frozen=True)
class AttentionInputs:
    trajectory_id: str
    step_ids: tuple[str, ...]
    x: np.ndarray
    failure: np.ndarray
    prior: np.ndarray  # [later target, earlier source]


def build_inputs(
    item: TrajectoryInput,
    pairs: Sequence[tuple[int, int]],
    numeric: np.ndarray,
) -> AttentionInputs:
    """numeric contains label-free semantic/anchor/clue/q/missing features."""
    n = len(item.steps)
    if n == 0 or [s.ordinal for s in item.steps] != list(range(n)):
        raise ValueError("Steps must be nonempty with contiguous ordinals")
    ids = tuple(s.source_step_id for s in item.steps)
    if len(set(ids)) != n:
        raise ValueError("Duplicate step IDs")
    numeric = np.asarray(numeric, dtype=np.float32)
    if numeric.ndim != 2 or numeric.shape[0] != n or not np.isfinite(numeric).all():
        raise ValueError("Invalid numeric features")
    # Postmortem test fields are admitted by TrajectoryInput's protocol.
    failure_text = "\n".join(
        [f"{t.code}\n{t.error_description}" for t in item.diagnostic_tests]
        + [o.content for o in item.failure_observations]
    )
    failure = text_vector(failure_text)
    task = text_vector(item.task)
    rows = []
    for i, step in enumerate(item.steps):
        decision = semantic_document(step, include_observation=False)
        d = text_vector(decision)
        o = text_vector(step.observation_text)
        position = np.array([
            i / max(n - 1, 1), 1 - i / max(n - 1, 1),
            math.log1p(len(decision)) / 15,
            math.log1p(len(step.observation_text)) / 15,
        ], dtype=np.float32)
        rows.append(np.concatenate([d, o, d * failure, d * task, numeric[i], position]))
    prior = np.zeros((n, n), dtype=np.float32)
    for source, target in pairs:
        if not 0 <= source < target < n:
            raise ValueError("Prior must contain forward temporal dependencies")
        prior[target, source] = 1.0
    return AttentionInputs(item.trajectory_id, ids, np.stack(rows),
                           np.concatenate([task, failure]), prior)


def grouped_folds(items: Sequence[TrajectoryInput], n_folds: int = 5) -> tuple[list[int], list[str]]:
    """Union same repository OR task before balancing label-free groups."""
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for i, item in enumerate(items):
        if not item.repository:
            raise ValueError("Repository metadata required for grouped evaluation")
        for kind, value in [("repo", item.repository), ("task", item.task_id)]:
            if not value:
                continue
            key = kind, value
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    if n_folds < 2 or len(groups) < n_folds:
        raise ValueError("Not enough independent groups")
    keys = {r: "|".join(sorted({items[i].repository for i in ids})) for r, ids in groups.items()}
    ordered = sorted(groups, key=lambda r: (-len(groups[r]), hashlib.sha256(keys[r].encode()).hexdigest()))
    loads = [0] * n_folds
    folds = [-1] * len(items)
    group_ids = [""] * len(items)
    for r in ordered:
        fold = min(range(n_folds), key=lambda f: (loads[f], f))
        for i in groups[r]:
            folds[i] = fold
            group_ids[i] = keys[r]
        loads[fold] += len(groups[r])
    return folds, group_ids
