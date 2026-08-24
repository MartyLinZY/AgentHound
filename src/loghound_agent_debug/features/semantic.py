"""Small, deterministic TF-IDF channel with no model or cross-trajectory state."""

from __future__ import annotations

import math
import re
from collections import Counter

from loghound_agent_debug.schema import Step, TrajectoryInput

_TOKEN_RE = re.compile(r"<SOLUTION_FILE>|[a-z][a-z0-9_+-]*", re.I)
_STOP = {
    "the", "and", "for", "from", "with", "that", "this", "your", "are", "was", "were",
    "you", "can", "use", "will", "into", "then", "not", "have", "has", "had", "task",
    "step", "file", "please", "following", "current", "plan", "message", "user", "agent",
}


def semantic_document(step: Step) -> str:
    """Build the step document while removing repeated orchestration scaffolding."""
    decision = step.decision_text
    # The history repeats the whole task in the orchestration scaffold. It is
    # retained in raw_content, but must not become a document-level shortcut.
    decision = re.sub(r"\bQuestion:\s+.*?(?=\bNo Plan\b|# Current Task|# Response Language|\Z)", " ", decision, flags=re.I | re.S)
    decision = re.sub(r"\bTask ID:\s*\S+", " ", decision, flags=re.I)
    return " ".join(filter(None, [step.agent, step.role, decision, step.action_text, step.observation_text, *step.artifacts_read, *step.artifacts_written]))


def _tokens(text: str) -> list[str]:
    """Tokenize normalized text and remove generic workflow words."""
    text = text.lower().replace("<solution_file>", " solution_file ")
    return [token for token in _TOKEN_RE.findall(text) if token not in _STOP]


def _cosine(left: Counter[str], right: Counter[str], idf: dict[str, float]) -> float:
    """Compute TF-IDF cosine similarity for two sparse token counters."""
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    lv = {key: left[key] * idf.get(key, 1.0) for key in keys}
    rv = {key: right[key] * idf.get(key, 1.0) for key in keys}
    denom = math.sqrt(sum(x * x for x in lv.values())) * math.sqrt(sum(x * x for x in rv.values()))
    return sum(lv[key] * rv[key] for key in keys) / denom if denom else 0.0


def scores(input_item: TrajectoryInput, failure_text: str, task_weight: float = 0.4) -> list[float]:
    """Score each step against task intent and observed failure text."""
    documents = [_tokens(semantic_document(step)) for step in input_item.steps]
    task_tokens = _tokens(input_item.task)
    failure_tokens = _tokens(failure_text)
    all_docs = documents + [task_tokens, failure_tokens]
    # IDF is intentionally built within one trajectory to avoid cross-sample
    # state leaking into the main ranking protocol.
    document_frequency = Counter(token for doc in all_docs if doc for token in set(doc))
    total = len(all_docs)
    idf = {token: math.log((total + 1) / (frequency + 1)) + 1.0 for token, frequency in document_frequency.items()}
    task_vector = Counter(task_tokens)
    failure_vector = Counter(failure_tokens)
    result = []
    for document in documents:
        vector = Counter(document)
        task_score = _cosine(vector, task_vector, idf)
        failure_score = _cosine(vector, failure_vector, idf)
        # A step can be relevant to the task without being close to the error,
        # hence the two query components remain separately weighted.
        result.append(task_weight * task_score + (1.0 - task_weight) * failure_score)
    return result


def minmax(values: list[float]) -> list[float]:
    """Normalize a channel within one trajectory; constant channels become 0."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]
