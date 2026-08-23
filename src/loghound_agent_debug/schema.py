"""Canonical input and output objects.

The dataclasses deliberately keep labels out of :class:`TrajectoryInput`.  A
ranker receives only this object and therefore cannot accidentally consume the
ground-truth fields from the source parquet row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FailureObservation:
    source_step_id: str
    kind: str
    content: str
    provenance: str = "trace"


@dataclass(frozen=True)
class Step:
    ordinal: int
    source_step_id: str
    source_event_ids: list[str]
    agent: str
    role: str
    raw_content: str
    decision_text: str = ""
    action_text: str = ""
    observation_text: str = ""
    tool_name: str | None = None
    artifacts_read: list[str] = field(default_factory=list)
    artifacts_written: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrajectoryInput:
    schema_version: str
    dataset: str
    trajectory_id: str
    task: str
    domain: str
    outcome: str
    failure_observations: list[FailureObservation]
    steps: list[Step]


@dataclass(frozen=True)
class Label:
    trajectory_id: str
    root_cause_steps: list[str]
    root_cause_agents: list[str] = field(default_factory=list)
    root_cause_modules: list[str] = field(default_factory=list)
    root_cause_types: list[str] = field(default_factory=list)
    reason: str = ""
    annotation_method: str = ""
    provenance: str = ""


def _asdict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: _asdict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_asdict(v) for v in value]
    if isinstance(value, dict):
        return {k: _asdict(v) for k, v in value.items()}
    return value


def input_to_dict(item: TrajectoryInput) -> dict[str, Any]:
    return _asdict(item)


def label_to_dict(item: Label) -> dict[str, Any]:
    return _asdict(item)


def input_from_dict(data: dict[str, Any]) -> TrajectoryInput:
    observations = [FailureObservation(**x) for x in data.get("failure_observations", [])]
    steps = [Step(**x) for x in data["steps"]]
    return TrajectoryInput(
        schema_version=data.get("schema_version", "0.1.0"),
        dataset=data.get("dataset", ""),
        trajectory_id=str(data["trajectory_id"]),
        task=data.get("task", ""),
        domain=data.get("domain", "coding"),
        outcome=data.get("outcome", "failure"),
        failure_observations=observations,
        steps=steps,
    )


def label_from_dict(data: dict[str, Any]) -> Label:
    return Label(
        trajectory_id=str(data["trajectory_id"]),
        root_cause_steps=[str(x) for x in data.get("root_cause_steps", [])],
        root_cause_agents=[str(x) for x in data.get("root_cause_agents", [])],
        root_cause_modules=[str(x) for x in data.get("root_cause_modules", [])],
        root_cause_types=[str(x) for x in data.get("root_cause_types", [])],
        reason=data.get("reason", ""),
        annotation_method=data.get("annotation_method", ""),
        provenance=data.get("provenance", ""),
    )
