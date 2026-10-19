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
    """A failure-related observation that is visible to the ranker."""
    source_step_id: str
    kind: str
    content: str
    provenance: str = "trace"


@dataclass(frozen=True)
class Step:
    """Canonical candidate unit: one agent decision and its attached effects.
    轨迹内序号（0,1,2,…）、规范化轨迹在主流程中的主键、
    构成这个 step 的原始事件 ID 集合（语义上是"一对多"——一个规范化 step 可能由多个底层事件（event）聚合而来，只是保留，基本不用）、执行该步的 agent 名、角色（默认 "assistant"）、原始内容原文、清洗后的推理/决策文本、实际执行的动作、观察到的输出、首个命令的工具名、该步读的文件类构件、该步写的构件
    """
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
    # RootSE keeps the original four fields separate for auditable LLM quotes.
    rootse_fields: dict[str, str] = field(default_factory=dict)
    working_dir: str = ""


@dataclass(frozen=True)
class TrajectoryInput:
    """Label-free representation consumed by ranking and feature modules.
    无标签的轨迹表示 --ranker和特征输入模块的唯一输入。
    数据模式的版本号、数据来源表示、轨迹的唯一id、任务描述（原始的question文本）、任务领域、轨迹执行结果、
    观察到的显示失败列表、该轨迹的执行步骤列表
    """
    schema_version: str
    dataset: str
    trajectory_id: str
    task: str
    domain: str
    outcome: str
    failure_observations: list[FailureObservation]
    steps: list[Step]
    # Failure-test symbols (file/class/function/assertion names).  Derived from
    # failed_tests.error_description only; never from failure_id/failure_reason.
    failure_anchors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Label:
    """Annotation-only root-cause information used by evaluation/reporting."""
    trajectory_id: str
    root_cause_steps: list[str]
    root_cause_agents: list[str] = field(default_factory=list)
    root_cause_modules: list[str] = field(default_factory=list)
    root_cause_types: list[str] = field(default_factory=list)
    reason: str = ""
    annotation_method: str = ""
    provenance: str = ""


def _asdict(value: Any) -> Any:
    """Recursively convert dataclasses and nested containers to JSON values."""
    if hasattr(value, "__dataclass_fields__"):
        return {k: _asdict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_asdict(v) for v in value]
    if isinstance(value, dict):
        return {k: _asdict(v) for k, v in value.items()}
    return value


def input_to_dict(item: TrajectoryInput) -> dict[str, Any]:
    """Serialize a canonical input without adding label fields."""
    return _asdict(item)


def label_to_dict(item: Label) -> dict[str, Any]:
    """Serialize an annotation record for the physically separate labels file."""
    return _asdict(item)


def input_from_dict(data: dict[str, Any]) -> TrajectoryInput:
    """Rebuild a canonical input, applying defaults for optional metadata."""
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
        failure_anchors=[str(x) for x in data.get("failure_anchors", [])],
    )


def label_from_dict(data: dict[str, Any]) -> Label:
    """Rebuild a label while normalizing all step and agent IDs to strings."""
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
