from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


CompetitionAgentStage = Literal[
    "source_resolution",
    "retrieval",
    "evidence_assembly",
    "excel_solver",
    "readiness",
    "sufficiency",
    "answer",
    "citation",
]


CompetitionAgentTraceOutcome = Literal[
    "success",
    "business_blocked",
    "system_failed",
]


CompetitionAgentFailureCode = Literal[
    "provider_request_error",
    "provider_response_error",
    "validation_error",
    "citation_binding_error",
]


class CompetitionAgentTraceEvent(
    BaseModel
):
    """
    Agent 一次节点执行事件。

    暂时不记录 wall-clock timestamp，
    保持测试与评测结果确定性。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    stage: CompetitionAgentStage

    outcome: CompetitionAgentTraceOutcome

    message: str = Field(
        min_length=1,
    )

    details: dict[
        str,
        str | int | bool | None,
    ] = Field(
        default_factory=dict,
    )


class CompetitionAgentFailure(
    BaseModel
):
    """
    系统执行失败。

    注意：
    它不是业务拒答。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    stage: CompetitionAgentStage

    failure_code: (
        CompetitionAgentFailureCode
    )

    retryable: bool

    error_type: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )