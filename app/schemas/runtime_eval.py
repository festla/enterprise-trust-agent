from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.agent_runtime import (
    AgentIntent,
    StopReason,
)
from app.schemas.complex_plan_eval_result import (
    StepAction,
)


RuntimeEvalCategory = Literal[
    "financial_fact",
    "financial_calculation",
    "financial_comparison",
    "document_evidence",
    "clarification",
    "unsupported",
]

RuntimeEvalFinalStatus = Literal[
    "completed",
    "refused",
    "awaiting_human",
]

RuntimeEvalToolName = Literal[
    "query_financial_data",
    "retrieve_documents",
    "execute_calculation",
]


class RuntimeEvalCase(
    BaseModel
):
    """Week 6 Runtime Control Dev 单条评测题。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    case_id: str = Field(
        pattern=r"^runtime_[0-9]{3}$",
    )

    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    category: RuntimeEvalCategory

    expected_intent: AgentIntent

    expected_company_ids: tuple[
        str,
        ...
    ] = ()

    expected_years: tuple[
        int,
        ...
    ] = ()

    # 这里使用 AgentState.metric_ids 的语义：
    # reported + derived target 的稳定并集。
    expected_metric_ids: tuple[
        str,
        ...
    ] = ()

    expected_plan_actions: tuple[
        StepAction,
        ...
    ] = ()

    expected_tool_sequence: tuple[
        RuntimeEvalToolName,
        ...
    ] = ()

    expected_final_status: (
        RuntimeEvalFinalStatus
    )

    expected_stop_reason: StopReason

    replay_required: bool = True

    @model_validator(mode="after")
    def validate_case_contract(
        self,
    ) -> Self:
        if (
            self.expected_final_status
            == "completed"
        ):
            if (
                self.expected_stop_reason
                != "completed"
            ):
                raise ValueError(
                    "completed Case 的 "
                    "expected_stop_reason "
                    "必须是 completed"
                )

            if (
                self.expected_intent
                == "unsupported"
            ):
                raise ValueError(
                    "completed Case "
                    "不能是 unsupported"
                )

            if not self.expected_plan_actions:
                raise ValueError(
                    "completed Case "
                    "必须包含 Plan"
                )

            if not self.replay_required:
                raise ValueError(
                    "completed Case "
                    "必须要求 Trajectory Replay"
                )

        if (
            self.expected_final_status
            == "awaiting_human"
        ):
            if (
                self.expected_stop_reason
                != "human_review_required"
            ):
                raise ValueError(
                    "awaiting_human Case "
                    "必须使用 "
                    "human_review_required"
                )

            if (
                self.expected_plan_actions
                or self.expected_tool_sequence
            ):
                raise ValueError(
                    "awaiting_human Case "
                    "不能提前包含 Plan / Tool"
                )

            if self.replay_required:
                raise ValueError(
                    "未终止 awaiting_human Case "
                    "不要求最终 Trajectory"
                )

        if (
            self.expected_final_status
            == "refused"
            and self.expected_intent
            == "unsupported"
        ):
            if (
                self.expected_stop_reason
                != "unsupported"
            ):
                raise ValueError(
                    "unsupported Case "
                    "必须使用 unsupported "
                    "stop_reason"
                )

        return self


class RuntimeEvalCaseResult(
    BaseModel
):
    """一条 Runtime Eval 的实际结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    case_id: str

    question: str

    expected_intent: AgentIntent

    actual_intent: (
        AgentIntent | None
    ) = None

    expected_company_ids: tuple[
        str,
        ...
    ]

    actual_company_ids: tuple[
        str,
        ...
    ] = ()

    expected_years: tuple[
        int,
        ...
    ]

    actual_years: tuple[
        int,
        ...
    ] = ()

    expected_metric_ids: tuple[
        str,
        ...
    ]

    actual_metric_ids: tuple[
        str,
        ...
    ] = ()

    expected_plan_actions: tuple[
        StepAction,
        ...
    ]

    actual_plan_actions: tuple[
        StepAction,
        ...
    ] = ()

    expected_tool_sequence: tuple[
        RuntimeEvalToolName,
        ...
    ]

    actual_tool_sequence: tuple[
        str,
        ...
    ] = ()

    expected_final_status: (
        RuntimeEvalFinalStatus
    )

    actual_final_status: (
        str | None
    ) = None

    expected_stop_reason: StopReason

    actual_stop_reason: (
        str | None
    ) = None

    intent_ok: bool

    argument_ok: bool

    plan_ok: bool

    tool_ok: bool

    tool_sequence_ok: bool

    termination_ok: bool

    replay_ok: bool | None

    case_pass: bool

    error_message: str | None = None


class RuntimeEvalSummary(
    BaseModel
):
    """50-case Runtime Control Eval 汇总。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    eval_name: str = (
        "runtime_control_dev_v1"
    )

    case_count: int = Field(
        ge=1,
    )

    passed_count: int = Field(
        ge=0,
    )

    completed_count: int = Field(
        ge=0,
    )

    refused_count: int = Field(
        ge=0,
    )

    awaiting_human_count: int = Field(
        ge=0,
    )

    failed_count: int = Field(
        ge=0,
    )

    intent_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    argument_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    plan_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    tool_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    tool_sequence_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    termination_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    task_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    replay_applicable_count: int = Field(
        ge=0,
    )

    replay_success_count: int = Field(
        ge=0,
    )

    replay_success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )