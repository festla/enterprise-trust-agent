from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexPlanOutput,
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.enums import StatementScope
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
    RetrievedDocument,
    ToolCallTrace,
    ToolExecutionResult,
)
from app.schemas.trust import (
    AnswerDraft,
)

AgentIntent = Literal[
    "financial_fact",
    "financial_calculation",
    "financial_comparison",
    "document_evidence",
    "unsupported",
]

SupportedAgentIntent = Literal[
    "financial_fact",
    "financial_calculation",
    "financial_comparison",
    "document_evidence",
]

AgentStatus = Literal[
    "created",
    "parsing",
    "routed",
    "planned",
    "executing",
    "verifying",
    "generating",
    "awaiting_human",
    "completed",
    "refused",
    "failed",
]

StopReason = Literal[
    "completed",
    "unsupported",
    "missing_required_fields",
    "insufficient_evidence",
    "tool_failure",
    "tool_timeout",
    "max_steps_exceeded",
    "calculation_failed",
    "human_review_required",
    "human_rejected",
    "incompatible_checkpoint",
    "internal_error",
]

NodeStatus = Literal[
    "completed",
    "failed",
    "interrupted",
]

RuntimeNode = Literal[
    "parse_query",
    "route_intent",
    "create_plan",
    "execute_plan",
    "verify_evidence",
    "prepare_answer",
    "generate_answer",
    "await_human",
    "handle_failure",
    "finish",
]

AnswerType = Literal[
    "financial",
    "document",
]

_ALLOWED_TOOL_NAMES = {
    "query_financial_data",
    "retrieve_documents",
    "execute_calculation",
}

_RUNTIME_ID_PATTERN = (
    r"^[a-zA-Z0-9_.:-]{1,160}$"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

_FACT_ID_PATTERN = (
    r"^fact_[a-z0-9_]+$"
)

_EVIDENCE_ID_PATTERN = (
    r"^evidence_[a-z0-9_]+$"
)

_CALCULATION_ID_PATTERN = (
    r"^calculation_[a-z0-9_]+$"
)

_STEP_ID_PATTERN = r"^s[1-9][0-9]*$"

_FORBIDDEN_SUMMARY_KEYS = {
    "api_key",
    "apikey",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
}


def _validate_unique_values(
    value: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    if len(value) != len(set(value)):
        raise ValueError(
            f"{field_name} 不能包含重复值"
        )

    return value

def _find_forbidden_summary_key(
    value: object,
    *,
    path: str,
) -> str | None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower()
            current_path = f"{path}.{key}"

            if normalized_key in (
                _FORBIDDEN_SUMMARY_KEYS
            ):
                return current_path

            nested_result = (
                _find_forbidden_summary_key(
                    child,
                    path=current_path,
                )
            )

            if nested_result is not None:
                return nested_result

    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            nested_result = (
                _find_forbidden_summary_key(
                    child,
                    path=f"{path}[{index}]",
                )
            )

            if nested_result is not None:
                return nested_result

    return None

# 对用户问题进行结构化解析
"""
用户问：
“比较美的集团 2024 年和 2025 年营业收入增长率”

解析为：
company_ids = ("midea_group",)
years = (2024, 2025)
metric_ids = ("revenue",)
calculation_metric_ids = ("revenue_growth_rate",)
comparison_requested = True
"""
class ParsedFinancialQuery(BaseModel):
    """生产 Runtime 的问题解析结果。

    该模型只保存从用户问题中解析出的业务身份，
    不允许包含任何 Gold Fact、Gold Page 或 Gold Answer。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    normalized_question: str = Field(
        min_length=1,
        max_length=4000,
    )

    company_ids: tuple[str, ...] = ()

    report_ids: tuple[str, ...] = ()

    years: tuple[int, ...] = ()

    metric_ids: tuple[str, ...] = ()

    calculation_metric_ids: tuple[
        str,
        ...,
    ] = ()

    statement_scope: (
        StatementScope | None
    ) = None

    comparison_requested: bool = False

    ranking_requested: bool = False

    explanation_requested: bool = False

    unsupported_reason: str | None = Field(
        default=None,
        max_length=1000,
    )

    missing_fields: tuple[str, ...] = ()

    assumptions: tuple[str, ...] = ()

    ambiguity_notes: tuple[str, ...] = ()

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "company_ids",
        "report_ids",
        "metric_ids",
        "calculation_metric_ids",
        "missing_fields",
        "assumptions",
        "ambiguity_notes",
    )
    @classmethod
    def validate_unique_strings(
        cls,
        value: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        return _validate_unique_values(
            value,
            field_name=info.field_name,
        )

    @field_validator("years")
    @classmethod
    def validate_years(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "years 不能包含重复年份"
            )

        if any(
            year < 2000 or year > 2100
            for year in value
        ):
            raise ValueError(
                "years 超出支持范围"
            )

        return value

    @model_validator(mode="after")
    def validate_query_contract(self) -> Self:
        if (
            self.unsupported_reason is not None
            and not self.unsupported_reason.strip()
        ):
            raise ValueError(
                "unsupported_reason 不能为空字符串"
            )

        return self


# 绑定查询、执行步骤和工具
"""
    Runtime Query
        ↓
    Plan Step
        ↓
        Tool

在 Agent 真正执行之前，提前发现计划与工具配置不一致的问题，
而不是运行到一半才报错。

例如：
    financial query q1
        ↓
    retrieve step s1
        ↓
    query_financial_data

    calculate step s2
        ↓
    execute_calculation
"""
class RuntimePlan(BaseModel):
    """Agent 使用的结构化计划及显示工具绑定。
    
    ComplexPlanOutput 负责业务步骤与拓扑校验；
    RuntimePlan 只增加查询集合和工具绑定。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    intent: SupportedAgentIntent

    planner_version: str = Field(
        min_length=1,
        max_length=128,
    )

    normalized_question: str = Field(
        min_length=1,
        max_length=4000,
    )

    financial_queries: tuple[
        ComplexRetrievalQueryOutput,
        ...
    ] = ()

    document_queries: tuple[
        DocumentEvidenceQuery,
        ...
    ] = ()

    plan: ComplexPlanOutput

    tool_by_step_id: dict[str, str]

    @model_validator(mode="after")
    def validate_plan_bindings(self) -> Self:
        """校验 Query、Plan Step 与工具绑定关系。"""

        financial_query_ids = {
            query.query_id
            for query in self.financial_queries
        }

        document_query_ids = {
            query.query_id
            for query in self.document_queries
        }

        overlapping_query_ids = (
            financial_query_ids
            & document_query_ids
        )

        if overlapping_query_ids:
            raise ValueError(
                "financial_queries 和 "
                "document_queries 不能使用相同 query_id"
            )

        if (
            self.intent == "document_evidence"
            and self.financial_queries
        ):
            raise ValueError(
                "document_evidence 计划不能包含 "
                "financial_queries"
            )

        if (
            self.intent != "document_evidence"
            and self.document_queries
        ):
            raise ValueError(
                "财务数值计划不能包含 "
                "document_queries"
            )

        all_query_ids = (
            financial_query_ids
            | document_query_ids
        )

        retrieve_steps = tuple(
            step
            for step in self.plan.steps
            if step.action == "retrieve"
        )

        query_id_by_step_id: dict[
            str,
            str,
        ] = {}

        plan_query_ids: set[str] = set()

        for step in retrieve_steps:
            step_values = step.model_dump()

            retrieval_query_id = (
                step_values.get(
                    "retrieval_query_id"
                )
            )

            if not isinstance(
                retrieval_query_id,
                str,
            ):
                raise ValueError(
                    f"{step.step_id} 是 retrieve 步骤，"
                    "但缺少合法的 retrieval_query_id"
                )

            if not retrieval_query_id:
                raise ValueError(
                    f"{step.step_id} 的 "
                    "retrieval_query_id 不能为空"
                )

            if (
                retrieval_query_id
                in plan_query_ids
            ):
                raise ValueError(
                    "多个 retrieve 步骤不能引用"
                    "同一个 retrieval_query_id："
                    f"{retrieval_query_id}"
                )

            plan_query_ids.add(
                retrieval_query_id
            )

            query_id_by_step_id[
                step.step_id
            ] = retrieval_query_id

        if plan_query_ids != all_query_ids:
            missing_query_ids = sorted(
                all_query_ids
                - plan_query_ids
            )

            unknown_query_ids = sorted(
                plan_query_ids
                - all_query_ids
            )

            raise ValueError(
                "Runtime Query 与 Plan retrieve "
                "步骤必须一一对应："
                f"missing={missing_query_ids}, "
                f"unknown={unknown_query_ids}"
            )

        plan_step_ids = {
            step.step_id
            for step in self.plan.steps
        }

        unknown_binding_step_ids = (
            set(self.tool_by_step_id)
            - plan_step_ids
        )

        if unknown_binding_step_ids:
            raise ValueError(
                "tool_by_step_id 引用了未知步骤："
                f"{sorted(unknown_binding_step_ids)}"
            )

        required_tool_step_ids = {
            step.step_id
            for step in self.plan.steps
            if step.action in {
                "retrieve",
                "calculate",
                "normalize_unit",
            }
        }

        actual_tool_step_ids = set(
            self.tool_by_step_id
        )

        if (
            actual_tool_step_ids
            != required_tool_step_ids
        ):
            missing_step_ids = sorted(
                required_tool_step_ids
                - actual_tool_step_ids
            )

            unexpected_step_ids = sorted(
                actual_tool_step_ids
                - required_tool_step_ids
            )

            raise ValueError(
                "所有工具步骤必须且只能绑定一个工具："
                f"missing={missing_step_ids}, "
                f"unexpected={unexpected_step_ids}"
            )

        for step_id, tool_name in (
            self.tool_by_step_id.items()
        ):
            if tool_name not in (
                _ALLOWED_TOOL_NAMES
            ):
                raise ValueError(
                    f"{step_id} 绑定了未允许的工具："
                    f"{tool_name}"
                )

        query_type_by_id = {
            **{
                query_id: "financial"
                for query_id
                in financial_query_ids
            },
            **{
                query_id: "document"
                for query_id
                in document_query_ids
            },
        }

        for step in retrieve_steps:
            retrieval_query_id = (
                query_id_by_step_id[
                    step.step_id
                ]
            )

            query_type = (
                query_type_by_id[
                    retrieval_query_id
                ]
            )

            expected_tool_name = (
                "query_financial_data"
                if query_type == "financial"
                else "retrieve_documents"
            )

            actual_tool_name = (
                self.tool_by_step_id[
                    step.step_id
                ]
            )

            if (
                actual_tool_name
                != expected_tool_name
            ):
                raise ValueError(
                    f"{step.step_id} 的查询类型"
                    "与工具绑定不一致："
                    f"expected={expected_tool_name}, "
                    f"actual={actual_tool_name}"
                )

        for step in self.plan.steps:
            if step.action not in {
                "calculate",
                "normalize_unit",
            }:
                continue

            actual_tool_name = (
                self.tool_by_step_id[
                    step.step_id
                ]
            )

            if (
                actual_tool_name
                != "execute_calculation"
            ):
                raise ValueError(
                    "计算步骤必须绑定 "
                    "execute_calculation："
                    f"step_id={step.step_id}, "
                    f"actual={actual_tool_name}"
                )

        return self


class AgentErrorRecord(BaseModel):
    """一次 Runtime 错误或恢复事件。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    stage: RuntimeNode

    error_type: str = Field(
        min_length=1,
        max_length=256,
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    retryable: bool = False

    step_id: str | None = Field(
        default=None,
        pattern=_STEP_ID_PATTERN,
    )

    tool_call_id: str | None = Field(
        default=None,
        pattern=_RUNTIME_ID_PATTERN,
    )

    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value


class CitationRecord(BaseModel):
    """最终答案中的一个可回放引用。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    citation_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    evidence_id: str | None = Field(
        default=None,
        pattern=_EVIDENCE_ID_PATTERN,
    )

    chunk_id: str | None = Field(
        default=None,
        min_length=1,
    )

    text_excerpt: str = Field(
        min_length=1,
        max_length=1000,
    )

    @model_validator(mode="after")
    def validate_source_reference(self) -> Self:
        if (
            self.evidence_id is None
            and self.chunk_id is None
        ):
            raise ValueError(
                "CitationRecord 至少需要 "
                "evidence_id 或 chunk_id"
            )

        return self

# 规范最终答案与引用证据

class AgentAnswer(BaseModel):
    """统一承载财务答案或文档证据答案。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    answer_type: AnswerType

    answer_text: str = Field(
        min_length=1,
        max_length=20_000,
    )

    supporting_fact_ids: tuple[
        str,
        ...,
    ] = ()

    supporting_calculation_ids: tuple[
        str,
        ...,
    ] = ()

    citation_evidence_ids: tuple[
        str,
        ...,
    ] = ()

    document_citation_ids: tuple[
        str,
        ...,
    ] = ()

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "supporting_fact_ids",
        "supporting_calculation_ids",
        "citation_evidence_ids",
        "document_citation_ids",
    )
    @classmethod
    def validate_unique_ids(
        cls,
        value: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _validate_unique_values(
            value,
            field_name=info.field_name,
        )

    @model_validator(mode="after")
    def validate_answer_contract(self) -> Self:
        if self.answer_type == "financial":
            if not self.supporting_fact_ids:
                raise ValueError(
                    "financial answer 必须包含 "
                    "supporting_fact_ids"
                )

            if not self.citation_evidence_ids:
                raise ValueError(
                    "financial answer 必须包含 "
                    "citation_evidence_ids"
                )

            if self.document_citation_ids:
                raise ValueError(
                    "financial answer 不能包含 "
                    "document_citation_ids"
                )

        if self.answer_type == "document":
            if not self.document_citation_ids:
                raise ValueError(
                    "document answer 必须包含 "
                    "document_citation_ids"
                )

            if (
                self.supporting_fact_ids
                or self.supporting_calculation_ids
                or self.citation_evidence_ids
            ):
                raise ValueError(
                    "document answer 不能伪装成 "
                    "结构化财务答案"
                )

        return self


# 记录每一个 Runtime 或 LangGraph 节点的执行信息
"""
节点名称；
第几次尝试；
成功、失败还是中断；
输入和输出摘要；
开始和结束时间；
执行耗时；
Checkpoint 版本；
错误类型和错误信息。
"""
class NodeSpan(BaseModel):
    """一次 Graph/Runtime Node 执行记录。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    span_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    node_name: RuntimeNode

    attempt: int = Field(
        default=1,
        ge=1,
    )

    status: NodeStatus

    input_summary: dict[str, Any]

    output_summary: dict[str, Any]

    started_at: datetime

    completed_at: datetime

    latency_ms: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    checkpoint_revision: int = Field(
        default=0,
        ge=0,
    )

    error_type: str | None = Field(
        default=None,
        max_length=256,
    )

    error_message: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @field_validator(
        "input_summary",
        "output_summary",
    )
    @classmethod
    def validate_summaries(
        cls,
        value: dict[str, Any],
        info: Any,
    ) -> dict[str, Any]:
        forbidden_path = (
            _find_forbidden_summary_key(
                value,
                path=info.field_name,
            )
        )

        if forbidden_path is not None:
            raise ValueError(
                "Node Span 摘要不能记录敏感字段："
                f"{forbidden_path}"
            )

        return value

    @model_validator(mode="after")
    def validate_span_contract(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at 不能早于 started_at"
            )

        if self.status == "completed":
            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "成功 Node Span 不能包含错误字段"
                )

        elif (
            not self.error_type
            or not self.error_message
        ):
            raise ValueError(
                "失败或中断 Node Span "
                "必须包含错误信息"
            )

        return self


class HumanReviewDecision(BaseModel):
    """Week 6 基础人工确认结果。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    approved: bool

    corrected_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
    )

    reason: str = Field(
        min_length=1,
        max_length=2000,
    )

    reviewer_id: str = Field(
        min_length=1,
        max_length=256,
    )

    decided_at: datetime

    @field_validator("decided_at")
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value


# 这部分是这份代码的关键：统一保存一次Agent运行中的所有信息
class AgentState(BaseModel):
    """可保存、可恢复的严格 Agent 状态。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    request_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    trace_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    thread_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    # 用户的原始问题
    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    parsed_query: (
        ParsedFinancialQuery | None
    ) = None

    intent: AgentIntent | None = None

    company_ids: tuple[str, ...] = ()

    report_ids: tuple[str, ...] = ()

    years: tuple[int, ...] = ()

    metric_ids: tuple[str, ...] = ()

    runtime_plan: RuntimePlan | None = None

    current_step: int = Field(
        default=0,
        ge=0,
    )

    runtime_refs: dict[
        str,
        tuple[str, ...],
    ] = Field(
        default_factory=dict,
    )

    completed_step_ids: tuple[
        str,
        ...,
    ] = ()

    tool_results: tuple[
        ToolExecutionResult,
        ...,
    ] = ()

    node_spans: tuple[
        NodeSpan,
        ...,
    ] = ()

    tool_call_traces: tuple[
        ToolCallTrace,
        ...,
    ] = ()

    retrieval_traces: tuple[
        ComplexRetrievalTrace,
        ...,
    ] = ()

    calculation_traces: tuple[
        ComplexCalculationTrace,
        ...,
    ] = ()

    retrieved_documents: tuple[
        RetrievedDocument,
        ...,
    ] = ()

    resolved_fact_ids: tuple[
        str,
        ...,
    ] = ()

    evidence_ids: tuple[
        str,
        ...,
    ] = ()

    calculation_ids: tuple[
        str,
        ...,
    ] = ()

    answer_draft: AnswerDraft | None = None

    answer: AgentAnswer | None = None

    citations: tuple[
        CitationRecord,
        ...,
    ] = ()

    status: AgentStatus = "created"

    retry_count: int = Field(
        default=0,
        ge=0,
    )

    max_steps: int = Field(
        default=32,
        ge=1,
        le=500,
    )

    step_count: int = Field(
        default=0,
        ge=0,
    )

    errors: tuple[
        AgentErrorRecord,
        ...,
    ] = ()

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    stop_reason: StopReason | None = None

    pending_human_review: bool = False

    human_review_reason: str | None = Field(
        default=None,
        max_length=2000,
    )

    human_decision: (
        HumanReviewDecision | None
    ) = None

    current_node: RuntimeNode = (
        "parse_query"
    )

    next_node: RuntimeNode = (
        "parse_query"
    )

    checkpoint_revision: int = Field(
        default=0,
        ge=0,
    )

    planner_version: str | None = Field(
        default=None,
        max_length=128,
    )

    retriever_version: str | None = Field(
        default=None,
        max_length=256,
    )

    calculator_version: str | None = Field(
        default=None,
        max_length=128,
    )

    generator_version: str | None = Field(
        default=None,
        max_length=128,
    )

    prompt_version: str | None = Field(
        default=None,
        max_length=128,
    )

    prompt_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )

    model_name: str | None = Field(
        default=None,
        max_length=256,
    )

    input_tokens: int = Field(
        default=0,
        ge=0,
    )

    output_tokens: int = Field(
        default=0,
        ge=0,
    )

    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    started_at: datetime

    updated_at: datetime

    completed_at: datetime | None = None

    @field_validator(
        "started_at",
        "updated_at",
        "completed_at",
    )
    @classmethod
    def validate_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return value

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @field_validator(
        "company_ids",
        "report_ids",
        "metric_ids",
        "completed_step_ids",
        "resolved_fact_ids",
        "evidence_ids",
        "calculation_ids",
    )
    @classmethod
    def validate_unique_ids(
        cls,
        value: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _validate_unique_values(
            value,
            field_name=info.field_name,
        )

    @field_validator("years")
    @classmethod
    def validate_state_years(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "years 不能包含重复年份"
            )

        if any(
            year < 2000 or year > 2100
            for year in value
        ):
            raise ValueError(
                "years 超出支持范围"
            )

        return value

    @field_validator("runtime_refs")
    @classmethod
    def validate_runtime_refs(
        cls,
        value: dict[
            str,
            tuple[str, ...],
        ],
    ) -> dict[str, tuple[str, ...]]:
        for output_ref, resolved_ids in (
            value.items()
        ):
            if not output_ref:
                raise ValueError(
                    "runtime_refs 的 key 不能为空"
                )

            if not resolved_ids:
                raise ValueError(
                    "runtime_refs 的 value 不能为空"
                )

            if len(resolved_ids) != len(
                set(resolved_ids)
            ):
                raise ValueError(
                    "runtime_refs 中不能包含重复 ID"
                )

        return value

    @model_validator(mode="after")
    def validate_state_contract(self) -> Self:
        self._validate_time_contract()
        self._validate_step_contract()
        self._validate_status_contract()
        self._validate_plan_contract()
        self._validate_trace_identity()
        self._validate_answer_contract()

        return self

    def _validate_time_contract(self) -> None:
        if self.updated_at < self.started_at:
            raise ValueError(
                "updated_at 不能早于 started_at"
            )

        if (
            self.completed_at is not None
            and self.completed_at
            < self.started_at
        ):
            raise ValueError(
                "completed_at 不能早于 started_at"
            )

    def _validate_step_contract(self) -> None:
        if self.step_count > self.max_steps:
            raise ValueError(
                "step_count 不能大于 max_steps"
            )

        if (
            self.runtime_plan is not None
            and self.current_step
            > len(self.runtime_plan.plan.steps)
        ):
            raise ValueError(
                "current_step 超出 Plan 步骤数量"
            )

    def _validate_status_contract(self) -> None:
        terminal_statuses = {
            "completed",
            "refused",
            "failed",
        }

        if self.status in terminal_statuses:
            if self.completed_at is None:
                raise ValueError(
                    "终止状态必须填写 completed_at"
                )

            if self.stop_reason is None:
                raise ValueError(
                    "终止状态必须填写 stop_reason"
                )

        elif self.status == "awaiting_human":
            if (
                self.stop_reason
                != "human_review_required"
            ):
                raise ValueError(
                    "awaiting_human 的 stop_reason "
                    "必须是 human_review_required"
                )

            if self.completed_at is not None:
                raise ValueError(
                    "awaiting_human 不能填写 "
                    "completed_at"
                )

        else:
            if self.completed_at is not None:
                raise ValueError(
                    "非终止状态不能填写 completed_at"
                )

            if self.stop_reason is not None:
                raise ValueError(
                    "运行中状态不能提前填写 "
                    "stop_reason"
                )

        if (
            self.pending_human_review
            != (
                self.status
                == "awaiting_human"
            )
        ):
            raise ValueError(
                "pending_human_review 必须与 "
                "awaiting_human 状态一致"
            )

        if self.status == "awaiting_human":
            if not self.human_review_reason:
                raise ValueError(
                    "等待人工确认时必须填写 "
                    "human_review_reason"
                )

        elif self.human_review_reason is not None:
            raise ValueError(
                "非 awaiting_human 状态不能填写 "
                "human_review_reason"
            )

        if self.status == "completed":
            if self.stop_reason != "completed":
                raise ValueError(
                    "completed 状态的 stop_reason "
                    "必须为 completed"
                )

            if self.answer is None:
                raise ValueError(
                    "completed 状态必须包含 answer"
                )

        if self.status in {
            "refused",
            "failed",
        }:
            if self.answer is not None:
                raise ValueError(
                    "refused 或 failed 状态 "
                    "不能包含 answer"
                )

    def _validate_plan_contract(self) -> None:
        if (
            self.runtime_plan is None
            and self.completed_step_ids
        ):
            raise ValueError(
                "没有 runtime_plan 时不能填写 "
                "completed_step_ids"
            )

        if self.runtime_plan is None:
            return

        plan_step_ids = {
            step.step_id
            for step
            in self.runtime_plan.plan.steps
        }

        unknown_completed_step_ids = (
            set(self.completed_step_ids)
            - plan_step_ids
        )

        if unknown_completed_step_ids:
            raise ValueError(
                "completed_step_ids 包含 "
                "Plan 中不存在的步骤："
                f"{sorted(unknown_completed_step_ids)}"
            )

    def _validate_trace_identity(self) -> None:
        tool_call_ids = [
            trace.tool_call_id
            for trace in self.tool_call_traces
        ]

        if len(tool_call_ids) != len(
            set(tool_call_ids)
        ):
            raise ValueError(
                "tool_call_traces 中的 "
                "tool_call_id 必须唯一"
            )

        query_ids = [
            trace.query_id
            for trace in self.retrieval_traces
        ]

        if len(query_ids) != len(
            set(query_ids)
        ):
            raise ValueError(
                "retrieval_traces 中的 "
                "query_id 必须唯一"
            )

        trace_calculation_ids = [
            trace.calculation_id
            for trace in self.calculation_traces
        ]

        if len(trace_calculation_ids) != len(
            set(trace_calculation_ids)
        ):
            raise ValueError(
                "calculation_traces 中的 "
                "calculation_id 必须唯一"
            )

        citation_ids = [
            citation.citation_id
            for citation in self.citations
        ]

        if len(citation_ids) != len(
            set(citation_ids)
        ):
            raise ValueError(
                "citations 中的 citation_id "
                "必须唯一"
            )

    def _validate_answer_contract(self) -> None:
        if self.answer is None:
            return

        if (
            tuple(self.answer.supporting_fact_ids)
            != self.resolved_fact_ids
        ):
            raise ValueError(
                "answer.supporting_fact_ids "
                "必须与 resolved_fact_ids 一致"
            )

        if (
            tuple(
                self.answer
                .supporting_calculation_ids
            )
            != self.calculation_ids
        ):
            raise ValueError(
                "answer 的 calculation 引用 "
                "必须与 calculation_ids 一致"
            )

        if (
            tuple(
                self.answer
                .citation_evidence_ids
            )
            != self.evidence_ids
        ):
            raise ValueError(
                "answer 的 Evidence 引用 "
                "必须与 evidence_ids 一致"
            )


class AgentTrajectory(BaseModel):
    """一次终止运行的完整、冻结轨迹。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    request_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    trace_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    thread_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    intent: AgentIntent | None = None

    planner_version: str | None = None

    retriever_version: str | None = None

    calculator_version: str | None = None

    generator_version: str | None = None

    prompt_version: str | None = None

    prompt_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )

    model_name: str | None = None

    parsed_query: (
        ParsedFinancialQuery | None
    ) = None

    runtime_plan: RuntimePlan | None = None

    node_spans: tuple[
        NodeSpan,
        ...,
    ] = ()

    tool_call_traces: tuple[
        ToolCallTrace,
        ...,
    ] = ()

    retrieval_traces: tuple[
        ComplexRetrievalTrace,
        ...,
    ] = ()

    calculation_traces: tuple[
        ComplexCalculationTrace,
        ...,
    ] = ()

    retrieved_documents: tuple[
        RetrievedDocument,
        ...,
    ] = ()

    resolved_fact_ids: tuple[
        str,
        ...,
    ] = ()

    evidence_ids: tuple[
        str,
        ...,
    ] = ()

    calculation_ids: tuple[
        str,
        ...,
    ] = ()

    citations: tuple[
        CitationRecord,
        ...,
    ] = ()

    answer_draft: AnswerDraft | None = None

    errors: tuple[
        AgentErrorRecord,
        ...,
    ] = ()

    answer: AgentAnswer | None = None

    input_tokens: int = Field(
        default=0,
        ge=0,
    )

    output_tokens: int = Field(
        default=0,
        ge=0,
    )

    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )

    started_at: datetime

    completed_at: datetime

    latency_ms: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    final_status: Literal[
        "completed",
        "refused",
        "failed",
    ]

    stop_reason: StopReason

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "datetime 必须包含时区信息"
            )

        return value

    @field_validator(
        "resolved_fact_ids",
        "evidence_ids",
        "calculation_ids",
    )
    @classmethod
    def validate_unique_ids(
        cls,
        value: tuple[str, ...],
        info: Any,
    ) -> tuple[str, ...]:
        return _validate_unique_values(
            value,
            field_name=info.field_name,
        )

    @model_validator(mode="after")
    def validate_trajectory_contract(
        self,
    ) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at 不能早于 started_at"
            )

        if self.final_status == "completed":
            if self.answer is None:
                raise ValueError(
                    "completed trajectory "
                    "必须包含 answer"
                )

            if self.stop_reason != "completed":
                raise ValueError(
                    "completed trajectory 的 "
                    "stop_reason 必须为 completed"
                )

        else:
            if self.answer is not None:
                raise ValueError(
                    "refused 或 failed trajectory "
                    "不能包含 answer"
                )

        return self


class TrajectoryReplay(BaseModel):
    """从 AgentTrajectory 中提取的回放摘要。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    nodes: tuple[str, ...]

    tools: tuple[str, ...]

    tool_arguments: tuple[
        dict[str, Any],
        ...,
    ]

    retries: tuple[str, ...]

    failures: tuple[str, ...]

    supporting_fact_ids: tuple[
        str,
        ...,
    ]

    evidence_ids: tuple[
        str,
        ...,
    ]

    calculation_ids: tuple[
        str,
        ...,
    ]

    final_status: Literal[
        "completed",
        "refused",
        "failed",
    ]

    stop_reason: StopReason