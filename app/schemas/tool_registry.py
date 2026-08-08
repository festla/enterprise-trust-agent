from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexRetrievalQueryOutput,
    ComplexRetrievalTrace,
)
from app.schemas.enums import ReportType


ToolPermission = Literal[
    "read_financial_data",
    "read_documents",
    "execute_calculation",
]

ToolCallStatus = Literal[
    "succeeded",
    "retryable_error",
    "permanent_error",
    "timed_out",
    "reused",
]

_TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{2,63}$"
_VERSION_PATTERN = r"^[a-zA-Z0-9_.-]{1,64}$"
_RUNTIME_ID_PATTERN = r"^[a-zA-Z0-9_.:-]{1,160}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_QUERY_ID_PATTERN = r"^q[1-9][0-9]*$"
_FACT_ID_PATTERN = r"^fact_[a-z0-9_]+$"
_CALCULATION_ID_PATTERN = (
    r"^calculation_[a-z0-9_]+$"
)

_FORBIDDEN_LOG_KEYS = {
    "api_key",
    "apikey",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
}

def _find_forbidden_log_key(
    value: object,
    *,
    path: str = "arguments",
) -> str | None:
    """递归检查参数摘要中是否存在敏感字段名。"""

    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized_key = key.strip().lower()

            current_path = (
                f"{path}.{key}"
            )

            if (
                normalized_key
                in _FORBIDDEN_LOG_KEYS
            ):
                return current_path

            child_result = (
                _find_forbidden_log_key(
                    child,
                    path=current_path,
                )
            )

            if child_result is not None:
                return child_result

    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_result = (
                _find_forbidden_log_key(
                    child,
                    path=f"{path}[{index}]",
                )
            )

            if child_result is not None:
                return child_result

    return None

class ToolDefinition(BaseModel):
    """工具注册表中的不可变工具契约。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    tool_name: str = Field(
        pattern=_TOOL_NAME_PATTERN,
    )

    description: str = Field(
        min_length=1,
        max_length=1000,
    )

    version: str = Field(
        pattern=_VERSION_PATTERN,
    )

    input_schema: dict[str, Any]

    output_schema: dict[str, Any]

    permission: ToolPermission

    timeout_seconds: float = Field(
        gt=0,
        le=600,
    )

    max_retries: int = Field(
        ge=0,
        le=10,
    )

    idempotent: bool

    max_result_bytes: int = Field(
        ge=128,
        le=50_000_000,
    )

    @field_validator(
        "input_schema",
        "output_schema",
    )
    @classmethod
    def validate_json_schema(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """工具输入输出必须是对象类型 JSON Schema。"""

        if value.get("type") != "object":
            raise ValueError(
                "工具输入输出 Schema 顶层必须为 object"
            )

        return value

    @model_validator(mode="after")
    def validate_retry_contract(self) -> Self:
        """只有幂等工具才能自动重试。"""

        if (
            self.max_retries > 0
            and not self.idempotent
        ):
            raise ValueError(
                "允许自动重试的工具必须声明 "
                "idempotent=True"
            )

        return self


class ToolCallTrace(BaseModel):
    """一次工具尝试的结构化审计记录。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    tool_call_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    request_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN
    )

    run_id: str = Field(
        pattern=_RUNTIME_ID_PATTERN,
    )

    step_id: str = Field(
        min_length=1,
        max_length=128,
    )

    tool_name: str = Field(
        pattern=_TOOL_NAME_PATTERN,
    )

    tool_version: str = Field(
        pattern=_VERSION_PATTERN,
    )

    argument_summary: dict[str, Any]

    arguments_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )

    idempotency_key: str = Field(
        pattern=_SHA256_PATTERN,
    )

    attempt: int = Field(
        ge=1,
    )

    status: ToolCallStatus

    started_at: datetime

    completed_at: datetime

    latency_ms: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    result_size_bytes: int = Field(
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

    @field_validator("argument_summary")
    @classmethod
    def validate_argument_summary(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        forbidden_path = (
            _find_forbidden_log_key(value)
        )

        if forbidden_path is not None:
            raise ValueError(
                "argument_summary 不能记录敏感字段："
                f"{forbidden_path}"
            )

        return value

    @model_validator(mode="after")
    def validate_trace_contract(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at 不能早于 started_at"
            )

        successful_statuses = {
            "succeeded",
            "reused",
        }

        if self.status in successful_statuses:
            if (
                self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "成功工具调用不能包含错误字段"
                )

        elif (
            not self.error_message
            or not self.error_type
        ):
            raise ValueError(
                "失败工具调用必须包含 "
                "error_type 和 error_message"
            )

        return self


class DocumentEvidenceQuery(BaseModel):
    """面向财报原文的文档证据检索请求。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query_id: str = Field(
        pattern=_QUERY_ID_PATTERN,
    )

    semantic_query: str = Field(
        min_length=1,
        max_length=2000,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    report_type: ReportType = (
        ReportType.ANNUAL_REPORT
    )

class RetrievedDocument(BaseModel):
    """Runtime 中受限制、可回放的文档命中。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
    )

    query_id: str = Field(
        pattern=_QUERY_ID_PATTERN,
    )

    rank: int = Field(
        ge=1,
    )

    chunk_id: str = Field(
        min_length=1,
    )

    document_id: str = Field(
        min_length=1,
    )

    page_id: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    score: float = Field(
        allow_inf_nan=False,
    )

    section_path: tuple[str, ...] = ()

    text: str = Field(
        min_length=1,
        max_length=8000,
    )


class QueryFinancialDataInput(BaseModel):
    """query_financial_data 的输入。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: ComplexRetrievalQueryOutput

    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
    )


class QueryFinancialDataOutput(BaseModel):
    """query_financial_data 的输出。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    trace: ComplexRetrievalTrace


class RetrieveDocumentsInput(BaseModel):
    """retrieve_documents 的输入。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: DocumentEvidenceQuery

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

class RetrieveDocumentsOutput(BaseModel):
    """retrieve_documents 的输出。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query_id: str = Field(
        pattern=_QUERY_ID_PATTERN,
    )

    documents: tuple[
        RetrievedDocument,
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        chunk_ids = [
            document.chunk_id
            for document in self.documents
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "documents 不能包含重复 chunk_id"
            )

        actual_ranks = tuple(
            document.rank
            for document in self.documents
        )

        expected_ranks = tuple(
            range(
                1,
                len(self.documents) + 1,
            )
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                "documents 的 rank 必须从 1 连续递增"
            )

        mismatched_query_ids = [
            document.query_id
            for document in self.documents
            if document.query_id != self.query_id
        ]

        if mismatched_query_ids:
            raise ValueError(
                "documents.query_id 必须与 "
                "RetrieveDocumentsOutput.query_id 一致"
            )

        return self

class ExecuteCalculationInput(BaseModel):
    """execute_calculation 的输入。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    calculation_id: str = Field(
        pattern=_CALCULATION_ID_PATTERN,
    )

    formula_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    input_fact_ids: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )

    @field_validator("input_fact_ids")
    @classmethod
    def validate_input_fact_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError(
                "input_fact_ids 不能包含重复 ID"
            )

        invalid_ids = [
            fact_id
            for fact_id in value
            if not fact_id.startswith("fact_")
        ]

        if invalid_ids:
            raise ValueError(
                "计算输入必须全部是 fact_id："
                f"{invalid_ids}"
            )

        return value


class ExecuteCalculationOutput(BaseModel):
    """execute_calculation 的输出。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    trace: ComplexCalculationTrace


class ToolExecutionResult(BaseModel):
    """ToolExecutor 返回的成功结果与全部尝试轨迹。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    output: dict[str, Any]

    traces: tuple[
        ToolCallTrace,
        ...,
    ] = Field(
        min_length=1,
    )

    reused: bool = False

    @model_validator(mode="after")
    def validate_execution_result(self) -> Self:
        final_trace = self.traces[-1]

        if final_trace.status not in {
            "succeeded",
            "reused",
        }:
            raise ValueError(
                "ToolExecutionResult 的最后一次调用 "
                "必须成功或复用缓存"
            )

        if self.reused != (
            final_trace.status == "reused"
        ):
            raise ValueError(
                "reused 字段必须与最终 Trace 状态一致"
            )

        tool_names = {
            trace.tool_name
            for trace in self.traces
        }

        tool_versions = {
            trace.tool_version
            for trace in self.traces
        }

        argument_hashes = {
            trace.arguments_sha256
            for trace in self.traces
        }

        idempotency_keys = {
            trace.idempotency_key
            for trace in self.traces
        }

        if len(tool_names) != 1:
            raise ValueError(
                "同一次 ToolExecutionResult "
                "只能包含同一个工具"
            )

        if len(tool_versions) != 1:
            raise ValueError(
                "同一次工具执行不能改变工具版本"
            )

        if len(argument_hashes) != 1:
            raise ValueError(
                "重试过程中不能改变工具参数"
            )

        if len(idempotency_keys) != 1:
            raise ValueError(
                "重试过程中不能改变幂等键"
            )

        if final_trace.status == "reused":
            if len(self.traces) != 1:
                raise ValueError(
                    "reused 结果只能包含一条 Trace"
                )

            if final_trace.attempt != 1:
                raise ValueError(
                    "reused Trace 的 attempt 必须为 1"
                )

            return self

        expected_attempts = tuple(
            range(
                1,
                len(self.traces) + 1,
            )
        )

        actual_attempts = tuple(
            trace.attempt
            for trace in self.traces
        )

        if actual_attempts != expected_attempts:
            raise ValueError(
                "工具重试 attempt 必须从 1 连续递增"
            )

        allowed_intermediate_statuses = {
            "retryable_error",
            "timed_out",
        }

        for trace in self.traces[:-1]:
            if (
                trace.status
                not in allowed_intermediate_statuses
            ):
                raise ValueError(
                    "最终成功前只能出现 "
                    "retryable_error 或 timed_out"
                )

        return self