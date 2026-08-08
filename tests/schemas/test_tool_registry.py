from datetime import datetime, timezone

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
)

from app.schemas.tool_registry import (
    ExecuteCalculationInput,
    RetrievedDocument,
    RetrieveDocumentsOutput,
    ToolCallTrace,
    ToolDefinition,
    ToolExecutionResult,
)


class DemoInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    value: int


class DemoOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    value: int


def _build_definition(
    *,
    max_retries: int = 1,
    idempotent: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        tool_name="demo_tool",
        description="用于测试的工具",
        version="1.0.0",
        input_schema=(
            DemoInput.model_json_schema()
        ),
        output_schema=(
            DemoOutput.model_json_schema()
        ),
        permission="execute_calculation",
        timeout_seconds=1.0,
        max_retries=max_retries,
        idempotent=idempotent,
        max_result_bytes=1024,
    )


def _build_trace(
    *,
    attempt: int,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> ToolCallTrace:
    now = datetime.now(timezone.utc)

    return ToolCallTrace(
        tool_call_id=(
            f"toolcall_{attempt}"
        ),
        request_id="request_1",
        run_id="run_1",
        step_id="s1",
        tool_name="demo_tool",
        tool_version="1.0.0",
        argument_summary={
            "value": 1,
        },
        arguments_sha256="0" * 64,
        idempotency_key="1" * 64,
        attempt=attempt,
        status=status,
        started_at=now,
        completed_at=now,
        latency_ms=0.0,
        result_size_bytes=10,
        error_type=error_type,
        error_message=error_message,
    )


def test_tool_definition_accepts_pydantic_schemas() -> None:
    definition = _build_definition()

    assert (
        definition.input_schema
        == DemoInput.model_json_schema()
    )

    assert (
        definition.output_schema
        == DemoOutput.model_json_schema()
    )


def test_tool_definition_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(
            tool_name="demo_tool",
            description="用于测试的工具",
            version="1.0.0",
            input_schema=(
                DemoInput.model_json_schema()
            ),
            output_schema=(
                DemoOutput.model_json_schema()
            ),
            permission="execute_calculation",
            timeout_seconds=1.0,
            max_retries=0,
            idempotent=True,
            max_result_bytes=1024,
            unexpected_field=True,
        )


def test_retryable_tool_must_be_idempotent() -> None:
    with pytest.raises(
        ValidationError,
        match="idempotent=True",
    ):
        _build_definition(
            max_retries=1,
            idempotent=False,
        )


def test_tool_trace_rejects_secret_argument_fields() -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(
        ValidationError,
        match="敏感字段",
    ):
        ToolCallTrace(
            tool_call_id="toolcall_1",
            request_id="request_1",
            run_id="run_1",
            step_id="s1",
            tool_name="demo_tool",
            tool_version="1.0.0",
            argument_summary={
                "api_key": "should_not_be_logged",
            },
            arguments_sha256="0" * 64,
            idempotency_key="1" * 64,
            attempt=1,
            status="succeeded",
            started_at=now,
            completed_at=now,
            latency_ms=0.0,
            result_size_bytes=10,
            error_type=None,
            error_message=None,
        )


def test_success_trace_rejects_error_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="成功工具调用",
    ):
        _build_trace(
            attempt=1,
            status="succeeded",
            error_type="UnexpectedError",
            error_message="不应存在",
        )


def test_failed_trace_requires_error_fields() -> None:
    with pytest.raises(
        ValidationError,
        match="失败工具调用",
    ):
        _build_trace(
            attempt=1,
            status="retryable_error",
        )


def test_retrieved_documents_require_stable_ranks() -> None:
    documents = (
        RetrievedDocument(
            query_id="q1",
            rank=2,
            chunk_id="chunk_1",
            document_id="document_1",
            page_id="page_1",
            company_id="midea_group",
            report_id="midea_group_2024",
            fiscal_year=2024,
            pdf_page=158,
            printed_page=157,
            score=0.8,
            section_path=(),
            text="营业收入相关内容",
        ),
    )

    with pytest.raises(
        ValidationError,
        match="rank 必须从 1 连续递增",
    ):
        RetrieveDocumentsOutput(
            query_id="q1",
            documents=documents,
        )


def test_calculation_input_only_accepts_unique_fact_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="不能包含重复 ID",
    ):
        ExecuteCalculationInput(
            calculation_id=(
                "calculation_midea_group_2024_"
                "gross_profit_margin"
            ),
            formula_id=(
                "gross_profit_margin_formula"
            ),
            input_fact_ids=(
                "fact_midea_group_2024_revenue",
                "fact_midea_group_2024_revenue",
            ),
        )

    with pytest.raises(
        ValidationError,
        match="必须全部是 fact_id",
    ):
        ExecuteCalculationInput(
            calculation_id=(
                "calculation_midea_group_2024_"
                "gross_profit_margin"
            ),
            formula_id=(
                "gross_profit_margin_formula"
            ),
            input_fact_ids=(
                "retrieval_result_q1",
            ),
        )


def test_tool_execution_result_accepts_retry_then_success() -> None:
    first_trace = _build_trace(
        attempt=1,
        status="retryable_error",
        error_type="TemporaryToolError",
        error_message="临时失败",
    )

    second_trace = _build_trace(
        attempt=2,
        status="succeeded",
    )

    result = ToolExecutionResult(
        output={
            "value": 1,
        },
        traces=(
            first_trace,
            second_trace,
        ),
        reused=False,
    )

    assert len(result.traces) == 2
    assert result.traces[-1].status == "succeeded"