from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.agent_runtime import (
    AgentAnswer,
    AgentState,
    AgentTrajectory,
    CitationRecord,
    NodeSpan,
    ParsedFinancialQuery,
    RuntimePlan,
)
from app.schemas.complex_plan_eval_result import (
    ComplexPlanOutput,
    ComplexPlanStepOutput,
    ComplexRetrievalQueryOutput,
)
from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.tool_registry import (
    DocumentEvidenceQuery,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_financial_query(
) -> ComplexRetrievalQueryOutput:
    return ComplexRetrievalQueryOutput(
        query_id="q1",
        semantic_query=(
            "美的集团 2024年 合并利润表 "
            "营业收入"
        ),
        company_id="midea_group",
        report_id="midea_group_2024",
        metric_id="revenue",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
    )


def _build_document_query(
) -> DocumentEvidenceQuery:
    return DocumentEvidenceQuery(
        query_id="q1",
        semantic_query=(
            "美的集团2024年营业收入增长原因"
        ),
        company_id="midea_group",
        report_id="midea_group_2024",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
    )


def _build_retrieval_plan(
) -> ComplexPlanOutput:
    return ComplexPlanOutput(
        steps=(
            ComplexPlanStepOutput(
                step_id="s1",
                action="retrieve",
                description="检索一个原子查询",
                input_refs=(),
                depends_on=(),
                output_ref=(
                    "retrieval_result_q1"
                ),
                retrieval_query_id="q1",
                calculation_id=None,
                formula_id=None,
            ),
            ComplexPlanStepOutput(
                step_id="s2",
                action="synthesize",
                description="生成最终答案",
                input_refs=(
                    "retrieval_result_q1",
                ),
                depends_on=(
                    "s1",
                ),
                output_ref="answer_result",
                retrieval_query_id=None,
                calculation_id=None,
                formula_id=None,
            ),
        ),
        final_step_id="s2",
    )


def _build_runtime_plan(
) -> RuntimePlan:
    return RuntimePlan(
        intent="financial_fact",
        planner_version=(
            "deterministic_planner_v1"
        ),
        normalized_question=(
            "查询美的集团2024年营业收入"
        ),
        financial_queries=(
            _build_financial_query(),
        ),
        document_queries=(),
        plan=_build_retrieval_plan(),
        tool_by_step_id={
            "s1": "query_financial_data",
        },
    )


def _build_financial_answer(
) -> AgentAnswer:
    return AgentAnswer(
        answer_type="financial",
        answer_text=(
            "美的集团2024年合并口径"
            "营业收入为407,149,600,000元。"
        ),
        supporting_fact_ids=(
            "fact_midea_group_2024_revenue",
        ),
        supporting_calculation_ids=(),
        citation_evidence_ids=(
            "evidence_midea_group_2024_revenue",
        ),
        document_citation_ids=(),
        confidence=1.0,
    )


def _build_completed_state(
) -> AgentState:
    now = _now()

    return AgentState(
        request_id="request_1",
        trace_id="trace_1",
        run_id="run_1",
        thread_id="thread_1",
        query=(
            "美的集团2024年营业收入是多少？"
        ),
        parsed_query=ParsedFinancialQuery(
            normalized_question=(
                "查询美的集团2024年营业收入"
            ),
            company_ids=(
                "midea_group",
            ),
            report_ids=(
                "midea_group_2024",
            ),
            years=(
                2024,
            ),
            metric_ids=(
                "revenue",
            ),
            calculation_metric_ids=(),
            statement_scope=(
                StatementScope.CONSOLIDATED
            ),
            comparison_requested=False,
            ranking_requested=False,
            explanation_requested=False,
            unsupported_reason=None,
            missing_fields=(),
            assumptions=(),
            ambiguity_notes=(),
            confidence=1.0,
        ),
        intent="financial_fact",
        company_ids=(
            "midea_group",
        ),
        report_ids=(
            "midea_group_2024",
        ),
        years=(
            2024,
        ),
        metric_ids=(
            "revenue",
        ),
        runtime_plan=_build_runtime_plan(),
        current_step=2,
        runtime_refs={
            "retrieval_result_q1": (
                "fact_midea_group_2024_revenue",
            ),
        },
        completed_step_ids=(
            "s1",
            "s2",
        ),
        tool_results=(),
        node_spans=(),
        tool_call_traces=(),
        retrieval_traces=(),
        calculation_traces=(),
        retrieved_documents=(),
        resolved_fact_ids=(
            "fact_midea_group_2024_revenue",
        ),
        evidence_ids=(
            "evidence_midea_group_2024_revenue",
        ),
        calculation_ids=(),
        answer=_build_financial_answer(),
        citations=(
            CitationRecord(
                citation_id="citation_1",
                report_id=(
                    "midea_group_2024"
                ),
                pdf_page=158,
                printed_page=157,
                evidence_id=(
                    "evidence_midea_group_"
                    "2024_revenue"
                ),
                chunk_id=None,
                text_excerpt=(
                    "营业收入407,149,600"
                ),
            ),
        ),
        status="completed",
        retry_count=0,
        max_steps=32,
        step_count=7,
        errors=(),
        confidence=1.0,
        stop_reason="completed",
        pending_human_review=False,
        human_review_reason=None,
        human_decision=None,
        current_node="finish",
        next_node="finish",
        checkpoint_revision=7,
        planner_version=(
            "deterministic_planner_v1"
        ),
        retriever_version=(
            "registry_financial_query_v1"
        ),
        calculator_version=None,
        generator_version=(
            "deterministic_financial_"
            "answer_generator_v2"
        ),
        prompt_version=None,
        prompt_sha256=None,
        model_name=None,
        input_tokens=0,
        output_tokens=0,
        estimated_cost=0.0,
        started_at=now,
        updated_at=now,
        completed_at=now,
    )


def test_parsed_query_rejects_duplicate_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="company_ids 不能包含重复值",
    ):
        ParsedFinancialQuery(
            normalized_question="查询营业收入",
            company_ids=(
                "midea_group",
                "midea_group",
            ),
        )


def test_runtime_plan_accepts_financial_tool_binding() -> None:
    runtime_plan = _build_runtime_plan()

    assert (
        runtime_plan.tool_by_step_id["s1"]
        == "query_financial_data"
    )


def test_runtime_plan_rejects_wrong_financial_tool() -> None:
    with pytest.raises(
        ValidationError,
        match="查询类型与工具绑定不一致",
    ):
        RuntimePlan(
            intent="financial_fact",
            planner_version=(
                "deterministic_planner_v1"
            ),
            normalized_question=(
                "查询美的集团2024年营业收入"
            ),
            financial_queries=(
                _build_financial_query(),
            ),
            document_queries=(),
            plan=_build_retrieval_plan(),
            tool_by_step_id={
                "s1": "retrieve_documents",
            },
        )


def test_runtime_plan_accepts_document_tool_binding() -> None:
    runtime_plan = RuntimePlan(
        intent="document_evidence",
        planner_version=(
            "deterministic_planner_v1"
        ),
        normalized_question=(
            "解释美的集团营业收入增长原因"
        ),
        financial_queries=(),
        document_queries=(
            _build_document_query(),
        ),
        plan=_build_retrieval_plan(),
        tool_by_step_id={
            "s1": "retrieve_documents",
        },
    )

    assert (
        runtime_plan.intent
        == "document_evidence"
    )


def test_runtime_plan_rejects_unknown_tool() -> None:
    with pytest.raises(
        ValidationError,
        match="未允许的工具",
    ):
        RuntimePlan(
            intent="financial_fact",
            planner_version=(
                "deterministic_planner_v1"
            ),
            normalized_question=(
                "查询美的集团2024年营业收入"
            ),
            financial_queries=(
                _build_financial_query(),
            ),
            document_queries=(),
            plan=_build_retrieval_plan(),
            tool_by_step_id={
                "s1": "python_eval",
            },
        )


def test_citation_requires_evidence_or_chunk() -> None:
    with pytest.raises(
        ValidationError,
        match="至少需要",
    ):
        CitationRecord(
            citation_id="citation_1",
            report_id="midea_group_2024",
            pdf_page=158,
            printed_page=157,
            evidence_id=None,
            chunk_id=None,
            text_excerpt="营业收入",
        )


def test_node_span_rejects_secret_summary() -> None:
    now = _now()

    with pytest.raises(
        ValidationError,
        match="敏感字段",
    ):
        NodeSpan(
            span_id="span_1",
            node_name="parse_query",
            attempt=1,
            status="completed",
            input_summary={
                "api_key": "secret-value",
            },
            output_summary={},
            started_at=now,
            completed_at=now,
            latency_ms=0.0,
            checkpoint_revision=1,
            error_type=None,
            error_message=None,
        )


def test_failed_node_span_requires_error_fields() -> None:
    now = _now()

    with pytest.raises(ValidationError):
        NodeSpan(
            span_id="span_1",
            node_name="execute_plan",
            attempt=1,
            status="failed",
            input_summary={},
            output_summary={},
            started_at=now,
            completed_at=now,
            latency_ms=0.0,
            checkpoint_revision=1,
            error_type=None,
            error_message=None,
        )

def test_agent_state_forbids_extra_fields() -> None:
    now = _now()

    with pytest.raises(ValidationError):
        AgentState(
            request_id="request_1",
            trace_id="trace_1",
            run_id="run_1",
            thread_id="thread_1",
            query="查询营业收入",
            started_at=now,
            updated_at=now,
            unexpected_field=True,
        )


def test_awaiting_human_requires_reason() -> None:
    now = _now()

    with pytest.raises(
        ValidationError,
        match="human_review_reason",
    ):
        AgentState(
            request_id="request_1",
            trace_id="trace_1",
            run_id="run_1",
            thread_id="thread_1",
            query="营业收入是多少？",
            status="awaiting_human",
            stop_reason=(
                "human_review_required"
            ),
            pending_human_review=True,
            human_review_reason=None,
            started_at=now,
            updated_at=now,
        )


def test_completed_state_requires_answer() -> None:
    now = _now()

    with pytest.raises(
        ValidationError,
        match="必须包含 answer",
    ):
        AgentState(
            request_id="request_1",
            trace_id="trace_1",
            run_id="run_1",
            thread_id="thread_1",
            query="查询营业收入",
            status="completed",
            stop_reason="completed",
            pending_human_review=False,
            answer=None,
            started_at=now,
            updated_at=now,
            completed_at=now,
        )


def test_completed_state_and_trajectory_are_valid() -> None:
    state = _build_completed_state()

    trajectory = AgentTrajectory(
        request_id=state.request_id,
        trace_id=state.trace_id,
        run_id=state.run_id,
        thread_id=state.thread_id,
        query=state.query,
        intent=state.intent,
        planner_version=(
            state.planner_version
        ),
        retriever_version=(
            state.retriever_version
        ),
        calculator_version=(
            state.calculator_version
        ),
        generator_version=(
            state.generator_version
        ),
        prompt_version=state.prompt_version,
        prompt_sha256=state.prompt_sha256,
        model_name=state.model_name,
        parsed_query=state.parsed_query,
        runtime_plan=state.runtime_plan,
        node_spans=state.node_spans,
        tool_call_traces=(
            state.tool_call_traces
        ),
        retrieval_traces=(
            state.retrieval_traces
        ),
        calculation_traces=(
            state.calculation_traces
        ),
        retrieved_documents=(
            state.retrieved_documents
        ),
        resolved_fact_ids=(
            state.resolved_fact_ids
        ),
        evidence_ids=state.evidence_ids,
        calculation_ids=(
            state.calculation_ids
        ),
        citations=state.citations,
        errors=state.errors,
        answer=state.answer,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        estimated_cost=(
            state.estimated_cost
        ),
        started_at=state.started_at,
        completed_at=(
            state.completed_at
            or state.updated_at
        ),
        latency_ms=0.0,
        final_status="completed",
        stop_reason="completed",
    )

    assert state.status == "completed"
    assert (
        trajectory.answer
        == state.answer
    )