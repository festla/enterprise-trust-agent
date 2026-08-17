from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal

import pytest

from app.schemas.company import (
    Company,
)
from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
)
from app.schemas.enums import (
    MetricOrigin,
    ReportType,
    StatementScope,
    StatementType,
    ValidationStatus,
)
from app.schemas.evidence import (
    SourceEvidence,
)
from app.schemas.financial_fact import (
    FinancialFact,
)
from app.schemas.metric import (
    FinancialMetric,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.schemas.report import (
    Report,
)
from app.services.agent_runtime import (
    AgentRuntime,
)
from app.services.calculation_tool import (
    register_execute_calculation_tool,
)
from app.services.document_retrieval_tool import (
    register_retrieve_documents_tool,
)
from app.services.financial_data_tool import (
    register_query_financial_data_tool,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.runtime_intent_router import (
    RuntimeIntentRouter,
)
from app.services.runtime_plan_executor import (
    RuntimePlanExecutor,
    RuntimePlanExecutorError,
)
from app.services.runtime_planner import (
    RuntimePlanner,
)
from app.services.runtime_query_parser import (
    RuntimeQueryParser,
)
from app.services.tool_registry import (
    ToolExecutor,
    ToolRegistry,
    ToolExecutionFailedError,
)

class FixedClock:
    def now(
        self,
    ) -> datetime:
        return datetime(
            2026,
            8,
            11,
            4,
            0,
            tzinfo=timezone.utc,
        )


class SequentialIdFactory:
    def __init__(
        self,
    ) -> None:
        self._counters: dict[
            str,
            int,
        ] = {}

    def new_id(
        self,
        prefix: str,
    ) -> str:
        value = (
            self._counters.get(
                prefix,
                0,
            )
            + 1
        )

        self._counters[
            prefix
        ] = value

        return (
            f"{prefix}_{value}"
        )


# ============================================================
# Calculation Provider
# ============================================================


class FakeCalculationProvider:
    def __init__(
        self,
        *,
        fail: bool = False,
    ) -> None:
        self.fail = fail

        self.calls: list[
            dict[str, object]
        ] = []

    def calculate(
        self,
        *,
        calculation_id: str,
        formula_id: str,
        input_fact_ids: tuple[
            str,
            ...
        ],
    ) -> ComplexCalculationTrace:
        self.calls.append(
            {
                "calculation_id": (
                    calculation_id
                ),
                "formula_id": (
                    formula_id
                ),
                "input_fact_ids": (
                    input_fact_ids
                ),
            }
        )

        if self.fail:
            return (
                ComplexCalculationTrace(
                    calculation_id=(
                        calculation_id
                    ),
                    metric_id=(
                        "gross_profit_margin"
                    ),
                    formula_id=(
                        formula_id
                    ),
                    input_fact_ids=(
                        input_fact_ids
                    ),
                    status="failed",
                    result_value=None,
                    result_unit=None,
                    latency_ms=1.0,
                    error_message=(
                        "测试计算失败"
                    ),
                )
            )

        return (
            ComplexCalculationTrace(
                calculation_id=(
                    calculation_id
                ),
                metric_id=(
                    "gross_profit_margin"
                ),
                formula_id=(
                    formula_id
                ),
                input_fact_ids=(
                    input_fact_ids
                ),
                status="completed",
                result_value=(
                    Decimal("20.7768")
                ),
                result_unit="percent",
                latency_ms=1.0,
                error_message=None,
            )
        )


# ============================================================
# Document Provider
#
# 这里不需要加载真实 BGE / CrossEncoder。
#
# RuntimePlanExecutor 测的是：
#
# Runtime
# ↓
# ToolExecutor
# ↓
# RetrieveDocumentsTool
# ↓
# Runtime State
#
# Hybrid/Reranker 自己已有独立测试。
# ============================================================


class FakeDocumentHitProvider:
    def __init__(
        self,
        *,
        empty: bool = False,
    ) -> None:
        self.empty = empty

        self.calls: list[
            dict[str, object]
        ] = []

    def search(
        self,
        *,
        query,
        top_k: int,
    ):
        self.calls.append(
            {
                "query_id": (
                    query.query_id
                ),
                "top_k": top_k,
            }
        )

        if self.empty:
            return ()

        hit = (
            RerankedRetrievalHit
            .model_construct(
                rank=1,
                chunk_id=(
                    "chunk_midea_group_"
                    "2024_risk_1"
                ),
                document_id=(
                    "document_midea_group_2024"
                ),
                page_id=(
                    "page_midea_group_2024_100"
                ),
                company_id=(
                    query.company_id
                ),
                report_id=(
                    query.report_id
                ),
                fiscal_year=(
                    query.fiscal_year
                ),
                report_type=(
                    query.report_type
                ),
                pdf_page=100,
                printed_page=99,
                score=0.91,
                section_path=(
                    "经营情况讨论与分析",
                    "风险因素",
                ),
                text=(
                    "公司面临原材料价格波动、"
                    "市场竞争等经营风险。"
                ),
            )
        )

        return (
            hit,
        )


# ============================================================
# Registry Fixture
# ============================================================


def _add_financial_support(
    bundle: RegistryBundle,
    *,
    metric_id: str,
    fact_id: str,
    evidence_id: str,
    chunk_id: str,
) -> None:
    fact = (
        FinancialFact
        .model_construct(
            fact_id=fact_id,
            company_id=(
                "midea_group"
            ),
            report_id=(
                "midea_group_2024"
            ),
            metric_id=(
                metric_id
            ),
            fiscal_year=2024,
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            statement_scope=(
                StatementScope
                .CONSOLIDATED
            ),
            primary_evidence_id=(
                evidence_id
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    evidence = (
        SourceEvidence
        .model_construct(
            evidence_id=(
                evidence_id
            ),
            report_id=(
                "midea_group_2024"
            ),
            chunk_id=(
                chunk_id
            ),
            validation_status=(
                ValidationStatus
                .VERIFIED
            ),
        )
    )

    bundle.financial_facts.add(
        fact
    )

    bundle.evidences.add(
        evidence
    )


def _build_bundle(
) -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(
        Company.model_construct(
            company_id=(
                "midea_group"
            ),
            legal_name_cn=(
                "美的集团股份有限公司"
            ),
            short_name_cn=(
                "美的集团"
            ),
            stock_code="000333",
        )
    )

    bundle.reports.add(
        Report.model_construct(
            report_id=(
                "midea_group_2024"
            ),
            company_id=(
                "midea_group"
            ),
            fiscal_year=2024,
            report_type=(
                ReportType
                .ANNUAL_REPORT
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric
        .model_construct(
            metric_id="revenue",
            display_name_cn=(
                "营业收入"
            ),
            display_name_en=(
                "Revenue"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric
        .model_construct(
            metric_id=(
                "operating_cost"
            ),
            display_name_cn=(
                "营业成本"
            ),
            display_name_en=(
                "Operating Cost"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric
        .model_construct(
            metric_id=(
                "gross_profit_margin"
            ),
            display_name_cn=(
                "毛利率"
            ),
            display_name_en=(
                "Gross Profit Margin"
            ),
            metric_origin=(
                MetricOrigin.DERIVED
            ),
            statement_type=(
                StatementType
                .INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope
                .CONSOLIDATED,
                StatementScope
                .PARENT_COMPANY,
            ],
            formula_id=(
                "gross_profit_margin_formula"
            ),
        )
    )

    _add_financial_support(
        bundle,
        metric_id="revenue",
        fact_id=(
            "fact_midea_group_2024_revenue"
        ),
        evidence_id=(
            "evidence_midea_group_2024_revenue"
        ),
        chunk_id=(
            "chunk_midea_group_2024_revenue"
        ),
    )

    _add_financial_support(
        bundle,
        metric_id=(
            "operating_cost"
        ),
        fact_id=(
            "fact_midea_group_2024_"
            "operating_cost"
        ),
        evidence_id=(
            "evidence_midea_group_2024_"
            "operating_cost"
        ),
        chunk_id=(
            "chunk_midea_group_2024_"
            "operating_cost"
        ),
    )

    return bundle


# ============================================================
# Service Assembly
# ============================================================


def _build_services(
    *,
    calculation_failure: bool = False,
    empty_documents: bool = False,
):
    bundle = _build_bundle()

    clock = FixedClock()

    id_factory = (
        SequentialIdFactory()
    )

    runtime = AgentRuntime(
        query_parser=(
            RuntimeQueryParser(
                registry_bundle=(
                    bundle
                )
            )
        ),
        intent_router=(
            RuntimeIntentRouter()
        ),
        planner=(
            RuntimePlanner(
                registry_bundle=(
                    bundle
                )
            )
        ),
        clock=clock,
        id_factory=id_factory,
    )

    calculation_provider = (
        FakeCalculationProvider(
            fail=(
                calculation_failure
            )
        )
    )

    document_provider = (
        FakeDocumentHitProvider(
            empty=(
                empty_documents
            )
        )
    )

    tool_registry = (
        ToolRegistry()
    )

    register_query_financial_data_tool(
        tool_registry=(
            tool_registry
        ),
        registry_bundle=bundle,
    )

    register_execute_calculation_tool(
        tool_registry=(
            tool_registry
        ),
        calculation_provider=(
            calculation_provider
        ),
    )

    register_retrieve_documents_tool(
        tool_registry=(
            tool_registry
        ),
        hit_provider=(
            document_provider
        ),
    )

    tool_executor = (
        ToolExecutor(
            tool_registry,
            retry_backoff_seconds=0,
        )
    )

    plan_executor = (
        RuntimePlanExecutor(
            tool_executor=(
                tool_executor
            ),
            granted_permissions=(
                frozenset(
                    {
                        "read_financial_data",
                        "read_documents",
                        "execute_calculation",
                    }
                )
            ),
            financial_max_results=5,
            document_top_k=5,
            clock=clock,
            id_factory=id_factory,
        )
    )

    return (
        runtime,
        plan_executor,
        calculation_provider,
        document_provider,
    )


# ============================================================
# 8B-1 Financial Retrieval
# ============================================================


def test_executes_financial_retrieval_into_runtime_refs(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        )
    )

    state = (
        plan_executor
        .execute_available_tool_steps(
            prepared
        )
    )

    assert (
        state.runtime_refs
        == {
            (
                "retrieval_result_q1"
            ): (
                (
                    "fact_midea_group_"
                    "2024_revenue"
                ),
            )
        }
    )

    assert (
        state.resolved_fact_ids
        == (
            (
                "fact_midea_group_"
                "2024_revenue"
            ),
        )
    )

    assert (
        state.evidence_ids
        == (
            (
                "evidence_midea_group_"
                "2024_revenue"
            ),
        )
    )

    assert state.current_step == 1

    assert (
        state.completed_step_ids
        == (
            "s1",
        )
    )


def test_financial_retrieval_records_tool_and_node_traces(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "营业收入是多少？"
                )
            )
        )
    )

    assert len(
        state.tool_results
    ) == 1

    assert len(
        state.tool_call_traces
    ) == 1

    assert (
        state.tool_call_traces[0]
        .tool_name
        == "query_financial_data"
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
        "create_plan",
        "execute_plan",
    )

    assert state.step_count == 4


def test_single_fact_plan_moves_toward_verification(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "营业收入是多少？"
                )
            )
        )
    )

    assert (
        state.next_node
        == "verify_evidence"
    )

    assert state.current_step == 1


def test_financial_retrieval_reuses_tool_executor_cache(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    first_state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "营业收入是多少？"
                ),
                request_id=(
                    "request_first"
                ),
                run_id="run_shared",
            )
        )
    )

    assert (
        first_state
        .tool_results[0]
        .reused
        is False
    )

    second_state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "营业收入是多少？"
                ),
                request_id=(
                    "request_second"
                ),
                run_id="run_shared",
            )
        )
    )

    assert (
        second_state
        .tool_results[0]
        .reused
        is True
    )

    assert (
        second_state
        .tool_call_traces[0]
        .status
        == "reused"
    )


# ============================================================
# 8B-2 Calculation
# ============================================================


# ============================================================
# s1 revenue
# s2 cost
# s3 calculation
#
# 现在全部执行。
# ============================================================

def test_viewer_cannot_execute_calculation(
) -> None:
    (
        runtime,
        plan_executor,
        calculation_provider,
        _,
    ) = _build_services()

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "毛利率是多少？"
        ),
        user_role="viewer",
    )

    # 注意：
    #
    # _build_services() 中 RuntimePlanExecutor
    # 构造时仍然拥有 execute_calculation。
    #
    # 如果系统错误地继续相信
    # self.granted_permissions，
    # 这个测试就会失败。
    #
    # Step4.3 要证明：
    #
    # state.user_role 才是真正 Authority。

    with pytest.raises(
        ToolExecutionFailedError,
    ) as exc_info:
        plan_executor.execute_all_steps(
            prepared
        )

    traces = (
        exc_info.value.traces
    )

    assert len(traces) == 1

    trace = traces[0]

    assert (
        trace.tool_name
        == "execute_calculation"
    )

    assert (
        trace.error_type
        == "ToolPermissionDeniedError"
    )

    assert (
        trace.status
        == "permanent_error"
    )

    # 权限检查发生在 Handler 调用前。
    #
    # 因此真正 Calculation Provider
    # 根本不能被执行。

    assert (
        calculation_provider.calls
        == []
    )

def test_rbac_permission_snapshot_tampering_is_rejected(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入是多少？"
        ),
        user_role="viewer",
    )

    # 正常 viewer 没有 execute_calculation。
    assert (
        "execute_calculation"
        not in (
            prepared
            .granted_permissions
        )
    )

    # 模拟 Checkpoint / State 被错误修改，
    # 给 viewer 塞入管理员权限。
    tampered_state = (
        prepared.model_copy(
            update={
                "granted_permissions": (
                    "execute_calculation",
                    "read_documents",
                    "read_financial_data",
                )
            }
        )
    )

    with pytest.raises(
        RuntimePlanExecutorError,
        match="RBAC 权限快照",
    ):
        (
            plan_executor
            .execute_next_step(
                tampered_state
            )
        )

def test_calculation_plan_executes_all_tool_steps(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "毛利率是多少？"
        )
    )

    state = (
        plan_executor
        .execute_available_tool_steps(
            prepared
        )
    )

    assert (
        state.completed_step_ids
        == (
            "s1",
            "s2",
            "s3",
        )
    )

    assert state.current_step == 3

    assert (
        state.runtime_refs[
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            )
        ]
        == (
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            ),
        )
    )

    assert len(
        state.calculation_traces
    ) == 1

    trace = (
        state.calculation_traces[0]
    )

    assert (
        trace.status
        == "completed"
    )

    assert (
        trace.result_value
        == Decimal("20.7768")
    )

    assert (
        state.calculation_ids
        == (
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            ),
        )
    )

    assert (
        state.next_node
        == "verify_evidence"
    )


# ============================================================
# Formula 参数顺序：
#
# revenue
# operating_cost
#
# 不允许 Runtime sorted。
# ============================================================


def test_calculation_preserves_formula_input_order(
) -> None:
    (
        runtime,
        plan_executor,
        calculation_provider,
        _,
    ) = _build_services()

    state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "毛利率是多少？"
                )
            )
        )
    )

    assert state.current_step == 3

    assert len(
        calculation_provider.calls
    ) == 1

    assert (
        calculation_provider
        .calls[0][
            "input_fact_ids"
        ]
        == (
            (
                "fact_midea_group_"
                "2024_revenue"
            ),
            (
                "fact_midea_group_"
                "2024_operating_cost"
            ),
        )
    )


# ============================================================
# ToolExecutor 成功返回
# 并不代表 Calculation Domain 成功。
# ============================================================


def test_failed_calculation_trace_is_rejected_by_runtime(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services(
        calculation_failure=True
    )

    with pytest.raises(
        RuntimePlanExecutorError,
        match=(
            "Calculation 领域执行失败"
        ),
    ):
        (
            plan_executor
            .execute_available_tool_steps(
                runtime.prepare(
                    query=(
                        "美的集团2024年"
                        "毛利率是多少？"
                    )
                )
            )
        )


# ============================================================
# 8B-3 Document Retrieval
# ============================================================


# ============================================================
# output_ref
#      ↓
# chunk_id
#
# documents 本体进入 retrieved_documents。
# ============================================================


def test_document_retrieval_writes_chunk_refs_and_documents(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "主要经营风险有哪些？"
                )
            )
        )
    )

    assert (
        state.intent
        == "document_evidence"
    )

    assert (
        state.runtime_refs
        == {
            (
                "retrieval_result_q1"
            ): (
                (
                    "chunk_midea_group_"
                    "2024_risk_1"
                ),
            )
        }
    )

    assert len(
        state.retrieved_documents
    ) == 1

    document = (
        state.retrieved_documents[0]
    )

    assert (
        document.chunk_id
        == (
            "chunk_midea_group_"
            "2024_risk_1"
        )
    )

    assert (
        document.report_id
        == "midea_group_2024"
    )

    assert (
        "经营风险"
        in document.text
    )

    assert (
        state.next_node
        == "verify_evidence"
    )


def test_document_retrieval_records_tool_trace(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        document_provider,
    ) = _build_services()

    state = (
        plan_executor
        .execute_available_tool_steps(
            runtime.prepare(
                query=(
                    "美的集团2024年"
                    "主要经营风险有哪些？"
                )
            )
        )
    )

    assert len(
        document_provider.calls
    ) == 1

    assert (
        document_provider.calls[0]
        == {
            "query_id": "q1",
            "top_k": 5,
        }
    )

    assert len(
        state.tool_results
    ) == 1

    assert (
        state.tool_call_traces[0]
        .tool_name
        == "retrieve_documents"
    )

    # ========================================================
    # Document Tool 没有返回
    # ComplexRetrievalTrace，
    # 它使用 RetrievedDocument Contract。
    # ========================================================

    assert (
        state.retrieval_traces
        == ()
    )

    assert tuple(
        span.node_name
        for span
        in state.node_spans
    ) == (
        "parse_query",
        "route_intent",
        "create_plan",
        "execute_plan",
    )


def test_empty_document_result_is_rejected(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services(
        empty_documents=True
    )

    with pytest.raises(
        RuntimePlanExecutorError,
        match="没有返回可用文档",
    ):
        (
            plan_executor
            .execute_available_tool_steps(
                runtime.prepare(
                    query=(
                        "美的集团2024年"
                        "主要经营风险有哪些？"
                    )
                )
            )
        )


# ============================================================
# 8B Boundary
# ============================================================


# ============================================================
# s1 revenue
# s2 cost
# s3 synthesize
#
# 8B：
#
# 执行 s1 / s2
#
# 遇到 s3：
#
# 没有 tool_by_step_id
#        ↓
# 停止
#
# 8C 再处理。
# ============================================================


def test_executor_stops_before_non_tool_synthesize_step(
) -> None:
    (
        runtime,
        plan_executor,
        _,
        _,
    ) = _build_services()

    prepared = runtime.prepare(
        query=(
            "美的集团2024年"
            "营业收入和营业成本是多少？"
        )
    )

    assert (
        prepared.intent
        == "financial_fact"
    )

    assert tuple(
        step.action
        for step
        in prepared
        .runtime_plan
        .plan
        .steps
    ) == (
        "retrieve",
        "retrieve",
        "synthesize",
    )

    state = (
        plan_executor
        .execute_available_tool_steps(
            prepared
        )
    )

    assert (
        state.completed_step_ids
        == (
            "s1",
            "s2",
        )
    )

    assert state.current_step == 2

    next_step = (
        state.runtime_plan
        .plan.steps[
            state.current_step
        ]
    )

    assert (
        next_step.step_id
        == "s3"
    )

    assert (
        next_step.action
        == "synthesize"
    )

    assert (
        state.next_node
        == "execute_plan"
    )

    assert "s3" not in (
        state.runtime_plan
        .tool_by_step_id
    )