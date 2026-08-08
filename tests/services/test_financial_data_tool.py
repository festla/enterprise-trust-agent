from __future__ import annotations

from typing import Any

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.enums import (
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
from app.schemas.tool_registry import (
    QueryFinancialDataInput,
    QueryFinancialDataOutput,
)
from app.services.financial_data_tool import (
    QueryFinancialDataTool,
    register_query_financial_data_tool,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.tool_registry import (
    ToolExecutor,
    ToolRegistry,
)


# ============================================================
# 这里使用 model_construct 是因为：
#
# FinancialFact / SourceEvidence 自己已经有大量独立
# Schema 测试。
#
# 本测试只关注 Tool Adapter 行为，
# 不需要再次构造几十个与本测试无关的字段。
# ============================================================


def _build_fact(
    *,
    fact_id: str,
    evidence_id: str,
    statement_type: StatementType = (
        StatementType.INCOME_STATEMENT
    ),
    validation_status: ValidationStatus = (
        ValidationStatus.VERIFIED
    ),
) -> FinancialFact:
    return FinancialFact.model_construct(
        fact_id=fact_id,
        company_id="midea_group",
        report_id="midea_group_2024",
        metric_id="revenue",
        fiscal_year=2024,
        statement_type=statement_type,
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
        primary_evidence_id=evidence_id,
        validation_status=(
            validation_status
        ),
    )


def _build_evidence(
    *,
    evidence_id: str,
    chunk_id: str,
    validation_status: ValidationStatus = (
        ValidationStatus.VERIFIED
    ),
) -> SourceEvidence:
    return SourceEvidence.model_construct(
        evidence_id=evidence_id,
        report_id="midea_group_2024",
        chunk_id=chunk_id,
        validation_status=(
            validation_status
        ),
    )


def _build_query(
) -> ComplexRetrievalQueryOutput:
    return ComplexRetrievalQueryOutput(
        query_id="q1",
        semantic_query=(
            "美的集团2024年合并利润表营业收入"
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


def _build_bundle(
) -> RegistryBundle:
    return RegistryBundle()


def _add_support(
    bundle: RegistryBundle,
    *,
    fact: FinancialFact,
    evidence: SourceEvidence,
) -> None:
    bundle.financial_facts.add(fact)
    bundle.evidences.add(evidence)


# ============================================================
# 验证最核心的生产规则：
#
# Query
# ↓
# verified Fact
# ↓
# verified Evidence
# ↓
# RetrievalTrace
# ============================================================


def test_query_financial_data_returns_verified_support(
) -> None:
    bundle = _build_bundle()

    fact = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue"
        ),
        evidence_id=(
            "evidence_midea_group_2024_revenue"
        ),
    )

    evidence = _build_evidence(
        evidence_id=(
            "evidence_midea_group_2024_revenue"
        ),
        chunk_id=(
            "chunk_midea_group_2024_revenue"
        ),
    )

    _add_support(
        bundle,
        fact=fact,
        evidence=evidence,
    )

    tool = QueryFinancialDataTool(
        registry_bundle=bundle
    )

    result = tool.handle(
        QueryFinancialDataInput(
            query=_build_query(),
            max_results=5,
        )
    )

    assert isinstance(
        result,
        QueryFinancialDataOutput,
    )

    assert (
        result.trace.status
        == "completed"
    )

    assert (
        result.trace.retrieved_fact_ids
        == (
            "fact_midea_group_2024_revenue",
        )
    )

    assert (
        result.trace.retrieved_evidence_ids
        == (
            "evidence_midea_group_2024_revenue",
        )
    )

    assert (
        result.trace.retrieved_chunk_ids
        == (
            "chunk_midea_group_2024_revenue",
        )
    )


# ============================================================
# 重点不是 assert 语法，
# 而是理解为什么 Agent 只能拿 verified 数据。
# ============================================================


def test_query_financial_data_filters_unverified_data(
) -> None:
    bundle = _build_bundle()

    verified_fact = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue"
        ),
        evidence_id="evidence_verified",
    )

    verified_evidence = _build_evidence(
        evidence_id="evidence_verified",
        chunk_id="chunk_verified",
    )

    pending_fact = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue_pending"
        ),
        evidence_id="evidence_pending_fact",
        validation_status=(
            ValidationStatus.PENDING
        ),
    )

    pending_fact_evidence = (
        _build_evidence(
            evidence_id=(
                "evidence_pending_fact"
            ),
            chunk_id="chunk_pending_fact",
        )
    )

    fact_with_pending_evidence = (
        _build_fact(
            fact_id=(
                "fact_midea_group_2024_"
                "revenue_pending_evidence"
            ),
            evidence_id=(
                "evidence_pending"
            ),
        )
    )

    pending_evidence = _build_evidence(
        evidence_id="evidence_pending",
        chunk_id="chunk_pending",
        validation_status=(
            ValidationStatus.PENDING
        ),
    )

    _add_support(
        bundle,
        fact=verified_fact,
        evidence=verified_evidence,
    )

    _add_support(
        bundle,
        fact=pending_fact,
        evidence=pending_fact_evidence,
    )

    _add_support(
        bundle,
        fact=fact_with_pending_evidence,
        evidence=pending_evidence,
    )

    tool = QueryFinancialDataTool(
        registry_bundle=bundle
    )

    result = tool.handle(
        QueryFinancialDataInput(
            query=_build_query(),
            max_results=5,
        )
    )

    assert (
        result.trace.retrieved_fact_ids
        == (
            "fact_midea_group_2024_revenue",
        )
    )

    assert (
        result.trace.retrieved_evidence_ids
        == (
            "evidence_verified",
        )
    )


def test_query_financial_data_filters_statement_type(
) -> None:
    bundle = _build_bundle()

    wrong_fact = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue_balance"
        ),
        evidence_id="evidence_balance",
        statement_type=(
            StatementType.BALANCE_SHEET
        ),
    )

    evidence = _build_evidence(
        evidence_id="evidence_balance",
        chunk_id="chunk_balance",
    )

    _add_support(
        bundle,
        fact=wrong_fact,
        evidence=evidence,
    )

    tool = QueryFinancialDataTool(
        registry_bundle=bundle
    )

    result = tool.handle(
        QueryFinancialDataInput(
            query=_build_query(),
            max_results=5,
        )
    )

    assert (
        result.trace.retrieved_fact_ids
        == ()
    )

    assert (
        result.trace.status
        == "completed"
    )


def test_query_financial_data_is_stable_and_limited(
) -> None:
    bundle = _build_bundle()

    fact_b = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue_b"
        ),
        evidence_id="evidence_b",
    )

    evidence_b = _build_evidence(
        evidence_id="evidence_b",
        chunk_id="chunk_b",
    )

    fact_a = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue_a"
        ),
        evidence_id="evidence_a",
    )

    evidence_a = _build_evidence(
        evidence_id="evidence_a",
        chunk_id="chunk_a",
    )

    # 故意 B 先加入。
    _add_support(
        bundle,
        fact=fact_b,
        evidence=evidence_b,
    )

    _add_support(
        bundle,
        fact=fact_a,
        evidence=evidence_a,
    )

    tool = QueryFinancialDataTool(
        registry_bundle=bundle
    )

    result = tool.handle(
        QueryFinancialDataInput(
            query=_build_query(),
            max_results=1,
        )
    )

    assert (
        result.trace.retrieved_fact_ids
        == (
            "fact_midea_group_2024_revenue_a",
        )
    )

    assert result.trace.top_k == 1


# ============================================================
# 这个测试第一次验证完整链路：
#
# Production Tool
#      ↓
# ToolRegistry
#      ↓
# ToolExecutor
#      ↓
# Permission / Schema / Hash / Idempotency
#      ↓
# RegistryBundle
# ============================================================


def test_financial_data_tool_runs_through_executor(
) -> None:
    bundle = _build_bundle()

    fact = _build_fact(
        fact_id=(
            "fact_midea_group_2024_revenue"
        ),
        evidence_id="evidence_revenue",
    )

    evidence = _build_evidence(
        evidence_id="evidence_revenue",
        chunk_id="chunk_revenue",
    )

    _add_support(
        bundle,
        fact=fact,
        evidence=evidence,
    )

    tool_registry = ToolRegistry()

    register_query_financial_data_tool(
        tool_registry=tool_registry,
        registry_bundle=bundle,
    )

    executor = ToolExecutor(
        tool_registry,
        retry_backoff_seconds=0,
    )

    arguments: dict[str, Any] = (
        QueryFinancialDataInput(
            query=_build_query(),
            max_results=5,
        ).model_dump(
            mode="json"
        )
    )

    result = executor.execute(
        tool_name="query_financial_data",
        arguments=arguments,
        request_id="request_1",
        run_id="run_1",
        step_id="s1",
        granted_permissions={
            "read_financial_data",
        },
    )

    assert result.reused is False

    assert (
        result.traces[0].status
        == "succeeded"
    )

    assert (
        tuple(
            result.output[
                "trace"
            ][
                "retrieved_fact_ids"
            ]
        )
        == (
            "fact_midea_group_2024_revenue",
        )
    )

    # 再执行相同逻辑调用。
    second_result = executor.execute(
        tool_name="query_financial_data",
        arguments=arguments,
        request_id="request_1",
        run_id="run_1",
        step_id="s1",
        granted_permissions={
            "read_financial_data",
        },
    )

    # query_financial_data 是幂等 Tool，
    # 所以第二次应该直接复用成功结果。
    assert second_result.reused is True

    assert (
        second_result.traces[0].status
        == "reused"
    )