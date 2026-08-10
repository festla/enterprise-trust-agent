from __future__ import annotations

import pytest

from app.schemas.agent_runtime import (
    ParsedFinancialQuery,
)
from app.schemas.company import (
    Company,
)
from app.schemas.enums import (
    MetricOrigin,
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.metric import (
    FinancialMetric,
)
from app.schemas.report import (
    Report,
)
from app.services.registry import (
    RegistryBundle,
)
from app.services.runtime_planner import (
    RuntimePlanner,
    RuntimePlannerError,
)



def _build_bundle(
) -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(
        Company.model_construct(
            company_id="midea_group",
            legal_name_cn=(
                "美的集团股份有限公司"
            ),
            short_name_cn="美的集团",
            stock_code="000333",
        )
    )

    bundle.companies.add(
        Company.model_construct(
            company_id="gree_electric",
            legal_name_cn=(
                "珠海格力电器股份有限公司"
            ),
            short_name_cn="格力电器",
            stock_code="000651",
        )
    )

    for company_id in (
        "midea_group",
        "gree_electric",
    ):
        for year in (
            2024,
            2025,
        ):
            bundle.reports.add(
                Report.model_construct(
                    report_id=(
                        f"{company_id}_{year}"
                    ),
                    company_id=company_id,
                    fiscal_year=year,
                    report_type=(
                        ReportType.ANNUAL_REPORT
                    ),
                )
            )
    # --------------------------------------------------------
    # Reported Metrics
    # --------------------------------------------------------

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn="营业收入",
            display_name_en="Revenue",
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType.INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="operating_cost",
            display_name_cn="营业成本",
            display_name_en=(
                "Operating Cost"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType.INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="current_assets",
            display_name_cn="流动资产",
            display_name_en=(
                "Current Assets"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType.BALANCE_SHEET
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id=(
                "current_liabilities"
            ),
            display_name_cn="流动负债",
            display_name_en=(
                "Current Liabilities"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
            statement_type=(
                StatementType.BALANCE_SHEET
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=None,
        )
    )

    # --------------------------------------------------------
    # Derived Metrics
    # --------------------------------------------------------

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id=(
                "gross_profit_margin"
            ),
            display_name_cn="毛利率",
            display_name_en=(
                "Gross Profit Margin"
            ),
            metric_origin=(
                MetricOrigin.DERIVED
            ),
            statement_type=(
                StatementType.INCOME_STATEMENT
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=(
                "gross_profit_margin_formula"
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="current_ratio",
            display_name_cn="流动比率",
            display_name_en=(
                "Current Ratio"
            ),
            metric_origin=(
                MetricOrigin.DERIVED
            ),
            statement_type=(
                StatementType.BALANCE_SHEET
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
                StatementScope.PARENT_COMPANY,
            ],
            formula_id=(
                "current_ratio_formula"
            ),
        )
    )

    return bundle


def _build_planner(
) -> RuntimePlanner:
    return RuntimePlanner(
        registry_bundle=_build_bundle()
    )


def _build_parsed_query(
    *,
    report_ids: tuple[
        str,
        ...
    ] = (
        "midea_group_2024",
    ),
    years: tuple[
        int,
        ...
    ] = (
        2024,
    ),
    metric_ids: tuple[
        str,
        ...
    ] = (
        "revenue",
    ),
    calculation_metric_ids: tuple[
        str,
        ...
    ] = (),
    statement_scope: (
        StatementScope | None
    ) = None,
    missing_fields: tuple[
        str,
        ...
    ] = (),
    question: str = (
        "美的集团2024年营业收入是多少"
    ),
    company_ids: tuple[
        str,
        ...
    ] = (
        "midea_group",
    ),

    comparison_requested: bool = False,

    ranking_requested: bool = False,
) -> ParsedFinancialQuery:
    return ParsedFinancialQuery(
        normalized_question=question,
        report_ids=report_ids,
        years=years,
        metric_ids=metric_ids,
        calculation_metric_ids=(
            calculation_metric_ids
        ),
        statement_scope=(
            statement_scope
        ),
        missing_fields=(
            missing_fields
        ),
        confidence=1.0,
        company_ids=company_ids,

        comparison_requested=(
            comparison_requested
        ),

        ranking_requested=(
            ranking_requested
        ),
    )


# ============================================================
# 7C-1 原有测试
# ============================================================


def test_planner_rejects_unsupported_intent(
) -> None:
    planner = _build_planner()

    with pytest.raises(
        RuntimePlannerError,
        match="unsupported",
    ):
        planner.create_plan(
            parsed_query=(
                _build_parsed_query()
            ),
            intent="unsupported",
        )


def test_planner_rejects_missing_identity(
) -> None:
    planner = _build_planner()

    parsed_query = (
        ParsedFinancialQuery(
            normalized_question=(
                "营业收入是多少"
            ),
            metric_ids=(
                "revenue",
            ),
            missing_fields=(
                "company_ids",
                "years",
            ),
            confidence=0.3333,
        )
    )

    with pytest.raises(
        RuntimePlannerError,
        match="身份字段缺失",
    ):
        planner.create_plan(
            parsed_query=parsed_query,
            intent="financial_fact",
        )


def test_builds_single_financial_fact_plan(
) -> None:
    planner = _build_planner()

    runtime_plan = (
        planner.create_plan(
            parsed_query=(
                _build_parsed_query()
            ),
            intent="financial_fact",
        )
    )

    assert (
        runtime_plan.intent
        == "financial_fact"
    )

    assert len(
        runtime_plan.financial_queries
    ) == 1

    query = (
        runtime_plan
        .financial_queries[0]
    )

    assert query.query_id == "q1"

    assert (
        query.metric_id
        == "revenue"
    )

    assert (
        query.report_id
        == "midea_group_2024"
    )

    assert (
        query.statement_scope
        is StatementScope.CONSOLIDATED
    )

    steps = (
        runtime_plan.plan.steps
    )

    assert len(steps) == 1

    assert (
        steps[0].step_id
        == "s1"
    )

    assert (
        steps[0].action
        == "retrieve"
    )

    assert (
        steps[0].retrieval_query_id
        == "q1"
    )

    assert (
        runtime_plan
        .tool_by_step_id
        == {
            "s1": (
                "query_financial_data"
            )
        }
    )

    assert (
        runtime_plan.plan.final_step_id
        == "s1"
    )


def test_multiple_financial_metrics_are_atomic(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(
                "revenue",
                "operating_cost",
            ),
            question=(
                "美的集团2024年"
                "营业收入和营业成本分别是多少"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent="financial_fact",
        )
    )

    assert tuple(
        query.metric_id
        for query
        in runtime_plan
        .financial_queries
    ) == (
        "revenue",
        "operating_cost",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "synthesize",
    )

    synthesize_step = (
        runtime_plan.plan.steps[-1]
    )

    assert (
        synthesize_step.input_refs
        == (
            "retrieval_result_q1",
            "retrieval_result_q2",
        )
    )

    assert (
        synthesize_step.depends_on
        == (
            "s1",
            "s2",
        )
    )

    assert (
        runtime_plan
        .tool_by_step_id
        == {
            "s1": (
                "query_financial_data"
            ),
            "s2": (
                "query_financial_data"
            ),
        }
    )


def test_explicit_parent_scope_is_preserved(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            statement_scope=(
                StatementScope
                .PARENT_COMPANY
            ),
            question=(
                "美的集团2024年"
                "母公司口径营业收入"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent="financial_fact",
        )
    )

    assert (
        runtime_plan
        .financial_queries[0]
        .statement_scope
        is StatementScope.PARENT_COMPANY
    )


def test_builds_document_evidence_plan(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            question=(
                "美的集团2024年"
                "主要经营风险有哪些"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent="document_evidence",
        )
    )

    assert (
        runtime_plan.intent
        == "document_evidence"
    )

    assert (
        runtime_plan
        .financial_queries
        == ()
    )

    assert len(
        runtime_plan.document_queries
    ) == 1

    query = (
        runtime_plan
        .document_queries[0]
    )

    assert query.query_id == "q1"

    assert (
        query.semantic_query
        == (
            "美的集团2024年"
            "主要经营风险有哪些"
        )
    )

    assert (
        runtime_plan
        .tool_by_step_id
        == {
            "s1": (
                "retrieve_documents"
            )
        }
    )


def test_multiple_document_reports_are_synthesized(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
                "midea_group_2025",
            ),
            years=(
                2024,
                2025,
            ),
            metric_ids=(),
            question=(
                "比较美的集团2024年和"
                "2025年的战略变化"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent="document_evidence",
        )
    )

    assert tuple(
        query.report_id
        for query
        in runtime_plan
        .document_queries
    ) == (
        "midea_group_2024",
        "midea_group_2025",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "synthesize",
    )

    assert (
        runtime_plan.plan.final_step_id
        == "s3"
    )

    assert (
        runtime_plan
        .tool_by_step_id
        == {
            "s1": "retrieve_documents",
            "s2": "retrieve_documents",
        }
    )


# ============================================================
# 7C-2 Calculation Planner
# ============================================================


# ============================================================
# 毛利率：
#
# q1 revenue
# q2 operating_cost
#       ↓
# s3 calculation
# ============================================================


def test_builds_gross_profit_margin_calculation_plan(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            calculation_metric_ids=(
                "gross_profit_margin",
            ),
            question=(
                "美的集团2024年毛利率是多少"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )
    )

    assert (
        runtime_plan.intent
        == "financial_calculation"
    )

    assert tuple(
        query.metric_id
        for query
        in runtime_plan
        .financial_queries
    ) == (
        "revenue",
        "operating_cost",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "calculate",
    )

    calculation_step = (
        runtime_plan.plan.steps[2]
    )

    assert (
        calculation_step.input_refs
        == (
            "retrieval_result_q1",
            "retrieval_result_q2",
        )
    )

    assert (
        calculation_step.depends_on
        == (
            "s1",
            "s2",
        )
    )

    assert (
        calculation_step.formula_id
        == (
            "gross_profit_margin_formula"
        )
    )

    assert (
        calculation_step.calculation_id
        == (
            "calculation_midea_group_"
            "2024_gross_profit_margin"
        )
    )

    assert (
        calculation_step.output_ref
        == calculation_step.calculation_id
    )

    assert (
        runtime_plan
        .tool_by_step_id
        == {
            "s1": (
                "query_financial_data"
            ),
            "s2": (
                "query_financial_data"
            ),
            "s3": (
                "execute_calculation"
            ),
        }
    )

    assert (
        runtime_plan.plan.final_step_id
        == "s3"
    )


# ============================================================
# revenue 既是：
#
# 用户直接请求结果
#
# 又是：
#
# gross_profit_margin input
#
# Planner 必须复用同一个 Retrieval。
# ============================================================


def test_calculation_reuses_direct_metric_retrieval(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(
                "revenue",
            ),
            calculation_metric_ids=(
                "gross_profit_margin",
            ),
            question=(
                "美的集团2024年"
                "营业收入和毛利率分别是多少"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )
    )

    # revenue 不能出现两次。
    assert tuple(
        query.metric_id
        for query
        in runtime_plan
        .financial_queries
    ) == (
        "revenue",
        "operating_cost",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "calculate",
        "synthesize",
    )

    calculation_step = (
        runtime_plan.plan.steps[2]
    )

    assert (
        calculation_step.input_refs
        == (
            "retrieval_result_q1",
            "retrieval_result_q2",
        )
    )

    synthesize_step = (
        runtime_plan.plan.steps[3]
    )

    # 最终回答需要：
    #
    # revenue
    # +
    # gross_profit_margin
    assert (
        synthesize_step.input_refs
        == (
            "retrieval_result_q1",
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            ),
        )
    )

    assert (
        synthesize_step.depends_on
        == (
            "s1",
            "s3",
        )
    )


# ============================================================
# 两个 Derived Metric：
#
# gross_profit_margin
# current_ratio
#
# 会分别产生 Calculation，
# 最后 synthesize。
# ============================================================


def test_multiple_calculations_are_synthesized(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            calculation_metric_ids=(
                "gross_profit_margin",
                "current_ratio",
            ),
            question=(
                "美的集团2024年"
                "毛利率和流动比率分别是多少"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )
    )

    assert tuple(
        query.metric_id
        for query
        in runtime_plan
        .financial_queries
    ) == (
        "revenue",
        "operating_cost",
        "current_assets",
        "current_liabilities",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "calculate",
        "retrieve",
        "retrieve",
        "calculate",
        "synthesize",
    )

    assert (
        runtime_plan.plan.final_step_id
        == "s7"
    )

    synthesize_step = (
        runtime_plan.plan.steps[-1]
    )

    assert (
        synthesize_step.input_refs
        == (
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            ),
            (
                "calculation_midea_group_"
                "2024_current_ratio"
            ),
        )
    )

    assert (
        synthesize_step.depends_on
        == (
            "s3",
            "s6",
        )
    )


# ============================================================
# 用户明确母公司口径，
# Calculation 的所有输入必须保持同一口径。
# ============================================================


def test_calculation_preserves_parent_scope(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            calculation_metric_ids=(
                "gross_profit_margin",
            ),
            statement_scope=(
                StatementScope
                .PARENT_COMPANY
            ),
            question=(
                "美的集团2024年"
                "母公司口径毛利率"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )
    )

    assert all(
        query.statement_scope
        is StatementScope.PARENT_COMPANY
        for query
        in runtime_plan.financial_queries
    )


# ============================================================
# Registry 可以存在 Derived Metric，
# 但 Runtime Planner 未必支持它对应的 Formula。
#
# 不能偷偷乱算。
# ============================================================


def test_rejects_unsupported_calculation_formula(
) -> None:
    bundle = _build_bundle()

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="broken_ratio",
            display_name_cn="测试比率",
            display_name_en=(
                "Broken Ratio"
            ),
            metric_origin=(
                MetricOrigin.DERIVED
            ),
            statement_type=(
                StatementType.FINANCIAL_SUMMARY
            ),
            allowed_scopes=[
                StatementScope.CONSOLIDATED,
            ],
            formula_id=(
                "unsupported_formula"
            ),
        )
    )

    planner = RuntimePlanner(
        registry_bundle=bundle
    )

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            calculation_metric_ids=(
                "broken_ratio",
            ),
        )
    )

    with pytest.raises(
        RuntimePlannerError,
        match="不支持 formula_id",
    ):
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )


# ============================================================
# calculation_metric_ids 必须真的是 Derived Metric。
# ============================================================


def test_rejects_reported_metric_as_calculation_metric(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            metric_ids=(),
            calculation_metric_ids=(
                "revenue",
            ),
        )
    )

    with pytest.raises(
        RuntimePlannerError,
        match="derived metric",
    ):
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_calculation"
            ),
        )

# ============================================================
# 7C-3 Comparison / Ranking Planner
# ============================================================


# ============================================================
# 2024 Revenue ──┐
#                ├→ compare
# 2025 Revenue ──┘
# ============================================================


def test_builds_multi_year_financial_comparison(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
                "midea_group_2025",
            ),
            years=(
                2024,
                2025,
            ),
            metric_ids=(
                "revenue",
            ),
            comparison_requested=True,
            question=(
                "比较美的集团2024年和"
                "2025年的营业收入"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )
    )

    assert tuple(
        query.report_id
        for query
        in runtime_plan.financial_queries
    ) == (
        "midea_group_2024",
        "midea_group_2025",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "compare",
    )

    compare_step = (
        runtime_plan.plan.steps[-1]
    )

    assert (
        compare_step.input_refs
        == (
            "retrieval_result_q1",
            "retrieval_result_q2",
        )
    )

    assert (
        compare_step.depends_on
        == (
            "s1",
            "s2",
        )
    )

    assert (
        runtime_plan.plan.final_step_id
        == "s3"
    )

    assert "s3" not in (
        runtime_plan.tool_by_step_id
    )


# ============================================================
# 美的 Revenue ──┐
#                ├→ rank
# 格力 Revenue ──┘
# ============================================================


def test_builds_cross_company_ranking(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            company_ids=(
                "midea_group",
                "gree_electric",
            ),
            report_ids=(
                "midea_group_2024",
                "gree_electric_2024",
            ),
            years=(
                2024,
            ),
            metric_ids=(
                "revenue",
            ),
            ranking_requested=True,
            question=(
                "美的和格力2024年"
                "谁的营业收入更高"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )
    )

    assert tuple(
        query.company_id
        for query
        in runtime_plan.financial_queries
    ) == (
        "midea_group",
        "gree_electric",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "rank",
    )

    rank_step = (
        runtime_plan.plan.steps[-1]
    )

    assert (
        rank_step.output_ref
        == "ranking_result"
    )

    assert "s3" not in (
        runtime_plan.tool_by_step_id
    )


# ============================================================
# Derived Metric 也可以比较：
#
# 2024 revenue ──┐
# 2024 cost ─────┴→ margin 2024 ──┐
#                                  ├→ compare
# 2025 revenue ──┐                 │
# 2025 cost ─────┴→ margin 2025 ──┘
# ============================================================


def test_builds_derived_metric_comparison(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
                "midea_group_2025",
            ),
            years=(
                2024,
                2025,
            ),
            metric_ids=(),
            calculation_metric_ids=(
                "gross_profit_margin",
            ),
            comparison_requested=True,
            question=(
                "比较美的集团2024年和"
                "2025年的毛利率"
            ),
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )
    )

    assert tuple(
        query.metric_id
        for query
        in runtime_plan.financial_queries
    ) == (
        "revenue",
        "operating_cost",
        "revenue",
        "operating_cost",
    )

    assert tuple(
        step.action
        for step
        in runtime_plan.plan.steps
    ) == (
        "retrieve",
        "retrieve",
        "calculate",
        "retrieve",
        "retrieve",
        "calculate",
        "compare",
    )

    compare_step = (
        runtime_plan.plan.steps[-1]
    )

    assert (
        compare_step.input_refs
        == (
            (
                "calculation_midea_group_"
                "2024_gross_profit_margin"
            ),
            (
                "calculation_midea_group_"
                "2025_gross_profit_margin"
            ),
        )
    )

    assert (
        compare_step.depends_on
        == (
            "s3",
            "s6",
        )
    )

    assert (
        runtime_plan
        .tool_by_step_id["s3"]
        == "execute_calculation"
    )

    assert (
        runtime_plan
        .tool_by_step_id["s6"]
        == "execute_calculation"
    )

    assert "s7" not in (
        runtime_plan.tool_by_step_id
    )


# ============================================================
# 如果同时：
#
# comparison_requested=True
# ranking_requested=True
#
# Ranking 更具体，因此优先。
# ============================================================


def test_ranking_has_priority_over_compare(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
                "midea_group_2025",
            ),
            years=(
                2024,
                2025,
            ),
            metric_ids=(
                "revenue",
            ),
            comparison_requested=True,
            ranking_requested=True,
        )
    )

    runtime_plan = (
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )
    )

    assert (
        runtime_plan.plan.steps[-1].action
        == "rank"
    )


# ============================================================
# 目前 Ranking 必须有唯一 Metric。
#
# 否则没有统一排序尺度。
# ============================================================


def test_ranking_rejects_multiple_target_metrics(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
                "midea_group_2025",
            ),
            years=(
                2024,
                2025,
            ),
            metric_ids=(
                "revenue",
                "operating_cost",
            ),
            ranking_requested=True,
        )
    )

    with pytest.raises(
        RuntimePlannerError,
        match="一个目标财务指标",
    ):
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )


# ============================================================
# 一个结果不能 Compare。
# ============================================================


def test_comparison_requires_two_results(
) -> None:
    planner = _build_planner()

    parsed_query = (
        _build_parsed_query(
            report_ids=(
                "midea_group_2024",
            ),
            years=(
                2024,
            ),
            metric_ids=(
                "revenue",
            ),
            comparison_requested=True,
        )
    )

    with pytest.raises(
        RuntimePlannerError,
        match="至少需要两个",
    ):
        planner.create_plan(
            parsed_query=parsed_query,
            intent=(
                "financial_comparison"
            ),
        )