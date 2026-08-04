import pytest

from app.rag.query_planning import (
    QueryPlanningError,
    build_financial_fact_query_plan,
)
from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
)


def test_build_consolidated_income_fact_plan(
) -> None:
    plan = build_financial_fact_query_plan(
        original_query=(
            "美的集团2024年营业收入是多少？"
        ),
        metric_name="营业收入",
        fiscal_year=2024,
        company_id="midea_group",
        report_id="midea_group_2024",
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

    assert plan.semantic_query == (
        "合并利润表 2024年度 营业收入 金额"
    )

    assert plan.original_query == (
        "美的集团2024年营业收入是多少？"
    )

    assert plan.filters.company_ids == (
        "midea_group",
    )

    assert plan.filters.report_ids == (
        "midea_group_2024",
    )

    assert plan.filters.fiscal_years == (
        2024,
    )


def test_build_parent_company_balance_sheet_plan(
) -> None:
    plan = build_financial_fact_query_plan(
        original_query="母公司货币资金是多少？",
        metric_name="货币资金",
        fiscal_year=2024,
        company_id="midea_group",
        report_id="midea_group_2024",
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.BALANCE_SHEET
        ),
        statement_scope=(
            StatementScope.PARENT_COMPANY
        ),
    )

    assert plan.semantic_query == (
        "公司资产负债表 2024年度 "
        "货币资金 金额"
    )


def test_query_plan_preserves_page_filter(
) -> None:
    plan = build_financial_fact_query_plan(
        original_query="营业收入是多少？",
        metric_name="营业收入",
        fiscal_year=2024,
        company_id="midea_group",
        report_id="midea_group_2024",
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
        pdf_pages=(158,),
    )

    assert plan.filters.pdf_pages == (
        158,
    )


def test_reject_unsupported_statement_scope(
) -> None:
    with pytest.raises(
        QueryPlanningError,
        match="statement_scope",
    ):
        build_financial_fact_query_plan(
            original_query="营业收入是多少？",
            metric_name="营业收入",
            fiscal_year=2024,
            company_id="midea_group",
            report_id="midea_group_2024",
            report_type=(
                ReportType.ANNUAL_REPORT
            ),
            statement_type=(
                StatementType.INCOME_STATEMENT
            ),
            statement_scope=(
                StatementScope.UNKNOWN
            ),
        )