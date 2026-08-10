from __future__ import annotations

import pytest

from app.schemas.company import (
    Company,
)
from app.schemas.enums import (
    MetricOrigin,
    StatementScope,
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
from app.services.runtime_query_parser import (
    RuntimeQueryParser,
    RuntimeQueryParserError,
)


# ============================================================
# Parser 测试不需要重新测试 Company/Report/Metric
# 的全部 Pydantic Schema。
#
# 所以只构造 Parser 真正需要读取的字段。
# ============================================================


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

    for (
        company_id,
        year,
    ) in (
        (
            "midea_group",
            2024,
        ),
        (
            "midea_group",
            2025,
        ),
        (
            "gree_electric",
            2024,
        ),
        (
            "gree_electric",
            2025,
        ),
    ):
        bundle.reports.add(
            Report.model_construct(
                report_id=(
                    f"{company_id}_{year}"
                ),
                company_id=company_id,
                fiscal_year=year,
            )
        )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="revenue",
            display_name_cn="营业收入",
            display_name_en="Revenue",
            metric_origin=(
                MetricOrigin.REPORTED
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id="net_profit",
            display_name_cn="净利润",
            display_name_en="Net Profit",
            metric_origin=(
                MetricOrigin.REPORTED
            ),
        )
    )

    bundle.metrics.add(
        FinancialMetric.model_construct(
            metric_id=(
                "net_profit_attributable_to_parent"
            ),
            display_name_cn=(
                "归属于母公司所有者的净利润"
            ),
            display_name_en=(
                "Net Profit Attributable "
                "to Owners of the Parent"
            ),
            metric_origin=(
                MetricOrigin.REPORTED
            ),
        )
    )

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
        )
    )

    return bundle


def _build_parser(
) -> RuntimeQueryParser:
    return RuntimeQueryParser(
        registry_bundle=(
            _build_bundle()
        )
    )



def test_empty_question_is_rejected(
) -> None:
    parser = _build_parser()

    with pytest.raises(
        RuntimeQueryParserError,
        match="不能为空",
    ):
        parser.parse("   ")


# ============================================================
#
# 最基础的一条：
#
# 自然语言
# ↓
# company / year / metric / report
# ============================================================


def test_parses_standard_financial_fact_query(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "美的集团2024年营业收入是多少？"
    )

    assert result.company_ids == (
        "midea_group",
    )

    assert result.years == (
        2024,
    )

    assert result.report_ids == (
        "midea_group_2024",
    )

    assert result.metric_ids == (
        "revenue",
    )

    assert (
        result.calculation_metric_ids
        == ()
    )

    assert (
        result.comparison_requested
        is False
    )

    assert result.confidence == 1.0


# ============================================================
# Alias Resolution：
#
# 美的 → midea_group
# 营收 → revenue
# ============================================================


def test_parses_common_aliases(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "美的2024年营收是多少"
    )

    assert result.company_ids == (
        "midea_group",
    )

    assert result.metric_ids == (
        "revenue",
    )

    assert result.report_ids == (
        "midea_group_2024",
    )


# ============================================================
# 两个年份本身就是一个强比较信号。
# ============================================================


def test_parses_multi_year_comparison(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "比较美的集团2024年和"
        "2025年的营业收入"
    )

    assert result.years == (
        2024,
        2025,
    )

    assert result.report_ids == (
        "midea_group_2024",
        "midea_group_2025",
    )

    assert (
        result.comparison_requested
        is True
    )


# ============================================================

# 毛利率是 Derived Metric，
# 所以不能进入 metric_ids。
#
# 后面 7C Planner 会把它展开成：
#
# revenue
# operating_cost
#      ↓
# gross_profit_margin_formula
# ============================================================


def test_separates_derived_metric(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "美的集团2024年毛利率是多少"
    )

    assert result.metric_ids == ()

    assert (
        result.calculation_metric_ids
        == (
            "gross_profit_margin",
        )
    )



def test_detects_explanation_request(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "为什么美的集团2024年"
        "营业收入增长？"
    )

    assert (
        result.explanation_requested
        is True
    )

    assert result.metric_ids == (
        "revenue",
    )



def test_parses_parent_company_scope(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "美的集团2024年"
        "母公司口径营业收入"
    )

    assert (
        result.statement_scope
        is StatementScope.PARENT_COMPANY
    )


# ============================================================

# 同时写“合并”和“母公司”，
# Parser 不应该偷偷替用户选一个。
#
# 正确行为：
#
# statement_scope = None
# + ambiguity_notes
# ============================================================


def test_marks_ambiguous_statement_scope(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "比较美的集团2024年"
        "合并口径和母公司口径营业收入"
    )

    assert (
        result.statement_scope
        is None
    )

    assert result.ambiguity_notes

    assert (
        "无法唯一确定"
        in result.ambiguity_notes[0]
    )


# ============================================================

# 非常重要：
#
# “归属于母公司所有者的净利润”
#
# 里面包含“净利润”。
#
# Parser 必须最长匹配优先。
# ============================================================


def test_longest_metric_phrase_wins(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "美的集团2024年"
        "归属于母公司所有者的净利润是多少"
    )

    assert result.metric_ids == (
        "net_profit_attributable_to_parent",
    )

    assert "net_profit" not in (
        result.metric_ids
    )


# ============================================================

# Parser 可以接受“不完整问题”。
#
# 不要在 Parser 直接抛异常：
#
# “营业收入是多少？”
#
# 因为后续 Runtime 可能：
#
# - 请求澄清
# - Router 判定缺字段
# - Human Interrupt
#
# ============================================================


def test_records_missing_identity_fields(
) -> None:
    parser = _build_parser()

    result = parser.parse(
        "营业收入是多少？"
    )

    assert result.company_ids == ()
    assert result.years == ()

    assert result.metric_ids == (
        "revenue",
    )

    assert result.missing_fields == (
        "company_ids",
        "years",
    )

    assert result.confidence < 1.0