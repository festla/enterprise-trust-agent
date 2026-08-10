from __future__ import annotations

from app.schemas.agent_runtime import (
    ParsedFinancialQuery,
)
from app.services.runtime_intent_router import (
    RuntimeIntentRouter,
)


def _build_query(
    *,
    metric_ids: tuple[
        str,
        ...
    ] = (),
    calculation_metric_ids: tuple[
        str,
        ...
    ] = (),
    company_ids: tuple[
        str,
        ...
    ] = (
        "midea_group",
    ),
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
    comparison_requested: bool = False,
    ranking_requested: bool = False,
    explanation_requested: bool = False,
    unsupported_reason: (
        str | None
    ) = None,
    missing_fields: tuple[
        str,
        ...
    ] = (),
    question: str = (
        "美的集团2024年财务问题"
    ),
) -> ParsedFinancialQuery:
    return ParsedFinancialQuery(
        normalized_question=question,
        company_ids=company_ids,
        report_ids=report_ids,
        years=years,
        metric_ids=metric_ids,
        calculation_metric_ids=(
            calculation_metric_ids
        ),
        comparison_requested=(
            comparison_requested
        ),
        ranking_requested=(
            ranking_requested
        ),
        explanation_requested=(
            explanation_requested
        ),
        unsupported_reason=(
            unsupported_reason
        ),
        missing_fields=(
            missing_fields
        ),
        confidence=1.0,
    )


def _build_router(
) -> RuntimeIntentRouter:
    return RuntimeIntentRouter()


# ============================================================
# reported metric
#      ↓
# financial_fact
#      ↓
# 后续 query_financial_data
# ============================================================


def test_routes_reported_metric_to_financial_fact(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        question=(
            "美的集团2024年营业收入是多少"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "financial_fact"
    )


# ============================================================
# derived metric
#       ↓
# financial_calculation
#       ↓
# Planner 后面负责：
#
# query input facts
#       ↓
# execute_calculation
# ============================================================


def test_routes_derived_metric_to_calculation(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        calculation_metric_ids=(
            "gross_profit_margin",
        ),
        question=(
            "美的集团2024年毛利率是多少"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "financial_calculation"
    )


# ============================================================
# comparison 本身会改变 Plan 拓扑，
# 所以单独作为一个 Intent。
# ============================================================


def test_routes_comparison(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        years=(
            2024,
            2025,
        ),
        report_ids=(
            "midea_group_2024",
            "midea_group_2025",
        ),
        comparison_requested=True,
        question=(
            "比较美的集团2024年和"
            "2025年的营业收入"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "financial_comparison"
    )


def test_routes_ranking_to_financial_comparison(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        ranking_requested=True,
        question=(
            "哪家公司营业收入最高"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "financial_comparison"
    )


# ============================================================
# explanation 必须覆盖普通 financial_fact。
#
# 因为：
#
# “营业收入是多少？”
# → financial_fact
#
# “为什么营业收入增长？”
# → document_evidence
# ============================================================


def test_explanation_has_priority_over_fact(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        explanation_requested=True,
        question=(
            "为什么美的集团2024年"
            "营业收入增长"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "document_evidence"
    )


# ============================================================
# 不一定所有文档问题都有 metric_id。
#
# 风险、战略、管理层讨论等内容
# 应该直接进入 document retrieval。
# ============================================================


def test_routes_document_evidence_without_metric(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        question=(
            "美的集团2024年"
            "主要经营风险有哪些"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "document_evidence"
    )


# ============================================================
# Missing != Unsupported
# ============================================================


def test_missing_fields_do_not_change_supported_intent(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        company_ids=(),
        report_ids=(),
        years=(),
        missing_fields=(
            "company_ids",
            "years",
        ),
        question=(
            "营业收入是多少"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "financial_fact"
    )


def test_explicit_unsupported_reason_wins(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        metric_ids=(
            "revenue",
        ),
        unsupported_reason=(
            "问题超出当前系统支持范围"
        ),
    )

    assert (
        router.route(
            parsed_query
        )
        == "unsupported"
    )


# ============================================================
# 一个问题同时要求：
#
# 营业收入 + 毛利率
#
# 因为至少包含一个 Derived Metric，
# 所以需要 Calculation Plan。
#
# Planner 后面仍然可以同时处理 revenue。
# ============================================================


def test_derived_metric_has_priority_over_plain_fact(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
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

    assert (
        router.route(
            parsed_query
        )
        == "financial_calculation"
    )


# ============================================================
# 没有识别到财务目标，
# 也不是支持的 document query，
# 则明确拒绝。
# ============================================================


def test_unknown_request_is_unsupported(
) -> None:
    router = _build_router()

    parsed_query = _build_query(
        company_ids=(),
        report_ids=(),
        years=(),
        question="帮我写一首诗",
    )

    assert (
        router.route(
            parsed_query
        )
        == "unsupported"
    )