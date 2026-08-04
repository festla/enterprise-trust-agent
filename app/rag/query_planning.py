from __future__ import annotations

from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.retrieval import (
    RetrievalFilter,
    RetrievalQueryPlan,
)


class QueryPlanningError(ValueError):
    """检索 Query 规划基础异常。"""


_STATEMENT_TYPE_LABELS = {
    StatementType.BALANCE_SHEET: (
        "资产负债表"
    ),
    StatementType.INCOME_STATEMENT: (
        "利润表"
    ),
    StatementType.CASH_FLOW_STATEMENT: (
        "现金流量表"
    ),
    StatementType.STATEMENT_OF_CHANGES_IN_EQUITY: (
        "所有者权益变动表"
    ),
    StatementType.FINANCIAL_SUMMARY: (
        "主要财务数据"
    ),
    StatementType.NOTE: (
        "财务报表附注"
    ),
}


_SCOPE_LABELS = {
    StatementScope.CONSOLIDATED: "合并",
    StatementScope.PARENT_COMPANY: "公司",
    StatementScope.GROUP: "集团",
}


def _build_statement_phrase(
    *,
    statement_type: StatementType,
    statement_scope: StatementScope,
) -> str:
    """将结构化报表口径转换为中文检索短语。"""

    statement_label = (
        _STATEMENT_TYPE_LABELS.get(
            statement_type
        )
    )

    if statement_label is None:
        raise QueryPlanningError(
            "当前财务事实检索不支持该 "
            "statement_type："
            f"{statement_type.value}"
        )

    scope_label = _SCOPE_LABELS.get(
        statement_scope
    )

    if scope_label is None:
        raise QueryPlanningError(
            "当前财务事实检索不支持该 "
            "statement_scope："
            f"{statement_scope.value}"
        )

    return (
        f"{scope_label}{statement_label}"
    )


def build_financial_fact_query_plan(
    *,
    original_query: str,
    metric_name: str,
    fiscal_year: int,
    company_id: str,
    report_id: str,
    report_type: ReportType,
    statement_type: StatementType,
    statement_scope: StatementScope,
    pdf_pages: tuple[int, ...] = (),
) -> RetrievalQueryPlan:
    """根据已解析的财务事实意图构造检索计划。

    本函数不负责从自然语言中猜公司、年份或指标，
    只负责把已经结构化的意图转换为稳定检索输入。
    """

    normalized_metric_name = (
        metric_name.strip()
    )

    if not normalized_metric_name:
        raise QueryPlanningError(
            "metric_name 不能为空"
        )

    statement_phrase = (
        _build_statement_phrase(
            statement_type=statement_type,
            statement_scope=statement_scope,
        )
    )

    semantic_query = " ".join(
        (
            statement_phrase,
            f"{fiscal_year}年度",
            normalized_metric_name,
            "金额",
        )
    )

    filters = RetrievalFilter(
        company_ids=(company_id,),
        report_ids=(report_id,),
        fiscal_years=(fiscal_year,),
        report_types=(report_type,),
        pdf_pages=pdf_pages,
    )

    return RetrievalQueryPlan(
        original_query=original_query,
        semantic_query=semantic_query,
        filters=filters,
        metric_name=normalized_metric_name,
        fiscal_year=fiscal_year,
        statement_type=statement_type,
        statement_scope=statement_scope,
    )