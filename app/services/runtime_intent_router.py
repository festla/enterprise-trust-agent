from __future__ import annotations

from dataclasses import dataclass

from app.schemas.agent_runtime import (
    AgentIntent,
    ParsedFinancialQuery,
)


# ============================================================
# 这些关键词用于识别：
#
# “没有标准财务 Metric，
#  但明显是在查询财报中的叙述性内容”
#
# 例如：
#
#   美的2024年有哪些经营风险？
#   管理层如何描述未来战略？
#
# 这类问题应该进入 retrieve_documents，
# 而不是 query_financial_data。
# ============================================================


_DOCUMENT_EVIDENCE_KEYWORDS = (
    "风险",
    "战略",
    "管理层",
    "经营情况",
    "经营状况",
    "业务情况",
    "业务布局",
    "市场情况",
    "市场表现",
    "行业情况",
    "竞争格局",
    "措施",
    "计划",
    "展望",
    "披露",
    "说明",
    "主要业务",
    "核心业务",
)


def _contains_any(
    text: str,
    keywords: tuple[str, ...],
) -> bool:
    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# Router 不查数据库、不查文档、不算公式。
#
# 它只根据 ParsedFinancialQuery 做任务分类。
#
#
# User Question
#       ↓
# Parser
#       ↓
# ParsedFinancialQuery
#       ↓
# Router          ← 当前这一步
#       ↓
# AgentIntent
#
# ============================================================


@dataclass(
    frozen=True,  #实例创建之后不能随便修改属性
    slots=True,   #限制实例运行的属性
)
class RuntimeIntentRouter:
    router_version: str = (
        "deterministic_runtime_intent_router_v1"
    )

    def route(
        self,
        parsed_query: ParsedFinancialQuery,
    ) -> AgentIntent:
        # ====================================================
        # 如果上游 Parser 已经明确说：
        #
        # unsupported_reason != None
        #
        # Router 不应该推翻它。
        # ====================================================

        if (
            parsed_query.unsupported_reason
            is not None
        ):
            return "unsupported"

        has_reported_metric = bool(
            parsed_query.metric_ids
        )

        has_derived_metric = bool(
            parsed_query
            .calculation_metric_ids
        )

        has_financial_target = (
            has_reported_metric
            or has_derived_metric
        )

        has_document_context = (
            bool(parsed_query.company_ids)
            or bool(parsed_query.report_ids)
            or has_financial_target
        )

        # ====================================================
        # “为什么营业收入增长？”
        #
        # 虽然里面有 revenue，
        # 但真正需求不是“营业收入是多少”，
        # 而是找原文中的解释。
        #
        # 所以 explanation 的优先级
        # 高于 financial_fact / calculation。
        # ====================================================

        if (
            parsed_query
            .explanation_requested
            and has_document_context
        ):
            return "document_evidence"

        normalized_question = (
            parsed_query
            .normalized_question
        )

        has_document_keyword = (
            _contains_any(
                normalized_question,
                _DOCUMENT_EVIDENCE_KEYWORDS,
            )
        )

        # ====================================================
        # 没有财务 Metric，但是用户明确问：
        #
        # 风险 / 战略 / 管理层 / 展望...
        #
        # 这种问题应该直接查询财报原文。
        # ====================================================

        if (
            has_document_keyword
            and has_document_context
            and not has_financial_target
        ):
            return "document_evidence"

        # ====================================================
        # 比较是一个“执行结构”。
        #
        # 例如：
        #
        # 比较2024与2025营业收入
        #
        # 后续 Planner 要生成至少两个 Retrieval Step。
        #
        # 即使比较的是 Derived Metric：
        #
        # 比较美的和格力毛利率
        #
        # 仍然属于 financial_comparison，
        # Planner 再展开具体 Calculation。
        # ====================================================

        if (
            has_financial_target
            and (
                parsed_query
                .comparison_requested
                or parsed_query
                .ranking_requested
            )
        ):
            return "financial_comparison"

        # ====================================================

        # Derived Metric：
        #
        # 毛利率
        # 流动比率
        # 资产负债率...
        #
        # 不能直接从 FinancialFact Registry
        # 当成普通披露数据拿出来，
        #
        # 而是：
        #
        # 查询输入 Fact
        #     ↓
        # execute_calculation
        # ====================================================

        if has_derived_metric:
            return "financial_calculation"

        if has_reported_metric:
            return "financial_fact"

        # ====================================================
        # 对于：
        #
        # “营业收入是多少？”
        #
        # 即使缺 company/year，
        # 上面仍然会得到 financial_fact。
        #
        # Router 不负责因为 missing_fields
        # 就把它改成 unsupported。
        #
        # 后面的 Runtime 可以：
        #
        # financial_fact
        #       +
        # missing_fields
        #       ↓
        # 请求澄清 / interrupt
        #
        # “信息不完整” != “系统不支持”。
        # ====================================================

        if (
            has_document_keyword
            and has_document_context
        ):
            return "document_evidence"

        return "unsupported"