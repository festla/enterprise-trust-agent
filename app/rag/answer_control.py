from __future__ import annotations

from app.schemas.evidence_context import (
    EvidenceContextItem,
    EvidenceReadinessResult,
)
from app.schemas.financial_fact_answer import (
    FinancialFactAnswerPacket,
)
from app.schemas.retrieval import (
    RetrievalQueryPlan,
)


class AnswerControlError(ValueError):
    """回答控制层基础异常。"""


class AnswerControlSourceMismatchError(
    AnswerControlError
):
    """Query Plan 与 Evidence Context 不一致。"""


def _validate_plan_and_readiness(
    *,
    plan: RetrievalQueryPlan,
    readiness: EvidenceReadinessResult,
) -> None:
    """检查 Query Plan 与 Evidence 决策是否对应。"""

    context = readiness.context

    if readiness.metric_name != plan.metric_name:
        raise AnswerControlSourceMismatchError(
            "EvidenceReadinessResult 的 "
            "metric_name 与 Query Plan 不一致"
        )

    if (
        context.original_query
        != plan.original_query
    ):
        raise AnswerControlSourceMismatchError(
            "EvidenceContext 的 original_query "
            "与 Query Plan 不一致"
        )

    if (
        context.semantic_query
        != plan.semantic_query
    ):
        raise AnswerControlSourceMismatchError(
            "EvidenceContext 的 semantic_query "
            "与 Query Plan 不一致"
        )

    if context.fiscal_year != plan.fiscal_year:
        raise AnswerControlSourceMismatchError(
            "EvidenceContext 的 fiscal_year "
            "与 Query Plan 不一致"
        )

    if (
        plan.filters.company_ids
        and context.company_id
        not in plan.filters.company_ids
    ):
        raise AnswerControlSourceMismatchError(
            "EvidenceContext 的 company_id "
            "不符合 Query Plan"
        )

    if (
        plan.filters.report_ids
        and context.report_id
        not in plan.filters.report_ids
    ):
        raise AnswerControlSourceMismatchError(
            "EvidenceContext 的 report_id "
            "不符合 Query Plan"
        )


def _render_supporting_item(
    item: EvidenceContextItem,
) -> str:
    """渲染一条允许进入模型的支持证据。"""

    citation = item.citation

    printed_page_text = (
        str(citation.printed_page)
        if citation.printed_page is not None
        else "未映射"
    )

    header = (
        f"[{citation.citation_id}] "
        f"{citation.report_title} | "
        f"来源：{citation.source_name} | "
        f"PDF第{citation.pdf_page}页 | "
        f"印刷第{printed_page_text}页"
    )

    return (
        f"{header}\n"
        f"{item.text}"
    )


def build_financial_fact_answer_packet(
    *,
    plan: RetrievalQueryPlan,
    readiness: EvidenceReadinessResult,
) -> FinancialFactAnswerPacket:
    """根据 Evidence Gate 构建回答或拒答控制结果。

    本函数不会调用模型，也不会提取最终数值。
    它只决定上层是否有资格调用模型，以及模型
    能看到哪些证据。
    """

    _validate_plan_and_readiness(
        plan=plan,
        readiness=readiness,
    )

    context = readiness.context

    if (
        readiness.status
        != "ready_for_generation"
    ):
        if (
            readiness.status
            == "no_retrieval_hits"
        ):
            message = (
                f"未检索到能够支持“"
                f"{plan.metric_name}”的证据，"
                "因此无法回答该问题。"
            )
        else:
            message = (
                f"当前检索结果不足以可靠确认“"
                f"{plan.metric_name}”的数值，"
                "因此拒绝生成答案。"
            )

        return FinancialFactAnswerPacket(
            status="refused",
            action="return_refusal",
            question=plan.original_query,
            semantic_query=plan.semantic_query,
            metric_name=plan.metric_name,
            company_id=context.company_id,
            report_id=context.report_id,
            fiscal_year=context.fiscal_year,
            message=message,
            evidence_context=context,
            supporting_items=(),
            supporting_citation_ids=(),
            used_chunk_ids=(),
            generation_context="",
            refusal_reason=readiness.reason,
        )

    item_by_citation_id = {
        item.citation.citation_id: item
        for item in context.items
    }

    supporting_items: list[
        EvidenceContextItem
    ] = []

    seen_citation_ids: set[str] = set()

    for citation_id in (
        readiness.supporting_citation_ids
    ):
        if citation_id in seen_citation_ids:
            raise AnswerControlError(
                "supporting_citation_ids "
                "不能包含重复值"
            )

        seen_citation_ids.add(citation_id)

        item = item_by_citation_id.get(
            citation_id
        )

        if item is None:
            raise AnswerControlError(
                "支持引用在 EvidenceContext "
                f"中不存在：{citation_id}"
            )

        supporting_items.append(item)

    if not supporting_items:
        raise AnswerControlError(
            "ready_for_generation "
            "缺少支持证据"
        )

    generation_context = "\n\n".join(
        _render_supporting_item(item)
        for item in supporting_items
    )

    supporting_items_tuple = tuple(
        supporting_items
    )

    return FinancialFactAnswerPacket(
        status="ready_for_generation",
        action="call_model",
        question=plan.original_query,
        semantic_query=plan.semantic_query,
        metric_name=plan.metric_name,
        company_id=context.company_id,
        report_id=context.report_id,
        fiscal_year=context.fiscal_year,
        message=(
            "证据已通过充分性检查，"
            "可以进入受控回答生成。"
        ),
        evidence_context=context,
        supporting_items=(
            supporting_items_tuple
        ),
        supporting_citation_ids=tuple(
            item.citation.citation_id
            for item in supporting_items_tuple
        ),
        used_chunk_ids=tuple(
            item.citation.chunk_id
            for item in supporting_items_tuple
        ),
        generation_context=(
            generation_context
        ),
        refusal_reason=None,
    )