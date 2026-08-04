from __future__ import annotations

import re
from typing import Protocol

from app.schemas.answer_generation import (
    FinancialFactFinalResult,
    GeneratedFinancialFactAnswer,
)
from app.schemas.financial_fact_answer import (
    FinancialFactAnswerPacket,
)


class AnswerGenerationError(ValueError):
    """受控回答生成基础异常。"""


class UnauthorizedGeneratedCitationError(
    AnswerGenerationError
):
    """生成结果引用了未授权证据。"""


class MissingInlineCitationError(
    AnswerGenerationError
):
    """回答正文没有包含结构化引用。"""


class InvalidAnswerProviderError(
    AnswerGenerationError
):
    """Answer Provider 配置无效。"""


class FinancialFactAnswerProvider(
    Protocol
):
    """财务事实回答 Provider 契约。"""

    @property
    def provider_id(self) -> str:
        """返回可追踪的 Provider 与模型标识。"""

    def generate(
        self,
        *,
        question: str,
        metric_name: str,
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...,
        ],
    ) -> GeneratedFinancialFactAnswer:
        """根据已通过 Gate 的证据生成回答。"""


_INLINE_CITATION_PATTERN = re.compile(
    r"\[(E[1-9][0-9]*)\]"
)


def _validate_generated_citations(
    *,
    generated: GeneratedFinancialFactAnswer,
    allowed_citation_ids: tuple[
        str,
        ...,
    ],
) -> None:
    """检查模型生成的引用是否全部经过授权。"""

    allowed_set = set(
        allowed_citation_ids
    )

    generated_set = set(
        generated.citation_ids
    )

    unauthorized = (
        generated_set - allowed_set
    )

    if unauthorized:
        values = ", ".join(
            sorted(unauthorized)
        )

        raise UnauthorizedGeneratedCitationError(
            "生成答案引用了未授权证据："
            f"{values}"
        )

    inline_citation_ids = tuple(
        _INLINE_CITATION_PATTERN.findall(
            generated.answer_text
        )
    )

    inline_set = set(
        inline_citation_ids
    )

    if not inline_citation_ids:
        raise MissingInlineCitationError(
            "回答正文必须包含 [E1] 形式的引用"
        )

    if inline_set != generated_set:
        raise MissingInlineCitationError(
            "回答正文中的引用与 "
            "citation_ids 不一致"
        )


def generate_financial_fact_answer(
    *,
    packet: FinancialFactAnswerPacket,
    provider: FinancialFactAnswerProvider,
) -> FinancialFactFinalResult:
    """根据 Answer Packet 生成最终回答或直接拒答。

    refused Packet 不得调用 Provider。

    ready Packet 只能把 generation_context
    发送给 Provider，并对生成引用进行白名单校验。
    """

    if packet.status == "refused":
        return FinancialFactFinalResult(
            status="refused",
            question=packet.question,
            metric_name=packet.metric_name,
            company_id=packet.company_id,
            report_id=packet.report_id,
            fiscal_year=packet.fiscal_year,
            answer_text=packet.message,
            citation_ids=(),
            citations=(),
            used_chunk_ids=(),
            generator_id=None,
            refusal_reason=(
                packet.refusal_reason
            ),
        )

    if packet.action != "call_model":
        raise AnswerGenerationError(
            "ready_for_generation Packet "
            "必须使用 call_model 动作"
        )

    provider_id = (
        provider.provider_id.strip()
    )

    if not provider_id:
        raise InvalidAnswerProviderError(
            "provider_id 不能为空"
        )

    if not packet.generation_context:
        raise AnswerGenerationError(
            "允许生成的 Packet "
            "缺少 generation_context"
        )

    generated = provider.generate(
        question=packet.question,
        metric_name=packet.metric_name,
        generation_context=(
            packet.generation_context
        ),
        allowed_citation_ids=(
            packet.supporting_citation_ids
        ),
    )

    _validate_generated_citations(
        generated=generated,
        allowed_citation_ids=(
            packet.supporting_citation_ids
        ),
    )

    item_by_citation_id = {
        item.citation.citation_id: item
        for item in packet.supporting_items
    }

    selected_items = tuple(
        item_by_citation_id[
            citation_id
        ]
        for citation_id
        in generated.citation_ids
    )

    citations = tuple(
        item.citation
        for item in selected_items
    )

    return FinancialFactFinalResult(
        status="answered",
        question=packet.question,
        metric_name=packet.metric_name,
        company_id=packet.company_id,
        report_id=packet.report_id,
        fiscal_year=packet.fiscal_year,
        answer_text=generated.answer_text,
        citation_ids=(
            generated.citation_ids
        ),
        citations=citations,
        used_chunk_ids=tuple(
            citation.chunk_id
            for citation in citations
        ),
        generator_id=provider_id,
        refusal_reason=None,
    )