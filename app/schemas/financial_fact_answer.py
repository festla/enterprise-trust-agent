from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .evidence_context import (
    EvidenceContext,
    EvidenceContextItem,
)


class FinancialFactAnswerPacket(
    BaseModel
):
    """财务事实回答生成前的控制结果。

    ready_for_generation:
        只允许 supporting_items 进入模型。

    refused:
        保留完整 Evidence Context 供审计，
        但 generation_context 必须为空，
        上层不得调用模型。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    status: Literal[
        "ready_for_generation",
        "refused",
    ]

    action: Literal[
        "call_model",
        "return_refusal",
    ]

    question: str = Field(
        min_length=1,
    )

    semantic_query: str = Field(
        min_length=1,
    )

    metric_name: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    message: str = Field(
        min_length=1,
    )

    evidence_context: EvidenceContext

    supporting_items: tuple[
        EvidenceContextItem,
        ...,
    ] = ()

    supporting_citation_ids: tuple[
        str,
        ...,
    ] = ()

    used_chunk_ids: tuple[str, ...] = ()

    generation_context: str = ""

    refusal_reason: str | None = None

    @model_validator(mode="after")
    def validate_packet(
        self,
    ) -> Self:
        """检查控制动作、证据和上下文的一致性。"""

        context = self.evidence_context

        if self.question != context.original_query:
            raise ValueError(
                "question 必须与 "
                "EvidenceContext.original_query 一致"
            )

        if (
            self.semantic_query
            != context.semantic_query
        ):
            raise ValueError(
                "semantic_query 必须与 "
                "EvidenceContext 一致"
            )

        if self.company_id != context.company_id:
            raise ValueError(
                "company_id 必须与 "
                "EvidenceContext 一致"
            )

        if self.report_id != context.report_id:
            raise ValueError(
                "report_id 必须与 "
                "EvidenceContext 一致"
            )

        if self.fiscal_year != context.fiscal_year:
            raise ValueError(
                "fiscal_year 必须与 "
                "EvidenceContext 一致"
            )

        expected_citation_ids = tuple(
            item.citation.citation_id
            for item in self.supporting_items
        )

        if (
            self.supporting_citation_ids
            != expected_citation_ids
        ):
            raise ValueError(
                "supporting_citation_ids 必须与 "
                "supporting_items 一致"
            )

        expected_chunk_ids = tuple(
            item.citation.chunk_id
            for item in self.supporting_items
        )

        if self.used_chunk_ids != expected_chunk_ids:
            raise ValueError(
                "used_chunk_ids 必须与 "
                "supporting_items 一致"
            )

        if (
            len(self.supporting_citation_ids)
            != len(
                set(
                    self.supporting_citation_ids
                )
            )
        ):
            raise ValueError(
                "支持引用不能重复"
            )

        available_items = {
            item.citation.citation_id: item
            for item in context.items
        }

        for item in self.supporting_items:
            citation_id = (
                item.citation.citation_id
            )

            if citation_id not in available_items:
                raise ValueError(
                    "supporting_items 包含 "
                    "EvidenceContext 中不存在的证据"
                )

            if item != available_items[citation_id]:
                raise ValueError(
                    "supporting_items 与原始 "
                    "EvidenceContext 内容不一致"
                )

        if self.status == "ready_for_generation":
            if self.action != "call_model":
                raise ValueError(
                    "ready_for_generation "
                    "必须使用 call_model"
                )

            if not self.supporting_items:
                raise ValueError(
                    "ready_for_generation "
                    "必须包含支持证据"
                )

            if not self.generation_context:
                raise ValueError(
                    "ready_for_generation "
                    "必须包含 generation_context"
                )

            if self.refusal_reason is not None:
                raise ValueError(
                    "ready_for_generation "
                    "不能包含 refusal_reason"
                )

            return self

        if self.action != "return_refusal":
            raise ValueError(
                "refused 必须使用 return_refusal"
            )

        if self.supporting_items:
            raise ValueError(
                "refused 不能包含支持证据"
            )

        if self.supporting_citation_ids:
            raise ValueError(
                "refused 不能包含支持引用"
            )

        if self.used_chunk_ids:
            raise ValueError(
                "refused 不能包含 used_chunk_ids"
            )

        if self.generation_context:
            raise ValueError(
                "refused 不能向模型提供上下文"
            )

        if not self.refusal_reason:
            raise ValueError(
                "refused 必须包含 refusal_reason"
            )

        return self