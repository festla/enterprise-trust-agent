from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class EvidenceCitation(BaseModel):
    """一条可追溯到报告页面和 Chunk 的引用。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    citation_id: str = Field(
        pattern=r"^E[1-9][0-9]*$",
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_title: str = Field(
        min_length=1,
    )

    source_name: str = Field(
        min_length=1,
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    document_id: str = Field(
        min_length=1,
    )

    page_id: str = Field(
        min_length=1,
    )

    pdf_page: int = Field(
        ge=1,
    )

    printed_page: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_id: str = Field(
        min_length=1,
    )

    retrieval_rank: int = Field(
        ge=1,
    )

    retrieval_score: float = Field(
        ge=-1,
        le=1,
    )

    @model_validator(mode="after")
    def validate_source_identity(
        self,
    ) -> Self:
        if not self.document_id.startswith(
            f"doc_{self.report_id}_"
        ):
            raise ValueError(
                "document_id 必须属于 report_id"
            )

        expected_page_id = (
            f"{self.document_id}_page_"
            f"{self.pdf_page:04d}"
        )

        if self.page_id != expected_page_id:
            raise ValueError(
                "page_id 必须由 document_id "
                "和 pdf_page 生成"
            )

        return self


class EvidenceContextItem(BaseModel):
    """Evidence Context 中的一条证据。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    citation: EvidenceCitation

    text: str = Field(
        min_length=1,
    )

    original_char_count: int = Field(
        ge=1,
    )

    included_char_count: int = Field(
        ge=1,
    )

    text_truncated: bool = False

    @model_validator(mode="after")
    def validate_text_counts(
        self,
    ) -> Self:
        if len(self.text) != self.included_char_count:
            raise ValueError(
                "included_char_count "
                "必须等于实际证据文本长度"
            )

        expected_truncated = (
            self.included_char_count
            < self.original_char_count
        )

        if (
            self.text_truncated
            != expected_truncated
        ):
            raise ValueError(
                "text_truncated 必须与 "
                "字符数量关系一致"
            )

        return self


class EvidenceContext(BaseModel):
    """交给后续回答模块的可追溯证据上下文。"""

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

    original_query: str = Field(
        min_length=1,
    )

    semantic_query: str = Field(
        min_length=1,
    )

    company_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_id: str = Field(
        pattern=r"^[a-z0-9_]+$",
    )

    report_title: str = Field(
        min_length=1,
    )

    source_name: str = Field(
        min_length=1,
    )

    fiscal_year: int = Field(
        ge=2000,
        le=2100,
    )

    items: tuple[
        EvidenceContextItem,
        ...,
    ] = ()

    context_text: str = ""

    max_hits: int = Field(
        ge=1,
    )

    max_chars: int = Field(
        ge=1,
    )

    used_chars: int = Field(
        ge=0,
    )

    duplicate_hit_count: int = Field(
        ge=0,
    )

    omitted_hit_count: int = Field(
        ge=0,
    )

    truncated: bool

    used_chunk_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_context(
        self,
    ) -> Self:
        expected_citation_ids = tuple(
            f"E{index}"
            for index in range(
                1,
                len(self.items) + 1,
            )
        )

        actual_citation_ids = tuple(
            item.citation.citation_id
            for item in self.items
        )

        if (
            actual_citation_ids
            != expected_citation_ids
        ):
            raise ValueError(
                "引用编号必须从 E1 连续递增"
            )

        expected_chunk_ids = tuple(
            item.citation.chunk_id
            for item in self.items
        )

        if (
            self.used_chunk_ids
            != expected_chunk_ids
        ):
            raise ValueError(
                "used_chunk_ids 必须与 "
                "Evidence Items 一致"
            )

        if self.used_chars != len(
            self.context_text
        ):
            raise ValueError(
                "used_chars 必须等于 "
                "context_text 实际长度"
            )

        if self.used_chars > self.max_chars:
            raise ValueError(
                "Evidence Context 超过字符预算"
            )

        if bool(self.items) != bool(
            self.context_text
        ):
            raise ValueError(
                "items 与 context_text "
                "必须同时为空或同时非空"
            )

        expected_truncated = (
            self.omitted_hit_count > 0
            or any(
                item.text_truncated
                for item in self.items
            )
        )

        if self.truncated != expected_truncated:
            raise ValueError(
                "truncated 与实际截断状态不一致"
            )

        return self


class EvidenceReadinessResult(BaseModel):
    """Evidence Context 是否可以进入回答生成。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    status: Literal[
        "ready_for_generation",
        "insufficient_evidence",
        "no_retrieval_hits",
    ]

    metric_name: str = Field(
        min_length=1,
    )

    supporting_citation_ids: tuple[
        str,
        ...,
    ] = ()

    reason: str = Field(
        min_length=1,
    )

    context: EvidenceContext

    @model_validator(mode="after")
    def validate_decision(
        self,
    ) -> Self:
        available_ids = {
            item.citation.citation_id
            for item in self.context.items
        }

        if any(
            citation_id not in available_ids
            for citation_id
            in self.supporting_citation_ids
        ):
            raise ValueError(
                "supporting_citation_ids "
                "引用了不存在的证据"
            )

        if (
            self.status
            == "ready_for_generation"
            and not self.supporting_citation_ids
        ):
            raise ValueError(
                "ready_for_generation "
                "必须包含支持证据"
            )

        if self.status == "no_retrieval_hits":
            if self.context.items:
                raise ValueError(
                    "no_retrieval_hits "
                    "不能包含 Evidence Items"
                )

            if self.supporting_citation_ids:
                raise ValueError(
                    "no_retrieval_hits "
                    "不能包含支持引用"
                )

        return self