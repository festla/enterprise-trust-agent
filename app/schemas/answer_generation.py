from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .evidence_context import (
    EvidenceCitation,
)


class GeneratedFinancialFactAnswer(
    BaseModel
):
    """回答 Provider 返回的结构化内容。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    answer_text: str = Field(
        min_length=1,
    )

    citation_ids: tuple[str, ...]

    @field_validator("citation_ids")
    @classmethod
    def validate_citation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "生成答案必须包含至少一个引用"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "生成答案的引用不能重复"
            )

        for citation_id in value:
            if (
                not citation_id.startswith("E")
                or not citation_id[1:].isdigit()
                or int(citation_id[1:]) < 1
            ):
                raise ValueError(
                    "citation_id 必须使用 E1、E2 "
                    "形式"
                )

        return value


class FinancialFactFinalResult(
    BaseModel
):
    """最终可返回给调用方的回答或拒答结果。"""

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
        "answered",
        "refused",
    ]

    question: str = Field(
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

    answer_text: str = Field(
        min_length=1,
    )

    citation_ids: tuple[str, ...] = ()

    citations: tuple[
        EvidenceCitation,
        ...,
    ] = ()

    used_chunk_ids: tuple[str, ...] = ()

    generator_id: str | None = None

    refusal_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(
        self,
    ) -> Self:
        expected_citation_ids = tuple(
            citation.citation_id
            for citation in self.citations
        )

        if (
            self.citation_ids
            != expected_citation_ids
        ):
            raise ValueError(
                "citation_ids 必须与 citations 一致"
            )

        expected_chunk_ids = tuple(
            citation.chunk_id
            for citation in self.citations
        )

        if (
            self.used_chunk_ids
            != expected_chunk_ids
        ):
            raise ValueError(
                "used_chunk_ids 必须与引用一致"
            )

        if self.status == "answered":
            if not self.citations:
                raise ValueError(
                    "answered 必须包含引用"
                )

            if not self.generator_id:
                raise ValueError(
                    "answered 必须记录 generator_id"
                )

            if self.refusal_reason is not None:
                raise ValueError(
                    "answered 不能包含 refusal_reason"
                )

            return self

        if self.citations:
            raise ValueError(
                "refused 不能包含引用"
            )

        if self.citation_ids:
            raise ValueError(
                "refused 不能包含 citation_ids"
            )

        if self.used_chunk_ids:
            raise ValueError(
                "refused 不能包含 used_chunk_ids"
            )

        if self.generator_id is not None:
            raise ValueError(
                "refused 不能记录 generator_id"
            )

        if not self.refusal_reason:
            raise ValueError(
                "refused 必须包含 refusal_reason"
            )

        return self