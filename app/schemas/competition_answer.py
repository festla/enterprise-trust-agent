from __future__ import annotations

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.competition import (
    CompetitionAnswerOption,
)


class CompetitionGeneratedAnswer(
    BaseModel
):
    """
    LLM Provider 返回的结构化比赛答案。

    注意：
    citation_ids 只能引用 Generation Context
    中暴露给模型的 E1 / E2 / ...。
    """

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

    answer: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
        max_length=20_000,
    )

    citation_ids: tuple[
        str,
        ...
    ] = Field(
        min_length=1,
    )

    @field_validator(
        "citation_ids"
    )
    @classmethod
    def validate_citation_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if (
            len(value)
            != len(set(value))
        ):
            raise ValueError(
                "citation_ids 不能重复"
            )

        for citation_id in value:
            if (
                not citation_id.startswith(
                    "E"
                )
                or not citation_id[
                    1:
                ].isdigit()
                or int(
                    citation_id[1:]
                ) < 1
            ):
                raise ValueError(
                    "citation_id 必须使用 "
                    "E1、E2 等格式"
                )

        return value


class CompetitionFinalAnswer(
    BaseModel
):
    """
    Competition Answer Generator
    的最终结构化输出。

    7C 会继续把 citation_ids / evidence_ids
    映射成完整 Citation。
    """

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

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    answer: CompetitionAnswerOption

    answer_text: str = Field(
        min_length=1,
        max_length=20_000,
    )

    citation_ids: tuple[
        str,
        ...
    ] = Field(
        min_length=1,
    )

    evidence_ids: tuple[
        str,
        ...
    ] = Field(
        min_length=1,
    )

    generator_id: str = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_binding(
        self,
    ) -> Self:
        if (
            len(self.citation_ids)
            != len(
                self.evidence_ids
            )
        ):
            raise ValueError(
                "citation_ids 与 "
                "evidence_ids 数量必须一致"
            )

        if (
            len(self.evidence_ids)
            != len(
                set(
                    self.evidence_ids
                )
            )
        ):
            raise ValueError(
                "evidence_ids 不能重复"
            )

        return self