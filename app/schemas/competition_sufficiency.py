from __future__ import annotations

from typing import Literal

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


class CompetitionEvidenceSufficiencyAssessment(
    BaseModel
):
    """
    判断当前 Evidence 是否足以唯一确定一个选项。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    status: Literal[
        "sufficient",
        "insufficient",
    ]

    supported_answer: (
        CompetitionAnswerOption
        | None
    ) = None

    reason: str = Field(
        min_length=1,
    )

    citation_ids: tuple[
        str,
        ...
    ] = ()

    @field_validator(
        "citation_ids"
    )
    @classmethod
    def validate_citation_ids(
        cls,
        value: tuple[
            str,
            ...
        ],
    ) -> tuple[
        str,
        ...
    ]:
        if (
            len(value)
            != len(set(value))
        ):
            raise ValueError(
                "citation_ids 不能重复"
            )

        for citation_id in value:
            if (
                not citation_id.startswith("E")
                or not citation_id[1:].isdigit()
                or int(
                    citation_id[1:]
                ) < 1
            ):
                raise ValueError(
                    "citation_id 必须使用 "
                    "E1/E2 形式"
                )

        return value

    @model_validator(
        mode="after"
    )
    def validate_status(
        self,
    ):
        if (
            self.status
            == "sufficient"
        ):
            if (
                self.supported_answer
                is None
            ):
                raise ValueError(
                    "sufficient 必须"
                    "给出 supported_answer"
                )

            if not self.citation_ids:
                raise ValueError(
                    "sufficient 必须"
                    "包含支持 Evidence"
                )

            return self

        if (
            self.supported_answer
            is not None
        ):
            raise ValueError(
                "insufficient 不能"
                "给出 supported_answer"
            )

        return self