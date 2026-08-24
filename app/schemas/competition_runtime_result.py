from __future__ import annotations

from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
)
from app.schemas.competition_citation import (
    CompetitionCitationBundle,
)


CompetitionRefusalCode = Literal[
    "source_unresolved",
    "no_retrieval_hits",
    "insufficient_evidence",
    "evidence_source_mismatch",
    "semantic_evidence_insufficient",
    "answer_sufficiency_conflict",
]


class CompetitionEvidenceReadinessDecision(
    BaseModel
):
    """
    Answer Generator 之前的证据充分性决策。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    status: Literal[
        "ready_for_generation",
        "refused",
    ]

    evidence_count: int = Field(
        ge=0,
    )

    refusal_code: (
        CompetitionRefusalCode
        | None
    ) = None

    reason: str = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_decision(
        self,
    ):
        if (
            self.status
            == "ready_for_generation"
        ):
            if self.evidence_count < 1:
                raise ValueError(
                    "ready_for_generation "
                    "必须至少包含一条 Evidence"
                )

            if self.refusal_code is not None:
                raise ValueError(
                    "ready_for_generation "
                    "不能包含 refusal_code"
                )

            return self

        if self.refusal_code is None:
            raise ValueError(
                "refused 必须包含 refusal_code"
            )

        return self


class CompetitionAnsweredResult(
    BaseModel
):
    """
    Competition 最终成功回答结果。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    status: Literal[
        "answered"
    ] = "answered"

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    answer: CompetitionFinalAnswer

    citations: CompetitionCitationBundle

    @model_validator(
        mode="after"
    )
    def validate_identity(
        self,
    ):
        if (
            self.answer.case_id
            != self.case_id
        ):
            raise ValueError(
                "Answer case_id 不一致"
            )

        if (
            self.citations.case_id
            != self.case_id
        ):
            raise ValueError(
                "Citation case_id 不一致"
            )

        return self


class CompetitionRefusedResult(
    BaseModel
):
    """
    Competition 最终拒答结果。

    拒答不是系统错误：
    它表示系统主动判断当前证据不足，
    因而拒绝猜测。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    schema_version: Literal[1] = 1

    status: Literal[
        "refused"
    ] = "refused"

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    answer_text: str = Field(
        min_length=1,
    )

    refusal_code: (
        CompetitionRefusalCode
    )

    refusal_reason: str = Field(
        min_length=1,
    )


CompetitionExecutionResult = Annotated[
    (
        CompetitionAnsweredResult
        | CompetitionRefusedResult
    ),
    Field(
        discriminator="status"
    ),
]