from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_evidence import (
    CompetitionEvidence,
)


class CompetitionCitation(
    BaseModel
):
    """
    Competition 最终回答中的一条正式引用。

    citation_id:
        模型可见的 E1 / E2 / ...

    evidence:
        项目内部真实 Evidence，
        可继续追溯到 chunk/source/location。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    citation_id: str = Field(
        pattern=r"^E[1-9][0-9]*$",
    )

    evidence: CompetitionEvidence

    @property
    def evidence_id(
        self,
    ) -> str:
        return (
            self.evidence.evidence_id
        )


class CompetitionCitationBundle(
    BaseModel
):
    """
    一个 Competition Answer
    最终实际使用的全部引用。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    case_id: str = Field(
        pattern=r"^Q[0-9]{3}$",
    )

    citations: tuple[
        CompetitionCitation,
        ...
    ] = Field(
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_citations(
        self,
    ):
        citation_ids = tuple(
            citation.citation_id
            for citation
            in self.citations
        )

        if (
            len(citation_ids)
            != len(
                set(citation_ids)
            )
        ):
            raise ValueError(
                "Citation Bundle "
                "包含重复 citation_id"
            )

        evidence_ids = tuple(
            citation.evidence_id
            for citation
            in self.citations
        )

        if (
            len(evidence_ids)
            != len(
                set(evidence_ids)
            )
        ):
            raise ValueError(
                "Citation Bundle "
                "包含重复 evidence_id"
            )

        return self