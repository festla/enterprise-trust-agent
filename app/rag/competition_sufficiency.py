from __future__ import annotations

from typing import Protocol

from app.rag.competition_answer_generation import (
    build_competition_generation_context,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)
from app.schemas.competition_sufficiency import (
    CompetitionEvidenceSufficiencyAssessment,
)


class CompetitionEvidenceSufficiencyError(
    ValueError
):
    """Evidence Sufficiency 基础错误。"""


class UnauthorizedSufficiencyCitationError(
    CompetitionEvidenceSufficiencyError
):
    """Judge 引用了未授权 Evidence。"""


class CompetitionEvidenceSufficiencyProvider(
    Protocol
):
    @property
    def provider_id(
        self,
    ) -> str:
        ...

    def assess(
        self,
        *,
        question: str,
        options: dict[
            str,
            str,
        ],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> (
        CompetitionEvidenceSufficiencyAssessment
    ):
        ...


def assess_competition_semantic_sufficiency(
    *,
    question: CompetitionQuestion,
    evidence_bundle: (
        CompetitionEvidenceBundle
    ),
    provider: (
        CompetitionEvidenceSufficiencyProvider
    ),
) -> CompetitionEvidenceSufficiencyAssessment:
    """
    对已经通过确定性 Gate 的 Evidence
    做语义充分性判断。
    """

    (
        generation_context,
        citation_map,
    ) = build_competition_generation_context(
        evidence_bundle
    )

    allowed_citation_ids = tuple(
        citation_map.keys()
    )

    assessment = provider.assess(
        question=question.question,
        options=question.options,
        generation_context=(
            generation_context
        ),
        allowed_citation_ids=(
            allowed_citation_ids
        ),
    )

    unauthorized = (
        set(
            assessment.citation_ids
        )
        - set(
            allowed_citation_ids
        )
    )

    if unauthorized:
        raise (
            UnauthorizedSufficiencyCitationError(
                "Evidence Sufficiency "
                "引用了未授权 Evidence："
                + ", ".join(
                    sorted(
                        unauthorized
                    )
                )
            )
        )

    return assessment