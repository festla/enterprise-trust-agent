from __future__ import annotations

from typing import (
    NotRequired,
    TypedDict,
)

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
)
from app.schemas.competition_citation import (
    CompetitionCitationBundle,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)
from app.schemas.competition_runtime_result import (
    CompetitionAnsweredResult,
    CompetitionEvidenceReadinessDecision,
    CompetitionRefusedResult,
)
from app.schemas.competition_sufficiency import (
    CompetitionEvidenceSufficiencyAssessment,
)


class CompetitionAgentState(
    TypedDict
):
    """
    Competition Agent 的共享状态。

    输入字段：
        question
        resolved_source_id
        retrieval_hit_count
        evidence_bundle

    后续节点逐步写入：
        readiness
        sufficiency
        answer
        citations
        result
    """

    question: CompetitionQuestion

    resolved_source_id: (
        str
        | None
    )

    retrieval_hit_count: int

    evidence_bundle: (
        CompetitionEvidenceBundle
        | None
    )

    readiness: NotRequired[
        CompetitionEvidenceReadinessDecision
    ]

    sufficiency: NotRequired[
        CompetitionEvidenceSufficiencyAssessment
    ]

    answer: NotRequired[
        CompetitionFinalAnswer
    ]

    citations: NotRequired[
        CompetitionCitationBundle
    ]

    result: NotRequired[
        (
            CompetitionAnsweredResult
            | CompetitionRefusedResult
        )
    ]