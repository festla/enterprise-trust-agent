from __future__ import annotations

from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
)
from app.schemas.competition_citation import (
    CompetitionCitation,
    CompetitionCitationBundle,
)
from app.schemas.competition_evidence import (
    CompetitionEvidenceBundle,
)


class CompetitionCitationError(
    ValueError
):
    """Competition Citation 基础异常。"""


class CompetitionCitationOutOfRangeError(
    CompetitionCitationError
):
    """E# 超出了 Evidence Bundle 范围。"""


class CompetitionCitationBindingError(
    CompetitionCitationError
):
    """
    Answer 中保存的 evidence_id
    与 E# 实际指向的 Evidence 不一致。
    """


def _citation_index(
    citation_id: str,
) -> int:
    """
    E1 -> 0
    E2 -> 1
    """

    if (
        not citation_id.startswith("E")
        or not citation_id[1:].isdigit()
    ):
        raise CompetitionCitationError(
            "非法 citation_id："
            f"{citation_id}"
        )

    number = int(
        citation_id[1:]
    )

    if number < 1:
        raise CompetitionCitationError(
            "citation_id 必须从 E1 开始"
        )

    return number - 1


def build_competition_citation_bundle(
    *,
    answer: CompetitionFinalAnswer,
    evidence_bundle: (
        CompetitionEvidenceBundle
    ),
) -> CompetitionCitationBundle:
    """
    把 Answer Generator 的 E# 引用
    确定性解析成真正 CompetitionEvidence。

    关键原则：

        E1
         ↓
        EvidenceBundle.evidences[0]
         ↓
        evidence_id / chunk_id
         ↓
        source
         ↓
        page / section / article / paragraph

    不信任 answer.evidence_ids 本身，
    而是重新根据 citation_id
    从 Evidence Bundle 解析并核验。
    """

    citations: list[
        CompetitionCitation
    ] = []

    for (
        citation_id,
        claimed_evidence_id,
    ) in zip(
        answer.citation_ids,
        answer.evidence_ids,
        strict=True,
    ):
        index = _citation_index(
            citation_id
        )

        if (
            index
            >= len(
                evidence_bundle.evidences
            )
        ):
            raise (
                CompetitionCitationOutOfRangeError(
                    "Citation 超出 "
                    "Evidence Bundle 范围："
                    f"{citation_id}"
                )
            )

        evidence = (
            evidence_bundle
            .evidences[index]
        )

        if (
            evidence.evidence_id
            != claimed_evidence_id
        ):
            raise (
                CompetitionCitationBindingError(
                    "Citation / Evidence "
                    "绑定不一致："
                    f"{citation_id}; "
                    "expected="
                    f"{evidence.evidence_id}; "
                    "claimed="
                    f"{claimed_evidence_id}"
                )
            )

        citations.append(
            CompetitionCitation(
                citation_id=(
                    citation_id
                ),
                evidence=evidence,
            )
        )

    if not citations:
        raise CompetitionCitationError(
            "最终 Answer 必须至少"
            "包含一条 Citation"
        )

    return CompetitionCitationBundle(
        case_id=answer.case_id,
        citations=tuple(
            citations
        ),
    )