from __future__ import annotations

import pytest

from app.schemas.competition_answer import (
    CompetitionFinalAnswer,
)
from app.schemas.competition_evidence import (
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionKnowledgeSource,
    CompetitionTextEvidenceLocation,
)
from app.services.competition_citation import (
    CompetitionCitationBindingError,
    CompetitionCitationOutOfRangeError,
    build_competition_citation_bundle,
)


def _evidence(
    *,
    index: int,
) -> CompetitionEvidence:
    return CompetitionEvidence(
        evidence_id=(
            f"chunk:test:{index:05d}"
        ),
        evidence_type="text",
        source=(
            CompetitionKnowledgeSource(
                source_id="src_test",
                doc_id="doc_test",
                title="测试法规",
                source_type="word",
                relative_path=(
                    "rules/test.docx"
                ),
            )
        ),
        location=(
            CompetitionTextEvidenceLocation(
                section="第一章",
                article=f"第{index}条",
                paragraph_index=index,
            )
        ),
        raw_content=(
            f"第{index}条测试证据"
        ),
    )


def _bundle(
) -> CompetitionEvidenceBundle:
    return CompetitionEvidenceBundle(
        evidences=(
            _evidence(
                index=1
            ),
            _evidence(
                index=2
            ),
            _evidence(
                index=3
            ),
        )
    )


def test_build_citation_bundle(
) -> None:
    answer = CompetitionFinalAnswer(
        case_id="Q001",
        answer="B",
        answer_text=(
            "答案为 B。[E1][E3]"
        ),
        citation_ids=(
            "E1",
            "E3",
        ),
        evidence_ids=(
            "chunk:test:00001",
            "chunk:test:00003",
        ),
        generator_id="fake:test",
    )

    citations = (
        build_competition_citation_bundle(
            answer=answer,
            evidence_bundle=_bundle(),
        )
    )

    assert (
        citations.case_id
        == "Q001"
    )

    assert len(
        citations.citations
    ) == 2

    assert (
        citations.citations[0]
        .citation_id
        == "E1"
    )

    assert (
        citations.citations[0]
        .evidence_id
        == "chunk:test:00001"
    )

    assert (
        citations.citations[1]
        .evidence.location.article
        == "第3条"
    )


def test_rejects_out_of_range_citation(
) -> None:
    answer = CompetitionFinalAnswer(
        case_id="Q001",
        answer="B",
        answer_text=(
            "答案为 B。[E4]"
        ),
        citation_ids=(
            "E4",
        ),
        evidence_ids=(
            "chunk:test:00001",
        ),
        generator_id="fake:test",
    )

    with pytest.raises(
        CompetitionCitationOutOfRangeError
    ):
        build_competition_citation_bundle(
            answer=answer,
            evidence_bundle=_bundle(),
        )


def test_rejects_wrong_evidence_binding(
) -> None:
    answer = CompetitionFinalAnswer(
        case_id="Q001",
        answer="B",
        answer_text=(
            "答案为 B。[E1]"
        ),
        citation_ids=(
            "E1",
        ),
        evidence_ids=(
            "chunk:test:00002",
        ),
        generator_id="fake:test",
    )

    with pytest.raises(
        CompetitionCitationBindingError
    ):
        build_competition_citation_bundle(
            answer=answer,
            evidence_bundle=_bundle(),
        )