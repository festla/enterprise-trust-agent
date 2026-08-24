from __future__ import annotations

import pytest

from app.rag.competition_sufficiency import (
    UnauthorizedSufficiencyCitationError,
    assess_competition_semantic_sufficiency,
)
from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_evidence import (
    CompetitionEvidence,
    CompetitionEvidenceBundle,
    CompetitionKnowledgeSource,
    CompetitionTextEvidenceLocation,
)
from app.schemas.competition_sufficiency import (
    CompetitionEvidenceSufficiencyAssessment,
)


class FakeSufficientProvider:
    @property
    def provider_id(
        self,
    ) -> str:
        return "fake:sufficient"

    def assess(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ):
        return (
            CompetitionEvidenceSufficiencyAssessment(
                status="sufficient",
                supported_answer="B",
                reason="E1 明确支持 B",
                citation_ids=("E1",),
            )
        )


class FakeUnauthorizedProvider:
    @property
    def provider_id(
        self,
    ) -> str:
        return "fake:unauthorized"

    def assess(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ):
        return (
            CompetitionEvidenceSufficiencyAssessment(
                status="sufficient",
                supported_answer="B",
                reason="错误引用",
                citation_ids=("E99",),
            )
        )


def _question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q001",
        source_type="word",
        qa_type="单事实检索",
        question="正确选项是什么？",
        option_a="A",
        option_b="B",
        option_c="C",
        option_d="D",
        source_title="测试文档",
        file_label="test.docx",
    )


def _bundle(
) -> CompetitionEvidenceBundle:
    return CompetitionEvidenceBundle(
        evidences=(
            CompetitionEvidence(
                evidence_id=(
                    "chunk:test:00001"
                ),
                evidence_type="text",
                source=(
                    CompetitionKnowledgeSource(
                        source_id="src_test",
                        doc_id="doc_test",
                        title="测试文档",
                        source_type="word",
                        relative_path=(
                            "test.docx"
                        ),
                    )
                ),
                location=(
                    CompetitionTextEvidenceLocation(
                        article="第一条",
                        paragraph_index=0,
                    )
                ),
                raw_content=(
                    "材料明确规定正确内容为 B。"
                ),
            ),
        )
    )


def test_sufficient_evidence(
) -> None:
    result = (
        assess_competition_semantic_sufficiency(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=(
                FakeSufficientProvider()
            ),
        )
    )

    assert result.status == "sufficient"
    assert result.supported_answer == "B"
    assert result.citation_ids == ("E1",)


def test_rejects_unauthorized_citation(
) -> None:
    with pytest.raises(
        UnauthorizedSufficiencyCitationError
    ):
        assess_competition_semantic_sufficiency(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=(
                FakeUnauthorizedProvider()
            ),
        )


def test_insufficient_cannot_have_answer(
) -> None:
    with pytest.raises(
        ValueError
    ):
        CompetitionEvidenceSufficiencyAssessment(
            status="insufficient",
            supported_answer="B",
            reason="证据不足",
            citation_ids=(),
        )


def test_sufficient_requires_citation(
) -> None:
    with pytest.raises(
        ValueError
    ):
        CompetitionEvidenceSufficiencyAssessment(
            status="sufficient",
            supported_answer="B",
            reason="证据充分",
            citation_ids=(),
        )