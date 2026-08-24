from __future__ import annotations

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_answer import (
    CompetitionGeneratedAnswer,
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
from app.services.competition_abstention import (
    assess_competition_evidence_readiness,
    run_competition_answer_with_abstention,
)


class FakeProvider:
    def __init__(
        self,
    ) -> None:
        self.request_count = 0

    @property
    def provider_id(
        self,
    ) -> str:
        return "fake:competition"

    def generate(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> CompetitionGeneratedAnswer:
        self.request_count += 1

        return CompetitionGeneratedAnswer(
            answer="B",
            answer_text=(
                "根据证据，答案为 B。[E1]"
            ),
            citation_ids=(
                "E1",
            ),
        )

class FakeSufficiencyProvider:
    def __init__(
        self,
        *,
        status: str = "sufficient",
        supported_answer: str | None = "B",
    ) -> None:
        self.status = status
        self.supported_answer = (
            supported_answer
        )
        self.request_count = 0

    @property
    def provider_id(
        self,
    ) -> str:
        return "fake:sufficiency"

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
    ) -> (
        CompetitionEvidenceSufficiencyAssessment
    ):
        self.request_count += 1

        if (
            self.status
            == "insufficient"
        ):
            return (
                CompetitionEvidenceSufficiencyAssessment(
                    status="insufficient",
                    supported_answer=None,
                    reason=(
                        "Evidence 不足以"
                        "唯一确定答案"
                    ),
                    citation_ids=(),
                )
            )

        return (
            CompetitionEvidenceSufficiencyAssessment(
                status="sufficient",
                supported_answer=(
                    self.supported_answer
                ),
                reason=(
                    "Evidence 明确支持答案"
                ),
                citation_ids=(
                    "E1",
                ),
            )
        )


class FakeDifferentAnswerProvider(
    FakeProvider
):
    def generate(
        self,
        *,
        question: str,
        options: dict[str, str],
        generation_context: str,
        allowed_citation_ids: tuple[
            str,
            ...
        ],
    ) -> CompetitionGeneratedAnswer:
        self.request_count += 1

        return CompetitionGeneratedAnswer(
            answer="C",
            answer_text=(
                "根据证据，答案为 C。[E1]"
            ),
            citation_ids=(
                "E1",
            ),
        )

def _question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q001",
        source_type="word",
        qa_type="单事实检索",
        question="测试问题？",
        option_a="选项A",
        option_b="选项B",
        option_c="选项C",
        option_d="选项D",
        source_title="测试法规",
        file_label="测试法规.docx",
    )


def _bundle(
    *,
    source_id: str = "src_test",
) -> CompetitionEvidenceBundle:
    evidence = CompetitionEvidence(
        evidence_id=(
            "chunk:test:00001"
        ),
        evidence_type="text",
        source=(
            CompetitionKnowledgeSource(
                source_id=source_id,
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
                article="第一条",
                paragraph_index=0,
            )
        ),
        raw_content=(
            "测试问题的正确答案是选项B。"
        ),
    )

    return CompetitionEvidenceBundle(
        evidences=(
            evidence,
        )
    )


def test_refuses_when_source_unresolved(
) -> None:
    provider = FakeProvider()

    sufficiency_provider = (
        FakeSufficiencyProvider()
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id=None,
            retrieval_hit_count=10,
            evidence_bundle=_bundle(),
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=provider,
        )
    )

    assert result.status == "refused"
    assert (
        result.refusal_code
        == "source_unresolved"
    )
    assert (
        sufficiency_provider.request_count
        == 0
    )

    assert provider.request_count == 0


def test_refuses_when_no_retrieval_hits(
) -> None:
    provider = FakeProvider()

    sufficiency_provider = (
        FakeSufficiencyProvider()
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=0,
            evidence_bundle=None,
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=provider,
        )
    )

    assert result.status == "refused"
    assert (
        result.refusal_code
        == "no_retrieval_hits"
    )
    assert (
        sufficiency_provider.request_count
        == 0
    )

    assert provider.request_count == 0


def test_refuses_when_evidence_source_mismatch(
) -> None:
    provider = FakeProvider()

    sufficiency_provider = (
        FakeSufficiencyProvider()
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=10,
            evidence_bundle=_bundle(
                source_id="src_other"
            ),
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=provider,
        )
    )

    assert result.status == "refused"
    assert (
        result.refusal_code
        == "evidence_source_mismatch"
    )
    assert (
        sufficiency_provider.request_count
        == 0
    )

    assert provider.request_count == 0


def test_ready_path_generates_answer(
) -> None:
    answer_provider = FakeProvider()

    sufficiency_provider = (
        FakeSufficiencyProvider(
            supported_answer="B"
        )
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=5,
            evidence_bundle=_bundle(),
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=(
                answer_provider
            ),
        )
    )

    assert result.status == "answered"

    assert (
        result.answer.answer
        == "B"
    )

    assert (
        result.answer.citation_ids
        == ("E1",)
    )

    assert (
        result.citations
        .citations[0]
        .evidence_id
        == "chunk:test:00001"
    )

    assert (
        sufficiency_provider
        .request_count
        == 1
    )

    assert (
        answer_provider
        .request_count
        == 1
    )

def test_readiness_reports_evidence_count(
) -> None:
    decision = (
        assess_competition_evidence_readiness(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=5,
            evidence_bundle=_bundle(),
        )
    )

    assert (
        decision.status
        == "ready_for_generation"
    )

    assert decision.evidence_count == 1

def test_semantic_insufficient_refuses_without_answer_generation(
) -> None:
    answer_provider = FakeProvider()

    sufficiency_provider = (
        FakeSufficiencyProvider(
            status="insufficient",
            supported_answer=None,
        )
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=5,
            evidence_bundle=_bundle(),
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=(
                answer_provider
            ),
        )
    )

    assert result.status == "refused"

    assert (
        result.refusal_code
        == "semantic_evidence_insufficient"
    )

    assert (
        sufficiency_provider
        .request_count
        == 1
    )

    # 证据不足时绝对不能继续调用 Answer LLM。
    assert (
        answer_provider
        .request_count
        == 0
    )

def test_answer_sufficiency_conflict_refuses(
) -> None:
    answer_provider = (
        FakeDifferentAnswerProvider()
    )

    sufficiency_provider = (
        FakeSufficiencyProvider(
            supported_answer="B"
        )
    )

    result = (
        run_competition_answer_with_abstention(
            question=_question(),
            resolved_source_id="src_test",
            retrieval_hit_count=5,
            evidence_bundle=_bundle(),
            sufficiency_provider=(
                sufficiency_provider
            ),
            answer_provider=(
                answer_provider
            ),
        )
    )

    assert result.status == "refused"

    assert (
        result.refusal_code
        == "answer_sufficiency_conflict"
    )

    assert (
        sufficiency_provider
        .request_count
        == 1
    )

    assert (
        answer_provider
        .request_count
        == 1
    )