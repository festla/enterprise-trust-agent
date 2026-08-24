from __future__ import annotations

import pytest

from app.rag.competition_answer_generation import (
    MissingCompetitionInlineCitationError,
    UnauthorizedCompetitionCitationError,
    _validate_generated_citations,
    build_competition_generation_context,
    generate_competition_answer,
)
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


def _question(
) -> CompetitionQuestion:
    return CompetitionQuestion(
        case_id="Q001",
        source_type="word",
        qa_type="单事实检索",
        question=(
            "核心一级资本工具应满足"
            "以下哪项要求？"
        ),
        option_a="可以间接发行",
        option_b="应当直接发行且实缴",
        option_c="必须具有到期日",
        option_d="可以由关联方担保",
        source_title="资本工具合格标准",
        file_label="test.docx",
    )


def _bundle(
) -> CompetitionEvidenceBundle:
    source = CompetitionKnowledgeSource(
        source_id="src_test",
        doc_id="doc_test",
        title="资本工具合格标准",
        source_type="word",
        relative_path="test.docx",
    )

    evidence = CompetitionEvidence(
        evidence_id="chunk:test:00002",
        evidence_type="text",
        source=source,
        location=(
            CompetitionTextEvidenceLocation(
                section=(
                    "一、核心一级资本工具"
                    "的合格标准"
                ),
                paragraph_index=5,
            )
        ),
        raw_content=(
            "（一）直接发行且实缴的。"
        ),
    )

    return CompetitionEvidenceBundle(
        evidences=(
            evidence,
        )
    )


class FakeProvider:
    @property
    def provider_id(
        self,
    ) -> str:
        return "fake:test"

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
        assert (
            options["B"]
            == "应当直接发行且实缴"
        )

        assert (
            "直接发行且实缴"
            in generation_context
        )

        assert (
            allowed_citation_ids
            == ("E1",)
        )

        return CompetitionGeneratedAnswer(
            answer="B",
            answer_text=(
                "答案为 B。"
                "核心一级资本工具应当"
                "直接发行且实缴。[E1]"
            ),
            citation_ids=(
                "E1",
            ),
        )


def test_build_generation_context(
) -> None:
    context, citation_map = (
        build_competition_generation_context(
            _bundle()
        )
    )

    assert "[E1]" in context

    assert (
        "直接发行且实缴"
        in context
    )

    assert (
        citation_map["E1"]
        .evidence_id
        == "chunk:test:00002"
    )


def test_generate_competition_answer(
) -> None:
    result = (
        generate_competition_answer(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=FakeProvider(),
        )
    )

    assert result.answer == "B"

    assert result.citation_ids == (
        "E1",
    )

    assert result.evidence_ids == (
        "chunk:test:00002",
    )

    assert (
        result.generator_id
        == "fake:test"
    )


class UnauthorizedProvider(
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
        return CompetitionGeneratedAnswer(
            answer="B",
            answer_text=(
                "答案为 B。[E2]"
            ),
            citation_ids=(
                "E2",
            ),
        )


def test_rejects_unauthorized_citation(
) -> None:
    with pytest.raises(
        UnauthorizedCompetitionCitationError
    ):
        generate_competition_answer(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=(
                UnauthorizedProvider()
            ),
        )


class MissingInlineProvider(
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
        return CompetitionGeneratedAnswer(
            answer="B",
            answer_text=(
                "答案为 B。"
            ),
            citation_ids=(
                "E1",
            ),
        )


def test_rejects_missing_inline_citation(
) -> None:
    with pytest.raises(
        MissingCompetitionInlineCitationError
    ):
        generate_competition_answer(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=(
                MissingInlineProvider()
            ),
        )

# 针对10个 answer_text 中的引用与 citation_ids 不一致
class CitationMismatchProvider(
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
        return CompetitionGeneratedAnswer(
            answer="B",
            answer_text=(
                "答案为 B。"
                "相关规定要求直接发行且实缴。[E1]"
            ),
            citation_ids=(
                "E1",
            ),
        )


def test_inline_citations_are_canonical(
) -> None:
    result = (
        generate_competition_answer(
            question=_question(),
            evidence_bundle=_bundle(),
            provider=(
                CitationMismatchProvider()
            ),
        )
    )

    assert result.citation_ids == (
        "E1",
    )

    assert result.evidence_ids == (
        "chunk:test:00002",
    )

def test_inline_citations_canonicalize_mismatch(
) -> None:
    generated = CompetitionGeneratedAnswer(
        answer="B",
        answer_text=(
            "答案为 B。[E1][E3]"
        ),
        citation_ids=(
            "E1",
            "E2",
            "E3",
        ),
    )

    canonical = (
        _validate_generated_citations(
            generated=generated,
            allowed_citation_ids=(
                "E1",
                "E2",
                "E3",
            ),
        )
    )

    assert canonical == (
        "E1",
        "E3",
    )

def test_inline_citation_must_be_authorized(
) -> None:
    generated = CompetitionGeneratedAnswer(
        answer="B",
        answer_text=(
            "答案为 B。[E2]"
        ),
        citation_ids=(
            "E1",
        ),
    )

    with pytest.raises(
        UnauthorizedCompetitionCitationError
    ):
        _validate_generated_citations(
            generated=generated,
            allowed_citation_ids=(
                "E1",
            ),
        )