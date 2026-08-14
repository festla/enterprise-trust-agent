from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.trust import (
    AnswerDraft,
    Claim,
    ClaimSupport,
)
from app.services.runtime_completion import (
    RuntimeAnswerGenerator,
)


def test_financial_answer_uses_draft_claim_text(
) -> None:
    draft = AnswerDraft(
        draft_id="draft_test",
        draft_type="financial",
        claims=(
            Claim(
                claim_id="claim_test",
                claim_type="financial_fact",
                claim_text=(
                    "这句话只存在于AnswerDraft中"
                ),
                support=ClaimSupport(
                    fact_ids=(
                        "fact_test",
                    ),
                    citation_ids=(
                        "citation_1",
                    ),
                ),
            ),
        ),
    )

    final_step = SimpleNamespace(
        action="synthesize",
    )

    state = SimpleNamespace(
        answer_draft=draft,

        runtime_plan=SimpleNamespace(
            plan=SimpleNamespace(
                steps=(
                    final_step,
                )
            )
        ),

        resolved_fact_ids=(
            "fact_test",
        ),

        calculation_ids=(),

        evidence_ids=(
            "evidence_test",
        ),

        confidence=1.0,
    )

    generator = (
        RuntimeAnswerGenerator(
            registry_bundle=MagicMock()
        )
    )

    answer = (
        generator
        ._build_financial_answer(
            state
        )
    )

    assert (
        "这句话只存在于AnswerDraft中"
        in answer.answer_text
    )

    assert (
        generator.registry_bundle
        .financial_facts
        .require.call_count
        == 0
    )


def test_document_answer_uses_draft_claim(
) -> None:
    draft = AnswerDraft(
        draft_id="draft_document",
        draft_type="document",
        claims=(
            Claim(
                claim_id="claim_document_test",
                claim_type="document_analysis",
                claim_text=(
                    "公司披露了原材料价格波动风险。"
                ),
                support=ClaimSupport(
                    citation_ids=(
                        "citation_1",
                    ),
                ),
            ),
        ),
    )

    state = SimpleNamespace(
        answer_draft=draft,
        confidence=0.9,
    )

    generator = (
        RuntimeAnswerGenerator(
            registry_bundle=MagicMock()
        )
    )

    answer = (
        generator
        ._build_document_answer(
            state
        )
    )

    assert (
        answer.answer_type
        == "document"
    )

    assert (
        "原材料价格波动风险"
        in answer.answer_text
    )

    assert (
        answer.document_citation_ids
        == (
            "citation_1",
        )
    )