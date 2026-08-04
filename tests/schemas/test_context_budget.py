from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.context_budget import (
    BudgetedContextSelection,
    ContextBudgetPolicy,
    ContextCandidateAudit,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.schemas.enums import (
    ReportType,
)


COMPANY_ID = "haier_smart_home"
REPORT_ID = "haier_smart_home_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

BASE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'b' * 24}"
)

FIRST_EXPANDED_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'c' * 24}"
)

SECOND_EXPANDED_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'d' * 24}"
)

THIRD_EXPANDED_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'e' * 24}"
)

BASE_TEXT = (
    "consolidated statement header"
)

FIRST_EXPANDED_TEXT = (
    "net profit 19575612501.68"
)

SECOND_EXPANDED_TEXT = (
    "other adjacent statement text"
)

THIRD_EXPANDED_TEXT = (
    "third adjacent statement text"
)


def build_base_item(
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=1,
        origin="retrieved",
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_0121"
        ),
        pdf_page=121,
        printed_page=119,
        chunk_id=BASE_CHUNK_ID,
        chunk_index=10,
        text=BASE_TEXT,
        text_char_count=len(
            BASE_TEXT
        ),
        retrieval_rank=1,
        retrieval_score=1.5,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        page_distance=0,
    )


def build_expanded_item(
    *,
    context_order: int,
    chunk_id: str,
    chunk_index: int,
    text: str,
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=context_order,
        origin="adjacent_page",
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_0122"
        ),
        pdf_page=122,
        printed_page=120,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text,
        text_char_count=len(text),
        retrieval_rank=None,
        retrieval_score=None,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        page_distance=1,
    )


def build_first_expanded_item(
    *,
    context_order: int = 2,
    text: str = (
        FIRST_EXPANDED_TEXT
    ),
) -> ContextExpansionItem:
    return build_expanded_item(
        context_order=context_order,
        chunk_id=(
            FIRST_EXPANDED_CHUNK_ID
        ),
        chunk_index=11,
        text=text,
    )


def build_second_expanded_item(
    *,
    context_order: int = 3,
) -> ContextExpansionItem:
    return build_expanded_item(
        context_order=context_order,
        chunk_id=(
            SECOND_EXPANDED_CHUNK_ID
        ),
        chunk_index=12,
        text=SECOND_EXPANDED_TEXT,
    )


def build_third_expanded_item(
    *,
    context_order: int = 4,
) -> ContextExpansionItem:
    return build_expanded_item(
        context_order=context_order,
        chunk_id=(
            THIRD_EXPANDED_CHUNK_ID
        ),
        chunk_index=13,
        text=THIRD_EXPANDED_TEXT,
    )


def build_context(
    *,
    expanded_items: tuple[
        ContextExpansionItem,
        ...,
    ] = (),
) -> AdjacentPageContextExpansion:
    base_item = build_base_item()

    items = (
        base_item,
        *expanded_items,
    )

    expanded_chunk_ids = tuple(
        item.chunk_id
        for item in expanded_items
    )

    expanded_char_count = sum(
        item.text_char_count
        for item in expanded_items
    )

    return AdjacentPageContextExpansion(
        schema_version=1,
        strategy_id=(
            "adjacent_page_context_v1"
        ),
        query_id="q3",
        original_query=(
            "What is the net profit?"
        ),
        semantic_query=(
            "net profit fiscal year 2024"
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        base_top_k=5,
        page_window=1,
        items=items,
        base_chunk_ids=(
            BASE_CHUNK_ID,
        ),
        expanded_chunk_ids=(
            expanded_chunk_ids
        ),
        used_chunk_ids=tuple(
            item.chunk_id
            for item in items
        ),
        base_item_count=1,
        expanded_item_count=len(
            expanded_items
        ),
        total_item_count=len(items),
        duplicate_chunk_count=0,
        base_char_count=len(
            BASE_TEXT
        ),
        expanded_char_count=(
            expanded_char_count
        ),
        total_char_count=(
            len(BASE_TEXT)
            + expanded_char_count
        ),
    )


def build_candidate(
    *,
    selection_rank: int,
    chunk_id: str,
    text_char_count: int,
    decision: str,
    token_hit_count: int = 2,
    bigram_overlap_count: int = 4,
    query_bigram_recall: float = 0.5,
) -> ContextCandidateAudit:
    return ContextCandidateAudit(
        selection_rank=selection_rank,
        chunk_id=chunk_id,
        pdf_page=122,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        text_char_count=(
            text_char_count
        ),
        token_hit_count=(
            token_hit_count
        ),
        bigram_overlap_count=(
            bigram_overlap_count
        ),
        query_bigram_recall=(
            query_bigram_recall
        ),
        decision=decision,
    )


def build_default_candidates(
) -> tuple[
    ContextCandidateAudit,
    ...,
]:
    return (
        build_candidate(
            selection_rank=1,
            chunk_id=(
                FIRST_EXPANDED_CHUNK_ID
            ),
            text_char_count=len(
                FIRST_EXPANDED_TEXT
            ),
            decision="selected",
        ),
        build_candidate(
            selection_rank=2,
            chunk_id=(
                SECOND_EXPANDED_CHUNK_ID
            ),
            text_char_count=len(
                SECOND_EXPANDED_TEXT
            ),
            decision="item_budget",
            token_hit_count=1,
            bigram_overlap_count=2,
            query_bigram_recall=0.25,
        ),
    )


def build_base_selection(
    **overrides: object,
) -> BudgetedContextSelection:
    values = {
        "schema_version": 1,
        "policy": ContextBudgetPolicy(),
        "query_id": "q3",
        "gate_decision": (
            "base_resolved"
        ),
        "base_fact_ids": (
            "fact_haier_smart_home_"
            "2024_net_profit",
        ),
        "candidate_item_count": 0,
        "candidate_char_count": 0,
        "candidates": (),
        "selected_expanded_chunk_ids": (),
        "dropped_expanded_chunk_ids": (),
        "selected_expanded_item_count": 0,
        "selected_expanded_char_count": 0,
        "selected_context": (
            build_context()
        ),
    }

    values.update(overrides)

    return BudgetedContextSelection(
        **values
    )


def build_expansion_selection(
    **overrides: object,
) -> BudgetedContextSelection:
    first_item = (
        build_first_expanded_item()
    )

    candidates = (
        build_default_candidates()
    )

    values = {
        "schema_version": 1,
        "policy": ContextBudgetPolicy(),
        "query_id": "q3",
        "gate_decision": (
            "expansion_required"
        ),
        "base_fact_ids": (),
        "candidate_item_count": 2,
        "candidate_char_count": (
            len(FIRST_EXPANDED_TEXT)
            + len(SECOND_EXPANDED_TEXT)
        ),
        "candidates": candidates,
        "selected_expanded_chunk_ids": (
            FIRST_EXPANDED_CHUNK_ID,
        ),
        "dropped_expanded_chunk_ids": (
            SECOND_EXPANDED_CHUNK_ID,
        ),
        "selected_expanded_item_count": 1,
        "selected_expanded_char_count": (
            len(FIRST_EXPANDED_TEXT)
        ),
        "selected_context": (
            build_context(
                expanded_items=(
                    first_item,
                )
            )
        ),
    }

    values.update(overrides)

    return BudgetedContextSelection(
        **values
    )


def test_budget_policy_defaults(
) -> None:
    policy = ContextBudgetPolicy()

    assert (
        policy.policy_id
        == (
            "gated_lexical_"
            "adjacent_budget_v1"
        )
    )

    assert (
        policy.expand_only_when_base_unresolved
        is True
    )

    assert (
        policy.max_expanded_items
        == 2
    )

    assert (
        policy.max_expanded_chars
        == 1600
    )


def test_accept_base_resolved_branch(
) -> None:
    selection = (
        build_base_selection()
    )

    assert (
        selection.gate_decision
        == "base_resolved"
    )

    assert (
        selection.base_fact_ids
        == (
            "fact_haier_smart_home_"
            "2024_net_profit",
        )
    )

    assert selection.candidates == ()

    assert (
        selection.selected_context
        .expanded_item_count
        == 0
    )


def test_accept_expansion_required_branch(
) -> None:
    selection = (
        build_expansion_selection()
    )

    assert (
        selection.gate_decision
        == "expansion_required"
    )

    assert (
        selection
        .selected_expanded_chunk_ids
        == (
            FIRST_EXPANDED_CHUNK_ID,
        )
    )

    assert (
        selection
        .dropped_expanded_chunk_ids
        == (
            SECOND_EXPANDED_CHUNK_ID,
        )
    )

    assert (
        selection
        .selected_context
        .expanded_item_count
        == 1
    )


def test_reject_query_id_mismatch(
) -> None:
    with pytest.raises(
        ValidationError,
        match="query_id",
    ):
        build_base_selection(
            query_id="q4",
        )


def test_base_resolved_requires_fact_ids(
) -> None:
    with pytest.raises(
        ValidationError,
        match="base_fact_ids",
    ):
        build_base_selection(
            base_fact_ids=(),
        )


def test_base_resolved_rejects_candidates(
) -> None:
    candidate = build_candidate(
        selection_rank=1,
        chunk_id=(
            FIRST_EXPANDED_CHUNK_ID
        ),
        text_char_count=len(
            FIRST_EXPANDED_TEXT
        ),
        decision="char_budget",
    )

    with pytest.raises(
        ValidationError,
        match="scored candidates",
    ):
        build_base_selection(
            candidate_item_count=1,
            candidate_char_count=len(
                FIRST_EXPANDED_TEXT
            ),
            candidates=(candidate,),
            dropped_expanded_chunk_ids=(
                FIRST_EXPANDED_CHUNK_ID,
            ),
        )


def test_expansion_required_rejects_base_facts(
) -> None:
    with pytest.raises(
        ValidationError,
        match="base_fact_ids",
    ):
        build_expansion_selection(
            base_fact_ids=(
                "fact_unexpected",
            ),
        )


def test_reject_noncontinuous_candidate_ranks(
) -> None:
    first_candidate = build_candidate(
        selection_rank=1,
        chunk_id=(
            FIRST_EXPANDED_CHUNK_ID
        ),
        text_char_count=len(
            FIRST_EXPANDED_TEXT
        ),
        decision="selected",
    )

    second_candidate = build_candidate(
        selection_rank=3,
        chunk_id=(
            SECOND_EXPANDED_CHUNK_ID
        ),
        text_char_count=len(
            SECOND_EXPANDED_TEXT
        ),
        decision="item_budget",
    )

    with pytest.raises(
        ValidationError,
        match="selection_rank",
    ):
        build_expansion_selection(
            candidates=(
                first_candidate,
                second_candidate,
            ),
        )


def test_reject_duplicate_candidate_chunks(
) -> None:
    first_candidate = build_candidate(
        selection_rank=1,
        chunk_id=(
            FIRST_EXPANDED_CHUNK_ID
        ),
        text_char_count=len(
            FIRST_EXPANDED_TEXT
        ),
        decision="selected",
    )

    duplicate_candidate = build_candidate(
        selection_rank=2,
        chunk_id=(
            FIRST_EXPANDED_CHUNK_ID
        ),
        text_char_count=len(
            FIRST_EXPANDED_TEXT
        ),
        decision="item_budget",
    )

    with pytest.raises(
        ValidationError,
        match="duplicate chunk_id",
    ):
        build_expansion_selection(
            candidates=(
                first_candidate,
                duplicate_candidate,
            ),
        )


def test_reject_incorrect_candidate_char_count(
) -> None:
    with pytest.raises(
        ValidationError,
        match="candidate_char_count",
    ):
        build_expansion_selection(
            candidate_char_count=1,
        )


def test_reject_decision_id_mismatch(
) -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "selected_expanded_chunk_ids"
        ),
    ):
        build_expansion_selection(
            selected_expanded_chunk_ids=(
                SECOND_EXPANDED_CHUNK_ID,
            ),
        )


def test_reject_selected_context_id_mismatch(
) -> None:
    second_item = (
        build_second_expanded_item(
            context_order=2,
        )
    )

    with pytest.raises(
        ValidationError,
        match="selected_context",
    ):
        build_expansion_selection(
            selected_context=(
                build_context(
                    expanded_items=(
                        second_item,
                    )
                )
            ),
            selected_expanded_char_count=(
                len(SECOND_EXPANDED_TEXT)
            ),
        )


def test_reject_item_budget_overflow(
) -> None:
    first_item = (
        build_first_expanded_item()
    )

    second_item = (
        build_second_expanded_item()
    )

    third_item = (
        build_third_expanded_item()
    )

    candidates = (
        build_candidate(
            selection_rank=1,
            chunk_id=(
                FIRST_EXPANDED_CHUNK_ID
            ),
            text_char_count=len(
                FIRST_EXPANDED_TEXT
            ),
            decision="selected",
        ),
        build_candidate(
            selection_rank=2,
            chunk_id=(
                SECOND_EXPANDED_CHUNK_ID
            ),
            text_char_count=len(
                SECOND_EXPANDED_TEXT
            ),
            decision="selected",
        ),
        build_candidate(
            selection_rank=3,
            chunk_id=(
                THIRD_EXPANDED_CHUNK_ID
            ),
            text_char_count=len(
                THIRD_EXPANDED_TEXT
            ),
            decision="selected",
        ),
    )

    selected_chars = sum(
        item.text_char_count
        for item in (
            first_item,
            second_item,
            third_item,
        )
    )

    with pytest.raises(
        ValidationError,
        match="item count exceeds",
    ):
        build_expansion_selection(
            candidate_item_count=3,
            candidate_char_count=(
                selected_chars
            ),
            candidates=candidates,
            selected_expanded_chunk_ids=(
                FIRST_EXPANDED_CHUNK_ID,
                SECOND_EXPANDED_CHUNK_ID,
                THIRD_EXPANDED_CHUNK_ID,
            ),
            dropped_expanded_chunk_ids=(),
            selected_expanded_item_count=3,
            selected_expanded_char_count=(
                selected_chars
            ),
            selected_context=(
                build_context(
                    expanded_items=(
                        first_item,
                        second_item,
                        third_item,
                    )
                )
            ),
        )


def test_reject_char_budget_overflow(
) -> None:
    oversized_text = "x" * 1601

    oversized_item = (
        build_first_expanded_item(
            text=oversized_text,
        )
    )

    oversized_candidate = (
        build_candidate(
            selection_rank=1,
            chunk_id=(
                FIRST_EXPANDED_CHUNK_ID
            ),
            text_char_count=1601,
            decision="selected",
        )
    )

    with pytest.raises(
        ValidationError,
        match="char count exceeds",
    ):
        build_expansion_selection(
            candidate_item_count=1,
            candidate_char_count=1601,
            candidates=(
                oversized_candidate,
            ),
            selected_expanded_chunk_ids=(
                FIRST_EXPANDED_CHUNK_ID,
            ),
            dropped_expanded_chunk_ids=(),
            selected_expanded_item_count=1,
            selected_expanded_char_count=1601,
            selected_context=(
                build_context(
                    expanded_items=(
                        oversized_item,
                    )
                )
            ),
        )


def test_reject_unknown_policy_field(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        ContextBudgetPolicy(
            unexpected_field=True,
        )