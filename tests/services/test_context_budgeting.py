from __future__ import annotations

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.context_budget import (
    ContextBudgetPolicy,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.schemas.enums import (
    ReportType,
    StatementScope,
    StatementType,
)
from app.services.context_budgeting import (
    ContextBudgetingError,
    build_base_only_context,
    score_lexical_context_candidate,
    select_budgeted_adjacent_context,
)
from app.services.context_fact_resolver import (
    ContextFactResolution,
    ResolvedContextFactSupport,
)


COMPANY_ID = "example_corp"
REPORT_ID = "example_corp_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

BASE_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'b' * 24}"
)

FIRST_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'c' * 24}"
)

SECOND_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'d' * 24}"
)

THIRD_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'e' * 24}"
)

FOURTH_CHUNK_ID = (
    f"chunk_{REPORT_ID}_{'f' * 24}"
)

BASE_TEXT = (
    "consolidated balance sheet header"
)

RELEVANT_TEXT = (
    "example corp consolidated balance "
    "sheet total assets"
)

SECOND_TEXT = (
    "consolidated total assets "
    "and liabilities"
)

NOISE_TEXT = (
    "employee training governance "
    "information"
)

OTHER_TEXT = (
    "sales expense and research expense"
)


def build_query(
) -> ComplexRetrievalQueryOutput:
    return ComplexRetrievalQueryOutput(
        query_id="q1",
        semantic_query=(
            "example corp consolidated "
            "balance sheet total assets"
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        metric_id="total_assets",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.BALANCE_SHEET
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
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
            f"{DOCUMENT_ID}_page_0050"
        ),
        pdf_page=50,
        printed_page=48,
        chunk_id=BASE_CHUNK_ID,
        chunk_index=10,
        text=BASE_TEXT,
        text_char_count=len(
            BASE_TEXT
        ),
        retrieval_rank=1,
        retrieval_score=1.8,
        anchor_chunk_id=(
            BASE_CHUNK_ID
        ),
        anchor_retrieval_rank=1,
        page_distance=0,
    )


def build_adjacent_item(
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
            f"{DOCUMENT_ID}_page_0051"
        ),
        pdf_page=51,
        printed_page=49,
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


def build_full_expansion(
    *,
    expanded_items: tuple[
        ContextExpansionItem,
        ...,
    ] | None = None,
    duplicate_chunk_count: int = 0,
) -> AdjacentPageContextExpansion:
    base_item = build_base_item()

    actual_expanded_items = (
        expanded_items
        if expanded_items is not None
        else (
            build_adjacent_item(
                context_order=2,
                chunk_id=(
                    THIRD_CHUNK_ID
                ),
                chunk_index=13,
                text=NOISE_TEXT,
            ),
            build_adjacent_item(
                context_order=3,
                chunk_id=(
                    SECOND_CHUNK_ID
                ),
                chunk_index=12,
                text=SECOND_TEXT,
            ),
            build_adjacent_item(
                context_order=4,
                chunk_id=(
                    FIRST_CHUNK_ID
                ),
                chunk_index=11,
                text=RELEVANT_TEXT,
            ),
            build_adjacent_item(
                context_order=5,
                chunk_id=(
                    FOURTH_CHUNK_ID
                ),
                chunk_index=14,
                text=OTHER_TEXT,
            ),
        )
    )

    items = (
        base_item,
        *actual_expanded_items,
    )

    expanded_char_count = sum(
        item.text_char_count
        for item
        in actual_expanded_items
    )

    return AdjacentPageContextExpansion(
        schema_version=1,
        strategy_id=(
            "adjacent_page_context_v1"
        ),
        query_id="q1",
        original_query=(
            "What are total assets?"
        ),
        semantic_query=(
            "example corp consolidated "
            "balance sheet total assets"
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
        expanded_chunk_ids=tuple(
            item.chunk_id
            for item
            in actual_expanded_items
        ),
        used_chunk_ids=tuple(
            item.chunk_id
            for item in items
        ),
        base_item_count=1,
        expanded_item_count=len(
            actual_expanded_items
        ),
        total_item_count=len(items),
        duplicate_chunk_count=(
            duplicate_chunk_count
        ),
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


def build_empty_resolution(
    *,
    query_id: str = "q1",
) -> ContextFactResolution:
    return ContextFactResolution(
        query_id=query_id,
        supports=(),
    )


def build_base_resolution(
) -> ContextFactResolution:
    return ContextFactResolution(
        query_id="q1",
        supports=(
            ResolvedContextFactSupport(
                fact_id=(
                    "fact_example_corp_"
                    "2024_total_assets"
                ),
                evidence_id=(
                    "evidence_example_corp_"
                    "2024_total_assets"
                ),
                supporting_chunk_id=(
                    BASE_CHUNK_ID
                ),
                supporting_context_order=1,
                supporting_origin=(
                    "retrieved"
                ),
                evidence_match_mode=(
                    "chunk_id"
                ),
            ),
        ),
    )


def test_lexical_score_prefers_relevant_text(
) -> None:
    relevant_score = (
        score_lexical_context_candidate(
            query_text=(
                "consolidated balance "
                "sheet total assets"
            ),
            candidate_text=(
                RELEVANT_TEXT
            ),
        )
    )

    unrelated_score = (
        score_lexical_context_candidate(
            query_text=(
                "consolidated balance "
                "sheet total assets"
            ),
            candidate_text=NOISE_TEXT,
        )
    )

    assert (
        relevant_score[0]
        > unrelated_score[0]
    )

    assert (
        relevant_score[1]
        > unrelated_score[1]
    )

    assert (
        relevant_score[2]
        > unrelated_score[2]
    )


def test_reject_blank_score_query(
) -> None:
    with pytest.raises(
        ContextBudgetingError,
        match="query_text",
    ):
        score_lexical_context_candidate(
            query_text="   ",
            candidate_text=(
                RELEVANT_TEXT
            ),
        )


def test_reject_empty_candidate_text(
) -> None:
    with pytest.raises(
        ContextBudgetingError,
        match="candidate_text",
    ):
        score_lexical_context_candidate(
            query_text="total assets",
            candidate_text="",
        )


def test_build_base_only_context(
) -> None:
    full_expansion = (
        build_full_expansion(
            duplicate_chunk_count=3,
        )
    )

    base_context = (
        build_base_only_context(
            full_expansion
        )
    )

    assert base_context.items == (
        build_base_item(),
    )

    assert base_context.base_chunk_ids == (
        BASE_CHUNK_ID,
    )

    assert (
        base_context.expanded_chunk_ids
        == ()
    )

    assert (
        base_context.expanded_item_count
        == 0
    )

    assert (
        base_context.expanded_char_count
        == 0
    )

    assert (
        base_context.duplicate_chunk_count
        == 0
    )


def test_base_resolution_skips_expansion(
) -> None:
    selection = (
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion()
            ),
            base_resolution=(
                build_base_resolution()
            ),
        )
    )

    assert (
        selection.gate_decision
        == "base_resolved"
    )

    assert selection.candidates == ()

    assert (
        selection
        .selected_expanded_chunk_ids
        == ()
    )

    assert (
        selection.selected_context
        .used_chunk_ids
        == (
            BASE_CHUNK_ID,
        )
    )


def test_rank_and_select_top_two_candidates(
) -> None:
    selection = (
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion()
            ),
            base_resolution=(
                build_empty_resolution()
            ),
        )
    )

    assert (
        selection.gate_decision
        == "expansion_required"
    )

    assert (
        selection
        .selected_expanded_chunk_ids
        == (
            FIRST_CHUNK_ID,
            SECOND_CHUNK_ID,
        )
    )

    assert (
        selection
        .selected_context
        .expanded_chunk_ids
        == (
            FIRST_CHUNK_ID,
            SECOND_CHUNK_ID,
        )
    )

    assert (
        selection
        .selected_expanded_item_count
        == 2
    )

    assert tuple(
        candidate.selection_rank
        for candidate
        in selection.candidates
    ) == (
        1,
        2,
        3,
        4,
    )

    assert tuple(
        candidate.decision
        for candidate
        in selection.candidates
    ) == (
        "selected",
        "selected",
        "item_budget",
        "item_budget",
    )


def test_metric_name_ranks_target_before_generic_query_text(
) -> None:
    generic_item = build_adjacent_item(
        context_order=2,
        chunk_id=SECOND_CHUNK_ID,
        chunk_index=12,
        text=(
            "example corp consolidated balance sheet"
        ),
    )

    metric_item = build_adjacent_item(
        context_order=3,
        chunk_id=FIRST_CHUNK_ID,
        chunk_index=11,
        text="total assets 100",
    )

    selection = select_budgeted_adjacent_context(
        query=build_query(),
        full_expansion=build_full_expansion(
            expanded_items=(
                generic_item,
                metric_item,
            )
        ),
        base_resolution=build_empty_resolution(),
        policy=ContextBudgetPolicy(
            policy_id=(
                "gated_metric_aware_"
                "context_budget_v2"
            ),
            lexical_score_version=(
                "metric_name_query_bigram_v1"
            ),
            max_expanded_items=1,
            max_expanded_chars=1600,
        ),
        metric_hints=("total assets",),
    )

    assert (
        selection.selected_expanded_chunk_ids
        == (FIRST_CHUNK_ID,)
    )
    assert (
        selection.candidates[0]
        .metric_exact_match
        == 1
    )
    assert (
        selection.candidates[0]
        .metric_bigram_recall
        == 1.0
    )


def test_enforce_character_budget(
) -> None:
    policy = ContextBudgetPolicy(
        max_expanded_items=2,
        max_expanded_chars=len(
            RELEVANT_TEXT
        ),
    )

    selection = (
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion()
            ),
            base_resolution=(
                build_empty_resolution()
            ),
            policy=policy,
        )
    )

    assert (
        selection
        .selected_expanded_chunk_ids
        == (
            FIRST_CHUNK_ID,
        )
    )

    assert (
        selection
        .selected_expanded_char_count
        == len(RELEVANT_TEXT)
    )

    assert (
        selection.candidates[1]
        .decision
        == "char_budget"
    )


def test_stable_chunk_id_tie_breaker(
) -> None:
    identical_text = "total assets"

    later_id_item = (
        build_adjacent_item(
            context_order=2,
            chunk_id=SECOND_CHUNK_ID,
            chunk_index=12,
            text=identical_text,
        )
    )

    earlier_id_item = (
        build_adjacent_item(
            context_order=3,
            chunk_id=FIRST_CHUNK_ID,
            chunk_index=11,
            text=identical_text,
        )
    )

    selection = (
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion(
                    expanded_items=(
                        later_id_item,
                        earlier_id_item,
                    )
                )
            ),
            base_resolution=(
                build_empty_resolution()
            ),
            policy=ContextBudgetPolicy(
                max_expanded_items=1,
                max_expanded_chars=1600,
            ),
        )
    )

    assert (
        selection
        .selected_expanded_chunk_ids
        == (
            FIRST_CHUNK_ID,
        )
    )

    assert (
        selection.candidates[0]
        .chunk_id
        == FIRST_CHUNK_ID
    )


def test_expansion_required_without_candidates(
) -> None:
    selection = (
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion(
                    expanded_items=(),
                )
            ),
            base_resolution=(
                build_empty_resolution()
            ),
        )
    )

    assert (
        selection.gate_decision
        == "expansion_required"
    )

    assert selection.candidates == ()

    assert (
        selection
        .selected_context
        .expanded_item_count
        == 0
    )


def test_reject_non_base_resolution_support(
) -> None:
    invalid_resolution = (
        ContextFactResolution(
            query_id="q1",
            supports=(
                ResolvedContextFactSupport(
                    fact_id=(
                        "fact_example_corp_"
                        "2024_total_assets"
                    ),
                    evidence_id=(
                        "evidence_example_corp_"
                        "2024_total_assets"
                    ),
                    supporting_chunk_id=(
                        FIRST_CHUNK_ID
                    ),
                    supporting_context_order=2,
                    supporting_origin=(
                        "adjacent_page"
                    ),
                    evidence_match_mode=(
                        "pdf_page"
                    ),
                ),
            ),
        )
    )

    with pytest.raises(
        ContextBudgetingError,
        match="retrieved items",
    ):
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion()
            ),
            base_resolution=(
                invalid_resolution
            ),
        )


def test_reject_query_expansion_mismatch(
) -> None:
    mismatched_expansion = (
        build_full_expansion().model_copy(
            update={
                "semantic_query": (
                    "different semantic query"
                ),
            }
        )
    )

    with pytest.raises(
        ContextBudgetingError,
        match="semantic_query",
    ):
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                mismatched_expansion
            ),
            base_resolution=(
                build_empty_resolution()
            ),
        )


def test_reject_resolution_query_mismatch(
) -> None:
    with pytest.raises(
        ContextBudgetingError,
        match="resolution query_id",
    ):
        select_budgeted_adjacent_context(
            query=build_query(),
            full_expansion=(
                build_full_expansion()
            ),
            base_resolution=(
                build_empty_resolution(
                    query_id="q2",
                )
            ),
        )
