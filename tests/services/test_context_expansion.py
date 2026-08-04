from __future__ import annotations

import hashlib

import pytest

from app.schemas.chunk import Chunk
from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
    StatementScope,
    StatementType,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)
from app.services.context_expansion import (
    ContextExpansionError,
    expand_adjacent_page_context,
)


COMPANY_ID = "haier_smart_home"
REPORT_ID = "haier_smart_home_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

OTHER_DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'d' * 24}"
)

PAGE_DATASET_ID = (
    f"page_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'c' * 24}"
)

OTHER_CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'e' * 24}"
)


def build_query(
) -> ComplexRetrievalQueryOutput:
    return ComplexRetrievalQueryOutput(
        query_id="q3",
        semantic_query=(
            "net profit fiscal year 2024"
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        metric_id="net_profit",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        statement_type=(
            StatementType.INCOME_STATEMENT
        ),
        statement_scope=(
            StatementScope.CONSOLIDATED
        ),
    )


def build_chunk(
    *,
    pdf_page: int,
    chunk_index: int,
    text: str | None = None,
    document_id: str = DOCUMENT_ID,
    chunk_dataset_id: str = (
        CHUNK_DATASET_ID
    ),
    chunk_suffix: str | None = None,
) -> Chunk:
    actual_text = (
        text
        if text is not None
        else (
            f"page {pdf_page} "
            f"chunk {chunk_index}"
        )
    )

    suffix = (
        chunk_suffix
        if chunk_suffix is not None
        else f"{chunk_index + 1:024x}"
    )

    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_{suffix}"
        ),
        chunk_dataset_id=(
            chunk_dataset_id
        ),
        page_dataset_id=(
            PAGE_DATASET_ID
        ),
        company_id=COMPANY_ID,
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=document_id,
        page_id=(
            f"{document_id}_page_"
            f"{pdf_page:04d}"
        ),
        pdf_page=pdf_page,
        printed_page=pdf_page - 2,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        content_type=(
            PageContentType.TEXT
        ),
        parse_status=(
            PageParseStatus.SUCCESS
        ),
        chunk_index=chunk_index,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        chunker_version=(
            "fixed_length_chunker_v1"
        ),
        source_text_field=(
            "normalized_text"
        ),
        source_start_char=0,
        source_end_char=len(
            actual_text
        ),
        text=actual_text,
        char_count=len(actual_text),
        text_sha256=(
            hashlib.sha256(
                actual_text.encode("utf-8")
            ).hexdigest()
        ),
        paragraph_start_index=None,
        paragraph_end_index=None,
        section_path=(),
        section_source_page_id=None,
        section_inherited=False,
    )


def build_hit(
    *,
    chunk: Chunk,
    rank: int,
) -> RerankedRetrievalHit:
    reranker_score = (
        2.0 - rank * 0.1
    )

    rrf_score = (
        1.0 / (60 + rank)
        + 1.0 / (60 + rank)
    )

    return RerankedRetrievalHit(
        rank=rank,
        retriever_type=(
            "hybrid_reranker"
        ),
        score_type=(
            "cross_encoder_logit"
        ),
        score=reranker_score,
        chunk_id=chunk.chunk_id,
        chunk_dataset_id=(
            chunk.chunk_dataset_id
        ),
        company_id=chunk.company_id,
        report_id=chunk.report_id,
        fiscal_year=chunk.fiscal_year,
        report_type=chunk.report_type,
        document_id=chunk.document_id,
        page_id=chunk.page_id,
        pdf_page=chunk.pdf_page,
        printed_page=(
            chunk.printed_page
        ),
        mapping_status=(
            chunk.mapping_status
        ),
        chunk_index=chunk.chunk_index,
        strategy=chunk.strategy,
        source_start_char=(
            chunk.source_start_char
        ),
        source_end_char=(
            chunk.source_end_char
        ),
        section_path=chunk.section_path,
        text=chunk.text,
        dense_rank=rank,
        bm25_rank=rank,
        rrf_rank=rank,
        rrf_score=rrf_score,
        reranker_score=(
            reranker_score
        ),
        source_retrievers=(
            "dense",
            "bm25",
        ),
    )


def run_expansion(
    *,
    hits: tuple[
        RerankedRetrievalHit,
        ...,
    ],
    chunks: tuple[Chunk, ...],
    base_top_k: int = 5,
):
    return expand_adjacent_page_context(
        original_query=(
            "What is Haier's net profit?"
        ),
        query=build_query(),
        hits=hits,
        chunks=chunks,
        base_top_k=base_top_k,
    )


def test_expand_previous_and_next_pages(
) -> None:
    previous_chunk = build_chunk(
        pdf_page=120,
        chunk_index=0,
    )

    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    next_chunk = build_chunk(
        pdf_page=122,
        chunk_index=2,
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=base_chunk,
                rank=1,
            ),
        ),
        chunks=(
            next_chunk,
            base_chunk,
            previous_chunk,
        ),
    )

    assert expansion.base_item_count == 1
    assert expansion.expanded_item_count == 2
    assert expansion.total_item_count == 3

    assert tuple(
        item.pdf_page
        for item in expansion.items
    ) == (
        121,
        120,
        122,
    )

    assert tuple(
        item.origin
        for item in expansion.items
    ) == (
        "retrieved",
        "adjacent_page",
        "adjacent_page",
    )

    assert tuple(
        item.page_distance
        for item in expansion.items
    ) == (
        0,
        -1,
        1,
    )

    assert all(
        item.anchor_chunk_id
        == base_chunk.chunk_id
        for item in expansion.items
    )

    assert (
        expansion.duplicate_chunk_count
        == 0
    )

    assert (
        expansion.total_char_count
        == sum(
            len(chunk.text)
            for chunk in (
                previous_chunk,
                base_chunk,
                next_chunk,
            )
        )
    )


def test_same_page_path_replaces_weaker_adjacent_path(
) -> None:
    first_base = build_chunk(
        pdf_page=120,
        chunk_index=20,
    )

    second_base = build_chunk(
        pdf_page=121,
        chunk_index=21,
    )

    target_sibling = build_chunk(
        pdf_page=121,
        chunk_index=22,
        text="net profit 100",
    )

    expansion = expand_adjacent_page_context(
        original_query="net profit",
        query=build_query(),
        hits=(
            build_hit(
                chunk=first_base,
                rank=1,
            ),
            build_hit(
                chunk=second_base,
                rank=2,
            ),
        ),
        chunks=(
            first_base,
            second_base,
            target_sibling,
        ),
        base_top_k=5,
        include_same_page_siblings=True,
    )

    target_item = next(
        item
        for item in expansion.items
        if item.chunk_id
        == target_sibling.chunk_id
    )

    assert expansion.strategy_id == (
        "same_and_adjacent_page_context_v2"
    )
    assert target_item.origin == (
        "same_page_sibling"
    )
    assert target_item.page_distance == 0
    assert target_item.anchor_chunk_id == (
        second_base.chunk_id
    )
    assert target_item.anchor_retrieval_rank == 2
    assert expansion.expanded_chunk_ids.count(
        target_sibling.chunk_id
    ) == 1


def test_include_all_adjacent_page_chunks_in_order(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    later_chunk = build_chunk(
        pdf_page=122,
        chunk_index=4,
    )

    earlier_chunk = build_chunk(
        pdf_page=122,
        chunk_index=3,
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=base_chunk,
                rank=1,
            ),
        ),
        chunks=(
            later_chunk,
            base_chunk,
            earlier_chunk,
        ),
    )

    assert expansion.expanded_chunk_ids == (
        earlier_chunk.chunk_id,
        later_chunk.chunk_id,
    )

    assert tuple(
        item.chunk_index
        for item in expansion.items[1:]
    ) == (
        3,
        4,
    )


def test_deduplicate_overlapping_expansions(
) -> None:
    first_base = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    middle_chunk = build_chunk(
        pdf_page=122,
        chunk_index=2,
    )

    second_base = build_chunk(
        pdf_page=123,
        chunk_index=3,
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=first_base,
                rank=1,
            ),
            build_hit(
                chunk=second_base,
                rank=2,
            ),
        ),
        chunks=(
            first_base,
            middle_chunk,
            second_base,
        ),
    )

    assert expansion.base_item_count == 2
    assert expansion.expanded_item_count == 1

    assert expansion.expanded_chunk_ids == (
        middle_chunk.chunk_id,
    )

    assert (
        expansion.items[2]
        .anchor_chunk_id
        == first_base.chunk_id
    )

    assert (
        expansion.items[2]
        .anchor_retrieval_rank
        == 1
    )

    assert (
        expansion.duplicate_chunk_count
        == 1
    )


def test_retrieved_hit_keeps_base_origin(
) -> None:
    first_base = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    second_base = build_chunk(
        pdf_page=122,
        chunk_index=2,
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=first_base,
                rank=1,
            ),
            build_hit(
                chunk=second_base,
                rank=2,
            ),
        ),
        chunks=(
            first_base,
            second_base,
        ),
    )

    assert expansion.base_item_count == 2
    assert expansion.expanded_item_count == 0

    assert tuple(
        item.origin
        for item in expansion.items
    ) == (
        "retrieved",
        "retrieved",
    )

    assert (
        expansion.duplicate_chunk_count
        == 2
    )


def test_no_adjacent_pages_returns_base_only(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=base_chunk,
                rank=1,
            ),
        ),
        chunks=(base_chunk,),
    )

    assert expansion.base_item_count == 1
    assert expansion.expanded_item_count == 0
    assert expansion.total_item_count == 1
    assert expansion.expanded_char_count == 0


def test_ignore_unrelated_chunk_inventory(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    next_chunk = build_chunk(
        pdf_page=122,
        chunk_index=2,
    )

    unrelated_chunk = build_chunk(
        pdf_page=122,
        chunk_index=90,
        document_id=OTHER_DOCUMENT_ID,
        chunk_dataset_id=(
            OTHER_CHUNK_DATASET_ID
        ),
    )

    expansion = run_expansion(
        hits=(
            build_hit(
                chunk=base_chunk,
                rank=1,
            ),
        ),
        chunks=(
            base_chunk,
            next_chunk,
            unrelated_chunk,
        ),
    )

    assert expansion.expanded_chunk_ids == (
        next_chunk.chunk_id,
    )

    assert unrelated_chunk.chunk_id not in (
        expansion.used_chunk_ids
    )


def test_reject_duplicate_chunk_inventory(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    with pytest.raises(
        ContextExpansionError,
        match="duplicate",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            ),
            chunks=(
                base_chunk,
                base_chunk,
            ),
        )


def test_reject_empty_hits() -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    with pytest.raises(
        ContextExpansionError,
        match="at least one",
    ):
        run_expansion(
            hits=(),
            chunks=(base_chunk,),
        )


def test_reject_noncontinuous_hit_ranks(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    with pytest.raises(
        ContextExpansionError,
        match="ranks",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=2,
                ),
            ),
            chunks=(base_chunk,),
        )


def test_reject_hit_count_above_top_k(
) -> None:
    first_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    second_chunk = build_chunk(
        pdf_page=130,
        chunk_index=2,
    )

    with pytest.raises(
        ContextExpansionError,
        match="base_top_k",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=first_chunk,
                    rank=1,
                ),
                build_hit(
                    chunk=second_chunk,
                    rank=2,
                ),
            ),
            chunks=(
                first_chunk,
                second_chunk,
            ),
            base_top_k=1,
        )


def test_reject_hit_source_text_mismatch(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    mismatched_hit = build_hit(
        chunk=base_chunk,
        rank=1,
    ).model_copy(
        update={
            "text": "different source text",
        }
    )

    with pytest.raises(
        ContextExpansionError,
        match="source Chunk field: text",
    ):
        run_expansion(
            hits=(mismatched_hit,),
            chunks=(base_chunk,),
        )


def test_reject_missing_source_chunk(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    other_chunk = build_chunk(
        pdf_page=130,
        chunk_index=2,
    )

    with pytest.raises(
        ContextExpansionError,
        match="does not exist",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            ),
            chunks=(other_chunk,),
        )


def test_reject_hits_from_multiple_documents(
) -> None:
    first_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    second_chunk = build_chunk(
        pdf_page=122,
        chunk_index=2,
        document_id=OTHER_DOCUMENT_ID,
    )

    with pytest.raises(
        ContextExpansionError,
        match="same document_id",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=first_chunk,
                    rank=1,
                ),
                build_hit(
                    chunk=second_chunk,
                    rank=2,
                ),
            ),
            chunks=(
                first_chunk,
                second_chunk,
            ),
        )


def test_reject_hits_from_multiple_chunk_datasets(
) -> None:
    first_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    second_chunk = build_chunk(
        pdf_page=122,
        chunk_index=2,
        chunk_dataset_id=(
            OTHER_CHUNK_DATASET_ID
        ),
    )

    with pytest.raises(
        ContextExpansionError,
        match="same chunk_dataset_id",
    ):
        run_expansion(
            hits=(
                build_hit(
                    chunk=first_chunk,
                    rank=1,
                ),
                build_hit(
                    chunk=second_chunk,
                    rank=2,
                ),
            ),
            chunks=(
                first_chunk,
                second_chunk,
            ),
        )


def test_reject_blank_original_query(
) -> None:
    base_chunk = build_chunk(
        pdf_page=121,
        chunk_index=1,
    )

    with pytest.raises(
        ContextExpansionError,
        match="original_query",
    ):
        expand_adjacent_page_context(
            original_query="   ",
            query=build_query(),
            hits=(
                build_hit(
                    chunk=base_chunk,
                    rank=1,
                ),
            ),
            chunks=(base_chunk,),
            base_top_k=5,
        )
