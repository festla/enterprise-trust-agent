from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from app.schemas.chunk import Chunk
from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalQueryOutput,
)
from app.schemas.context_expansion import (
    AdjacentPageContextExpansion,
    ContextExpansionItem,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
)


class ContextExpansionError(ValueError):
    """Base error for adjacent-page expansion."""


def expand_adjacent_page_context(
    *,
    original_query: str,
    query: ComplexRetrievalQueryOutput,
    hits: Sequence[
        RerankedRetrievalHit
    ],
    chunks: Sequence[Chunk],
    base_top_k: int,
    include_same_page_siblings: bool = False,
) -> AdjacentPageContextExpansion:
    """
    Expand reranked hits with chunks from immediately
    adjacent PDF pages.

    Version 1 uses a fixed page window of one page:
    - previous PDF page;
    - next PDF page.

    When include_same_page_siblings is true, non-retrieved
    chunks from the anchor page are also candidates. If the
    same chunk is reachable through multiple anchors, the
    nearest page distance and then the best anchor rank win.

    All expanded chunks must come from the same document
    and the same ChunkDataset as the retrieved hits.
    """

    normalized_original_query = (
        original_query.strip()
    )

    if not normalized_original_query:
        raise ContextExpansionError(
            "original_query cannot be empty"
        )

    if base_top_k <= 0:
        raise ContextExpansionError(
            "base_top_k must be greater than 0"
        )

    validated_hits = tuple(
        RerankedRetrievalHit.model_validate(
            hit
        )
        for hit in hits
    )

    if not validated_hits:
        raise ContextExpansionError(
            "at least one retrieved hit is required"
        )

    if len(validated_hits) > base_top_k:
        raise ContextExpansionError(
            "retrieved hit count cannot exceed "
            "base_top_k"
        )

    expected_ranks = tuple(
        range(
            1,
            len(validated_hits) + 1,
        )
    )

    actual_ranks = tuple(
        hit.rank
        for hit in validated_hits
    )

    if actual_ranks != expected_ranks:
        raise ContextExpansionError(
            "retrieved hit ranks must start at 1 "
            "and increase continuously"
        )

    hit_chunk_ids = tuple(
        hit.chunk_id
        for hit in validated_hits
    )

    if len(hit_chunk_ids) != len(
        set(hit_chunk_ids)
    ):
        raise ContextExpansionError(
            "retrieved hits contain duplicate "
            "chunk_id values"
        )

    first_hit = validated_hits[0]

    document_id = first_hit.document_id

    chunk_dataset_id = (
        first_hit.chunk_dataset_id
    )

    for hit in validated_hits:
        _validate_hit_identity(
            query=query,
            hit=hit,
            expected_document_id=(
                document_id
            ),
            expected_chunk_dataset_id=(
                chunk_dataset_id
            ),
        )

    validated_chunks = tuple(
        Chunk.model_validate(chunk)
        for chunk in chunks
    )

    if not validated_chunks:
        raise ContextExpansionError(
            "chunk inventory cannot be empty"
        )

    chunk_by_id: dict[str, Chunk] = {}

    for chunk in validated_chunks:
        if chunk.chunk_id in chunk_by_id:
            raise ContextExpansionError(
                "chunk inventory contains duplicate "
                f"chunk_id: {chunk.chunk_id}"
            )

        chunk_by_id[chunk.chunk_id] = chunk

    for hit in validated_hits:
        source_chunk = chunk_by_id.get(
            hit.chunk_id
        )

        if source_chunk is None:
            raise ContextExpansionError(
                "retrieved hit does not exist in "
                "chunk inventory: "
                f"{hit.chunk_id}"
            )

        _validate_hit_source_chunk(
            hit=hit,
            chunk=source_chunk,
        )

    relevant_chunks = tuple(
        chunk
        for chunk in validated_chunks
        if (
            chunk.document_id == document_id
            and chunk.chunk_dataset_id
            == chunk_dataset_id
        )
    )

    if not relevant_chunks:
        raise ContextExpansionError(
            "no chunks belong to the retrieved "
            "document and ChunkDataset"
        )

    for chunk in relevant_chunks:
        _validate_chunk_identity(
            query=query,
            chunk=chunk,
            expected_document_id=(
                document_id
            ),
            expected_chunk_dataset_id=(
                chunk_dataset_id
            ),
        )

    chunks_by_pdf_page: dict[
        int,
        list[Chunk],
    ] = defaultdict(list)

    for chunk in relevant_chunks:
        chunks_by_pdf_page[
            chunk.pdf_page
        ].append(chunk)

    for page_chunks in (
        chunks_by_pdf_page.values()
    ):
        page_chunks.sort(
            key=lambda chunk: (
                chunk.chunk_index,
                chunk.chunk_id,
            )
        )

    base_items = tuple(
        _build_retrieved_item(
            context_order=index,
            hit=hit,
        )
        for index, hit in enumerate(
            validated_hits,
            start=1,
        )
    )

    expanded_records_by_chunk_id: dict[
        str,
        tuple[
            Chunk,
            RerankedRetrievalHit,
            int,
        ],
    ] = {}

    duplicate_chunk_count = 0

    page_distances = (
        (0, -1, 1)
        if include_same_page_siblings
        else (-1, 1)
    )

    for anchor_hit in validated_hits:
        for page_distance in page_distances:
            candidate_pdf_page = (
                anchor_hit.pdf_page
                + page_distance
            )

            if candidate_pdf_page < 1:
                continue

            page_chunks = (
                chunks_by_pdf_page.get(
                    candidate_pdf_page,
                    (),
                )
            )

            for chunk in page_chunks:
                if chunk.chunk_id in hit_chunk_ids:
                    duplicate_chunk_count += 1
                    continue

                candidate_record = (
                    chunk,
                    anchor_hit,
                    page_distance,
                )

                existing_record = (
                    expanded_records_by_chunk_id.get(
                        chunk.chunk_id
                    )
                )

                if existing_record is None:
                    expanded_records_by_chunk_id[
                        chunk.chunk_id
                    ] = candidate_record
                    continue

                duplicate_chunk_count += 1

                existing_anchor = existing_record[1]
                existing_distance = existing_record[2]

                candidate_key = (
                    abs(page_distance),
                    anchor_hit.rank,
                    page_distance,
                )

                existing_key = (
                    abs(existing_distance),
                    existing_anchor.rank,
                    existing_distance,
                )

                if candidate_key < existing_key:
                    expanded_records_by_chunk_id[
                        chunk.chunk_id
                    ] = candidate_record

    expanded_records = tuple(
        sorted(
            expanded_records_by_chunk_id.values(),
            key=lambda value: (
                value[1].rank,
                abs(value[2]),
                value[2],
                value[0].chunk_index,
                value[0].chunk_id,
            ),
        )
    )

    expanded_items = tuple(
        _build_adjacent_item(
            context_order=(
                len(base_items) + index
            ),
            chunk=chunk,
            anchor_hit=anchor_hit,
            page_distance=page_distance,
        )
        for index, (
            chunk,
            anchor_hit,
            page_distance,
        ) in enumerate(
            expanded_records,
            start=1,
        )
    )

    items = (
        base_items
        + expanded_items
    )

    base_chunk_ids = tuple(
        item.chunk_id
        for item in base_items
    )

    expanded_chunk_ids = tuple(
        item.chunk_id
        for item in expanded_items
    )

    used_chunk_ids = tuple(
        item.chunk_id
        for item in items
    )

    base_char_count = sum(
        item.text_char_count
        for item in base_items
    )

    expanded_char_count = sum(
        item.text_char_count
        for item in expanded_items
    )

    return AdjacentPageContextExpansion(
        schema_version=1,
        strategy_id=(
            (
                "same_and_adjacent_page_context_v2"
                if include_same_page_siblings
                else "adjacent_page_context_v1"
            )
        ),
        query_id=query.query_id,
        original_query=(
            normalized_original_query
        ),
        semantic_query=(
            query.semantic_query
        ),
        company_id=query.company_id,
        report_id=query.report_id,
        fiscal_year=query.fiscal_year,
        report_type=query.report_type,
        document_id=document_id,
        base_top_k=base_top_k,
        page_window=1,
        items=items,
        base_chunk_ids=base_chunk_ids,
        expanded_chunk_ids=(
            expanded_chunk_ids
        ),
        used_chunk_ids=used_chunk_ids,
        base_item_count=len(
            base_items
        ),
        expanded_item_count=len(
            expanded_items
        ),
        total_item_count=len(items),
        duplicate_chunk_count=(
            duplicate_chunk_count
        ),
        base_char_count=(
            base_char_count
        ),
        expanded_char_count=(
            expanded_char_count
        ),
        total_char_count=(
            base_char_count
            + expanded_char_count
        ),
    )


def _validate_hit_identity(
    *,
    query: ComplexRetrievalQueryOutput,
    hit: RerankedRetrievalHit,
    expected_document_id: str,
    expected_chunk_dataset_id: str,
) -> None:
    if hit.company_id != query.company_id:
        raise ContextExpansionError(
            "retrieved hit company_id does not "
            "match query"
        )

    if hit.report_id != query.report_id:
        raise ContextExpansionError(
            "retrieved hit report_id does not "
            "match query"
        )

    if hit.fiscal_year != query.fiscal_year:
        raise ContextExpansionError(
            "retrieved hit fiscal_year does not "
            "match query"
        )

    if hit.report_type != query.report_type:
        raise ContextExpansionError(
            "retrieved hit report_type does not "
            "match query"
        )

    if (
        hit.document_id
        != expected_document_id
    ):
        raise ContextExpansionError(
            "retrieved hits must belong to the "
            "same document_id"
        )

    if (
        hit.chunk_dataset_id
        != expected_chunk_dataset_id
    ):
        raise ContextExpansionError(
            "retrieved hits must belong to the "
            "same chunk_dataset_id"
        )


def _validate_chunk_identity(
    *,
    query: ComplexRetrievalQueryOutput,
    chunk: Chunk,
    expected_document_id: str,
    expected_chunk_dataset_id: str,
) -> None:
    if chunk.company_id != query.company_id:
        raise ContextExpansionError(
            "chunk inventory company_id does not "
            "match query"
        )

    if chunk.report_id != query.report_id:
        raise ContextExpansionError(
            "chunk inventory report_id does not "
            "match query"
        )

    if chunk.fiscal_year != query.fiscal_year:
        raise ContextExpansionError(
            "chunk inventory fiscal_year does not "
            "match query"
        )

    if chunk.report_type != query.report_type:
        raise ContextExpansionError(
            "chunk inventory report_type does not "
            "match query"
        )

    if (
        chunk.document_id
        != expected_document_id
    ):
        raise ContextExpansionError(
            "chunk inventory document_id does not "
            "match retrieved document"
        )

    if (
        chunk.chunk_dataset_id
        != expected_chunk_dataset_id
    ):
        raise ContextExpansionError(
            "chunk inventory chunk_dataset_id "
            "does not match retrieved dataset"
        )


def _validate_hit_source_chunk(
    *,
    hit: RerankedRetrievalHit,
    chunk: Chunk,
) -> None:
    field_names = (
        "chunk_dataset_id",
        "company_id",
        "report_id",
        "fiscal_year",
        "report_type",
        "document_id",
        "page_id",
        "pdf_page",
        "printed_page",
        "chunk_index",
        "text",
    )

    for field_name in field_names:
        if (
            getattr(hit, field_name)
            != getattr(chunk, field_name)
        ):
            raise ContextExpansionError(
                "retrieved hit does not match its "
                "source Chunk field: "
                f"{field_name}"
            )


def _build_retrieved_item(
    *,
    context_order: int,
    hit: RerankedRetrievalHit,
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=context_order,
        origin="retrieved",
        company_id=hit.company_id,
        report_id=hit.report_id,
        fiscal_year=hit.fiscal_year,
        report_type=hit.report_type,
        document_id=hit.document_id,
        page_id=hit.page_id,
        pdf_page=hit.pdf_page,
        printed_page=(
            hit.printed_page
        ),
        chunk_id=hit.chunk_id,
        chunk_index=hit.chunk_index,
        text=hit.text,
        text_char_count=len(
            hit.text
        ),
        retrieval_rank=hit.rank,
        retrieval_score=(
            hit.reranker_score
        ),
        anchor_chunk_id=hit.chunk_id,
        anchor_retrieval_rank=(
            hit.rank
        ),
        page_distance=0,
    )


def _build_adjacent_item(
    *,
    context_order: int,
    chunk: Chunk,
    anchor_hit: RerankedRetrievalHit,
    page_distance: int,
) -> ContextExpansionItem:
    return ContextExpansionItem(
        context_order=context_order,
        origin=(
            "same_page_sibling"
            if page_distance == 0
            else "adjacent_page"
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
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        text_char_count=len(
            chunk.text
        ),
        retrieval_rank=None,
        retrieval_score=None,
        anchor_chunk_id=(
            anchor_hit.chunk_id
        ),
        anchor_retrieval_rank=(
            anchor_hit.rank
        ),
        page_distance=(
            page_distance
        ),
    )
