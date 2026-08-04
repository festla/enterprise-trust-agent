from __future__ import annotations

import hashlib
from math import log1p

import pytest

from app.rag.bm25 import (
    BM25IndexError,
    DuplicateBM25ChunkError,
    EmptyBM25DocumentError,
    EmptyBM25IndexError,
    EmptyBM25QueryError,
    ExactBM25Index,
    InvalidBM25TopKError,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.chunk import Chunk
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)


REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

PAGE_DATASET_ID = (
    f"page_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'c' * 24}"
)


def build_chunk(
    *,
    suffix: str,
    text: str,
    pdf_page: int,
) -> Chunk:
    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_{suffix * 24}"
        ),
        chunk_dataset_id=CHUNK_DATASET_ID,
        page_dataset_id=PAGE_DATASET_ID,
        company_id="midea_group",
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        pdf_page=pdf_page,
        printed_page=pdf_page,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        content_type=PageContentType.TEXT,
        parse_status=PageParseStatus.SUCCESS,
        chunk_index=pdf_page - 1,
        strategy=ChunkStrategy.FIXED_LENGTH,
        chunker_version=(
            "fixed_length_chunker_v1"
        ),
        source_text_field=(
            "normalized_text"
        ),
        source_start_char=0,
        source_end_char=len(text),
        text=text,
        char_count=len(text),
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    )


def build_chunks() -> tuple[Chunk, ...]:
    return (
        build_chunk(
            suffix="1",
            text="存货会计政策与计量方法",
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text=(
                "合并资产负债表存货金额"
                "为12345元"
            ),
            pdf_page=2,
        ),
        build_chunk(
            suffix="3",
            text=(
                "合并利润表营业收入金额"
                "为98765元"
            ),
            pdf_page=3,
        ),
    )


def test_rank_exact_financial_fact_first(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="合并资产负债表存货金额",
        tokenizer=tokenizer,
        top_k=3,
    )

    assert hits[0].pdf_page == 2
    assert hits[0].retriever_type == "bm25"
    assert hits[0].score_type == "bm25"
    assert hits[0].score > 0
    assert [
        hit.rank
        for hit in hits
    ] == [1, 2, 3]


def test_bm25_score_matches_manual_formula(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    chunks = (
        build_chunk(
            suffix="1",
            text="苹果苹果",
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text="苹果",
            pdf_page=2,
        ),
    )

    index = ExactBM25Index.build(
        chunks=chunks,
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="苹果",
        tokenizer=tokenizer,
        top_k=2,
    )

    first = next(
        hit
        for hit in hits
        if hit.pdf_page == 1
    )

    document_count = 2
    document_frequency = 2
    term_count = 2
    document_length = 3
    average_document_length = 2
    k1 = 1.2
    b = 0.75

    idf = log1p(
        (
            document_count
            - document_frequency
            + 0.5
        )
        / (
            document_frequency
            + 0.5
        )
    )

    denominator = (
        term_count
        + k1
        * (
            1
            - b
            + b
            * (
                document_length
                / average_document_length
            )
        )
    )

    expected_score = (
        idf
        * (
            term_count
            * (k1 + 1)
        )
        / denominator
    )

    assert first.score == pytest.approx(
        expected_score
    )


def test_equal_scores_use_stable_chunk_id(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    chunks = (
        build_chunk(
            suffix="2",
            text="营业收入",
            pdf_page=2,
        ),
        build_chunk(
            suffix="1",
            text="营业收入",
            pdf_page=1,
        ),
    )

    index = ExactBM25Index.build(
        chunks=chunks,
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="营业收入",
        tokenizer=tokenizer,
        top_k=2,
    )

    assert [
        hit.chunk_id
        for hit in hits
    ] == sorted(
        hit.chunk_id
        for hit in hits
    )


def test_apply_filter_before_ranking(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="存货金额",
        tokenizer=tokenizer,
        top_k=5,
        filters=RetrievalFilter(
            pdf_pages=(3,),
        ),
    )

    assert len(hits) == 1
    assert hits[0].pdf_page == 3


def test_top_k_larger_than_candidates_is_safe(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="营业收入",
        tokenizer=tokenizer,
        top_k=100,
    )

    assert len(hits) == 3


def test_no_filter_candidate_returns_empty(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    hits = index.search(
        query="营业收入",
        tokenizer=tokenizer,
        filters=RetrievalFilter(
            fiscal_years=(2025,),
        ),
    )

    assert hits == ()


def test_reject_empty_index() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    with pytest.raises(
        EmptyBM25IndexError,
    ):
        ExactBM25Index.build(
            chunks=(),
            tokenizer=tokenizer,
        )


def test_reject_duplicate_chunk_ids(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    chunk = build_chunks()[0]

    with pytest.raises(
        DuplicateBM25ChunkError,
    ):
        ExactBM25Index.build(
            chunks=(chunk, chunk),
            tokenizer=tokenizer,
        )


def test_reject_empty_token_document(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    chunk = build_chunk(
        suffix="4",
        text="！！！",
        pdf_page=4,
    )

    with pytest.raises(
        EmptyBM25DocumentError,
    ):
        ExactBM25Index.build(
            chunks=(chunk,),
            tokenizer=tokenizer,
        )


def test_reject_blank_query() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    with pytest.raises(
        BM25IndexError,
    ):
        index.search(
            query="   ",
            tokenizer=tokenizer,
        )


def test_reject_empty_token_query(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    with pytest.raises(
        EmptyBM25QueryError,
    ):
        index.search(
            query="！！！",
            tokenizer=tokenizer,
        )


def test_reject_invalid_top_k() -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = ExactBM25Index.build(
        chunks=build_chunks(),
        tokenizer=tokenizer,
    )

    with pytest.raises(
        InvalidBM25TopKError,
    ):
        index.search(
            query="存货",
            tokenizer=tokenizer,
            top_k=0,
        )