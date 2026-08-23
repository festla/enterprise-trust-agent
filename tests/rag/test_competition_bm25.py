from __future__ import annotations


import pytest

from app.rag.competition_bm25 import (
    CompetitionExactBM25Index,
    DuplicateCompetitionBM25ChunkError,
    InvalidCompetitionBM25TopKError,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_retrieval import (
    CompetitionRetrievalFilter,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _chunks(
    *,
    source_id: str,
    source_type: str,
    text: str,
    sha_char: str,
    table: bool = False,
):
    extension = (
        "pdf"
        if source_type == "pdf"
        else "docx"
    )

    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title=f"测试来源 {source_id}",
        source_type=source_type,
        relative_path=(
            f"{source_type}/"
            f"{source_id}.{extension}"
        ),
        sha256=sha_char * 64,
    )

    if table:
        block = CompetitionTextBlock(
            block_id=(
                f"block:{source.doc_id}:00000"
            ),
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type="pdf",
            block_index=0,
            block_type="table",
            text=text,
            page=1,
            pdf_bbox=(
                10.0,
                20.0,
                300.0,
                400.0,
            ),
            table_index=0,
            table_rows=(
                (
                    "项目",
                    "披露要求",
                ),
                (
                    "资本充足率",
                    "按季度披露",
                ),
            ),
        )
    else:
        block_type = (
            "page_text"
            if source_type == "pdf"
            else "paragraph"
        )

        block = CompetitionTextBlock(
            block_id=(
                f"block:{source.doc_id}:00000"
            ),
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type=source_type,
            block_index=0,
            block_type=block_type,
            text=text,
            page=(
                1
                if source_type == "pdf"
                else None
            ),
            paragraph_index=(
                0
                if source_type == "word"
                else None
            ),
        )

    document = CompetitionTextDocument(
        source=source,
        blocks=(block,),
    )

    return build_competition_text_chunks(
        document,
        max_chars=900,
    )


def _corpus_chunks():
    capital_chunks = _chunks(
        source_id="src_word_capital",
        source_type="word",
        text=(
            "资本充足率资本充足率"
            "监管要求。"
        ),
        sha_char="a",
    )

    liquidity_chunks = _chunks(
        source_id="src_pdf_liquidity",
        source_type="pdf",
        text="流动性风险管理办法。",
        sha_char="b",
    )

    table_chunks = _chunks(
        source_id="src_pdf_table",
        source_type="pdf",
        text=(
            "项目\t披露要求\n"
            "资本充足率\t按季度披露"
        ),
        sha_char="c",
        table=True,
    )

    return (
        capital_chunks
        + liquidity_chunks
        + table_chunks
    )


def test_build_and_search_competition_bm25(
) -> None:
    chunks = _corpus_chunks()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    hits = index.search(
        query="资本充足率",
        tokenizer=tokenizer,
        top_k=5,
    )

    assert len(hits) == 2

    assert hits[0].source_id == (
        "src_word_capital"
    )

    assert all(
        hit.score > 0
        for hit in hits
    )

    assert [
        hit.rank
        for hit in hits
    ] == [
        1,
        2,
    ]


def test_search_supports_table_filter(
) -> None:
    chunks = _corpus_chunks()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    hits = index.search(
        query="资本充足率",
        tokenizer=tokenizer,
        filters=(
            CompetitionRetrievalFilter(
                chunk_types=(
                    "table",
                )
            )
        ),
    )

    assert len(hits) == 1
    assert hits[0].chunk_type == "table"
    assert hits[0].source_id == (
        "src_pdf_table"
    )


def test_search_supports_source_filter(
) -> None:
    chunks = _corpus_chunks()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    hits = index.search(
        query="资本充足率",
        tokenizer=tokenizer,
        filters=(
            CompetitionRetrievalFilter(
                source_types=(
                    "word",
                )
            )
        ),
    )

    assert len(hits) == 1
    assert hits[0].source_id == (
        "src_word_capital"
    )


def test_search_returns_empty_without_match(
) -> None:
    chunks = _corpus_chunks()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=chunks,
            tokenizer=tokenizer,
        )
    )

    hits = index.search(
        query="天气预报",
        tokenizer=tokenizer,
    )

    assert hits == ()


def test_build_rejects_duplicate_chunk_ids(
) -> None:
    chunk = _corpus_chunks()[0]

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    with pytest.raises(
        DuplicateCompetitionBM25ChunkError,
    ):
        CompetitionExactBM25Index.build(
            chunks=(
                chunk,
                chunk,
            ),
            tokenizer=tokenizer,
        )


def test_search_rejects_invalid_top_k(
) -> None:
    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    index = (
        CompetitionExactBM25Index.build(
            chunks=_corpus_chunks(),
            tokenizer=tokenizer,
        )
    )

    with pytest.raises(
        InvalidCompetitionBM25TopKError,
    ):
        index.search(
            query="资本充足率",
            tokenizer=tokenizer,
            top_k=0,
        )


def test_filter_normalizes_and_rejects_duplicates(
) -> None:
    filters = CompetitionRetrievalFilter(
        source_ids=(
            "src_b",
            "src_a",
        ),
        chunk_types=(
            "table",
            "text",
        ),
    )

    assert filters.source_ids == (
        "src_a",
        "src_b",
    )

    assert filters.chunk_types == (
        "table",
        "text",
    )

    with pytest.raises(
        ValueError,
        match="不能重复",
    ):
        CompetitionRetrievalFilter(
            source_ids=(
                "src_a",
                "src_a",
            )
        )