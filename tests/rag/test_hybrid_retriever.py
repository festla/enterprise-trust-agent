from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from app.rag.bm25 import (
    ExactBM25Index,
)
from app.rag.hybrid_retriever import (
    HybridIndexMismatchError,
    HybridProviderMismatchError,
    HybridRetriever,
    InvalidHybridQueryError,
    InvalidHybridTopKError,
)
from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
from app.rag.vector_index import (
    ExactVectorIndex,
)
from app.schemas.chunk import Chunk
from app.schemas.embedding import (
    EmbeddingSpec,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    ReportType,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
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
            f"chunk_{REPORT_ID}_"
            f"{suffix * 24}"
        ),
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        page_dataset_id=(
            PAGE_DATASET_ID
        ),
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
        content_type=(
            PageContentType.TEXT
        ),
        parse_status=(
            PageParseStatus.SUCCESS
        ),
        chunk_index=pdf_page - 1,
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
            text="营业收入",
            pdf_page=1,
        ),
        build_chunk(
            suffix="2",
            text="资产总计",
            pdf_page=2,
        ),
        build_chunk(
            suffix="3",
            text="经营活动现金流量净额",
            pdf_page=3,
        ),
        build_chunk(
            suffix="4",
            text="资产结构",
            pdf_page=4,
        ),
    )


class FakeEmbeddingProvider:
    """完全确定性的测试 Embedding。"""

    def __init__(
        self,
        *,
        model_version: str = "fake_v1",
    ) -> None:
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake_embedding",
            model_version=model_version,
            dimension=3,
            normalize_embeddings=False,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        vectors = {
            "营业收入": (1.0, 0.0, 0.0),
            "资产总计": (0.9, 0.1, 0.0),
            "经营活动现金流量净额": (
                0.0,
                1.0,
                0.0,
            ),
            "资产结构": (0.0, 0.0, 1.0),
        }

        return np.asarray(
            [
                vectors[text]
                for text in texts
            ],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        if text != "资产总计":
            raise KeyError(text)

        return np.asarray(
            (1.0, 0.0, 0.0),
            dtype=np.float32,
        )


def build_retriever(
    *,
    config: RRFConfig | None = None,
) -> HybridRetriever:
    chunks = build_chunks()

    provider = FakeEmbeddingProvider()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    dense_index = ExactVectorIndex.build(
        chunks=chunks,
        provider=provider,
    )

    bm25_index = ExactBM25Index.build(
        chunks=chunks,
        tokenizer=tokenizer,
    )

    return HybridRetriever(
        dense_index=dense_index,
        bm25_index=bm25_index,
        provider=provider,
        tokenizer=tokenizer,
        config=(
            config
            if config is not None
            else RRFConfig(
                dense_candidate_count=4,
                bm25_candidate_count=4,
            )
        ),
    )


def test_search_and_fuse_real_indices(
) -> None:
    retriever = build_retriever()

    hits = retriever.search(
        query="资产总计",
        top_k=4,
    )

    assert [
        hit.pdf_page
        for hit in hits
    ] == [2, 1, 4, 3]

    first = hits[0]

    assert first.retriever_type == (
        "hybrid_rrf"
    )

    assert first.dense_rank == 2
    assert first.bm25_rank == 1

    assert first.source_retrievers == (
        "dense",
        "bm25",
    )


def test_forward_metadata_filter(
) -> None:
    retriever = build_retriever()

    hits = retriever.search(
        query="资产总计",
        top_k=5,
        filters=RetrievalFilter(
            pdf_pages=(2,),
        ),
    )

    assert len(hits) == 1
    assert hits[0].pdf_page == 2

    assert hits[0].dense_rank == 1
    assert hits[0].bm25_rank == 1


def test_return_empty_when_filter_has_no_candidate(
) -> None:
    retriever = build_retriever()

    hits = retriever.search(
        query="资产总计",
        filters=RetrievalFilter(
            fiscal_years=(2025,),
        ),
    )

    assert hits == ()


def test_apply_candidate_count_limits(
) -> None:
    retriever = build_retriever(
        config=RRFConfig(
            dense_candidate_count=2,
            bm25_candidate_count=2,
        )
    )

    hits = retriever.search(
        query="资产总计",
        top_k=10,
    )

    # Dense Top-2：页面 1、2
    # BM25 Top-2：页面 2、4
    # 去重后共有页面 1、2、4。
    assert {
        hit.pdf_page
        for hit in hits
    } == {1, 2, 4}

    assert len(hits) == 3


def test_reject_mismatched_index_chunks(
) -> None:
    chunks = build_chunks()

    provider = FakeEmbeddingProvider()

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    dense_index = ExactVectorIndex.build(
        chunks=chunks,
        provider=provider,
    )

    bm25_index = ExactBM25Index.build(
        # 故意改变 Chunk 顺序。
        chunks=tuple(reversed(chunks)),
        tokenizer=tokenizer,
    )

    with pytest.raises(
        HybridIndexMismatchError,
        match="来源或顺序不一致",
    ):
        HybridRetriever(
            dense_index=dense_index,
            bm25_index=bm25_index,
            provider=provider,
            tokenizer=tokenizer,
        )


def test_reject_provider_spec_mismatch(
) -> None:
    chunks = build_chunks()

    index_provider = (
        FakeEmbeddingProvider(
            model_version="fake_v1"
        )
    )

    query_provider = (
        FakeEmbeddingProvider(
            model_version="fake_v2"
        )
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    with pytest.raises(
        HybridProviderMismatchError,
        match="Provider",
    ):
        HybridRetriever(
            dense_index=(
                ExactVectorIndex.build(
                    chunks=chunks,
                    provider=index_provider,
                )
            ),
            bm25_index=(
                ExactBM25Index.build(
                    chunks=chunks,
                    tokenizer=tokenizer,
                )
            ),
            provider=query_provider,
            tokenizer=tokenizer,
        )


def test_reject_blank_query() -> None:
    retriever = build_retriever()

    with pytest.raises(
        InvalidHybridQueryError,
    ):
        retriever.search(
            query="   ",
        )


def test_reject_invalid_top_k() -> None:
    retriever = build_retriever()

    with pytest.raises(
        InvalidHybridTopKError,
    ):
        retriever.search(
            query="资产总计",
            top_k=0,
        )