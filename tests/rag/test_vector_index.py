from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from app.rag.embedding import (
    InvalidEmbeddingShapeError,
    InvalidEmbeddingValueError,
)
from app.rag.vector_index import (
    DuplicateIndexedChunkError,
    ExactVectorIndex,
)
from app.schemas.chunk import Chunk
from app.schemas.embedding import EmbeddingSpec
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
    company_id: str = "midea_group",
    fiscal_year: int = 2024,
) -> Chunk:
    text_sha256 = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return Chunk(
        chunk_id=(
            f"chunk_{REPORT_ID}_{suffix * 24}"
        ),
        chunk_dataset_id=CHUNK_DATASET_ID,
        page_dataset_id=PAGE_DATASET_ID,
        company_id=company_id,
        report_id=REPORT_ID,
        fiscal_year=fiscal_year,
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
        chunk_index=0,
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
        text_sha256=text_sha256,
    )


class FakeEmbeddingProvider:
    """完全确定性的测试 Embedding。"""

    def __init__(
        self,
        *,
        document_vectors: dict[
            str,
            tuple[float, ...],
        ],
        query_vectors: dict[
            str,
            tuple[float, ...],
        ],
    ) -> None:
        self._document_vectors = (
            document_vectors
        )

        self._query_vectors = query_vectors

        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake_embedding",
            model_version="fake_v1",
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
        return np.asarray(
            [
                self._document_vectors[text]
                for text in texts
            ],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        return np.asarray(
            self._query_vectors[text],
            dtype=np.float32,
        )


def build_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        document_vectors={
            "营业收入": (1, 0, 0),
            "净利润": (0, 1, 0),
            "经营风险": (0, 0, 1),
        },
        query_vectors={
            "收入是多少": (1, 0, 0),
            "利润是多少": (0, 1, 0),
            "风险有哪些": (0, 0, 1),
        },
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
            text="净利润",
            pdf_page=2,
        ),
        build_chunk(
            suffix="3",
            text="经营风险",
            pdf_page=3,
        ),
    )


def test_rank_by_cosine_similarity() -> None:
    provider = build_provider()

    index = ExactVectorIndex.build(
        chunks=build_chunks(),
        provider=provider,
    )

    hits = index.search(
        query="利润是多少",
        provider=provider,
        top_k=3,
    )

    assert hits[0].text == "净利润"
    assert hits[0].score == pytest.approx(1)
    assert [hit.rank for hit in hits] == [
        1,
        2,
        3,
    ]


def test_equal_scores_use_stable_chunk_id(
) -> None:
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

    provider = FakeEmbeddingProvider(
        document_vectors={
            "营业收入": (1, 0, 0),
        },
        query_vectors={
            "收入是多少": (1, 0, 0),
        },
    )

    index = ExactVectorIndex.build(
        chunks=chunks,
        provider=provider,
    )

    hits = index.search(
        query="收入是多少",
        provider=provider,
        top_k=2,
    )

    assert [
        hit.chunk_id
        for hit in hits
    ] == sorted(
        hit.chunk_id
        for hit in hits
    )


def test_apply_page_filter_before_ranking(
) -> None:
    provider = build_provider()

    index = ExactVectorIndex.build(
        chunks=build_chunks(),
        provider=provider,
    )

    hits = index.search(
        query="收入是多少",
        provider=provider,
        top_k=5,
        filters=RetrievalFilter(
            pdf_pages=(2,),
        ),
    )

    assert len(hits) == 1
    assert hits[0].pdf_page == 2
    assert hits[0].text == "净利润"


def test_top_k_larger_than_candidates_is_safe(
) -> None:
    provider = build_provider()

    index = ExactVectorIndex.build(
        chunks=build_chunks(),
        provider=provider,
    )

    hits = index.search(
        query="收入是多少",
        provider=provider,
        top_k=100,
    )

    assert len(hits) == 3


def test_no_filter_candidate_returns_empty(
) -> None:
    provider = build_provider()

    index = ExactVectorIndex.build(
        chunks=build_chunks(),
        provider=provider,
    )

    hits = index.search(
        query="收入是多少",
        provider=provider,
        filters=RetrievalFilter(
            fiscal_years=(2025,),
        ),
    )

    assert hits == ()


def test_reject_duplicate_chunk_ids() -> None:
    provider = build_provider()
    chunk = build_chunks()[0]

    with pytest.raises(
        DuplicateIndexedChunkError,
    ):
        ExactVectorIndex.build(
            chunks=(chunk, chunk),
            provider=provider,
        )


def test_reject_invalid_embedding_values(
) -> None:
    chunks = build_chunks()

    provider = FakeEmbeddingProvider(
        document_vectors={
            "营业收入": (
                float("nan"),
                0,
                0,
            ),
            "净利润": (0, 1, 0),
            "经营风险": (0, 0, 1),
        },
        query_vectors={
            "收入是多少": (1, 0, 0),
        },
    )

    with pytest.raises(
        InvalidEmbeddingValueError,
    ):
        ExactVectorIndex.build(
            chunks=chunks,
            provider=provider,
        )