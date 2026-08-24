from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.rag.competition_dense import (
    CompetitionDenseProviderMismatchError,
    CompetitionExactDenseIndex,
    DuplicateCompetitionDenseChunkError,
    InvalidCompetitionDenseTopKError,
)
from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.schemas.competition_retrieval import (
    CompetitionRetrievalFilter,
)
from app.schemas.embedding import (
    EmbeddingSpec,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        spec: EmbeddingSpec,
        vectors: dict[
            str,
            tuple[float, ...],
        ],
    ) -> None:
        self._spec = spec
        self._vectors = vectors

    @property
    def spec(
        self,
    ) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts,
    ):
        return np.asarray(
            [
                self._vectors[text]
                for text in texts
            ],
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ):
        return np.asarray(
            self._vectors[text],
            dtype=np.float32,
        )


def _embedding_spec(
    *,
    model_version: str = "v1",
) -> EmbeddingSpec:
    return EmbeddingSpec(
        provider="fake",
        model_name="fake-dense-model",
        model_version=model_version,
        dimension=3,
        dtype="float32",
        normalize_embeddings=True,
        query_prefix="query:",
        document_prefix="doc:",
        max_sequence_length=128,
    )


def _chunk(
    *,
    chunk_id: str,
    source_id: str,
    text: str,
    chunk_index: int,
) -> CompetitionTextChunk:
    text_sha256 = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return CompetitionTextChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        doc_id=f"doc_{source_id}",
        source_type="word",
        chunk_index=chunk_index,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=(
                    f"block:{chunk_id}"
                ),
                block_index=chunk_index,
                start_char=0,
                end_char=len(text),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=text_sha256,
    )


def _corpus_chunks():
    return (
        _chunk(
            chunk_id="chunk_capital_main",
            source_id="src_capital",
            text="资本充足率监管要求",
            chunk_index=0,
        ),
        _chunk(
            chunk_id="chunk_liquidity",
            source_id="src_liquidity",
            text="流动性风险管理要求",
            chunk_index=0,
        ),
        _chunk(
            chunk_id="chunk_capital_supplement",
            source_id="src_capital",
            text="资本监管补充要求",
            chunk_index=1,
        ),
    )


def _provider():
    spec = _embedding_spec()

    return FakeEmbeddingProvider(
        spec=spec,
        vectors={
            (
                "doc:"
                "资本充足率监管要求"
            ): (
                1.0,
                0.0,
                0.0,
            ),
            (
                "doc:"
                "流动性风险管理要求"
            ): (
                0.0,
                1.0,
                0.0,
            ),
            (
                "doc:"
                "资本监管补充要求"
            ): (
                0.8,
                0.2,
                0.0,
            ),
            "query:资本充足率": (
                1.0,
                0.0,
                0.0,
            ),
            "query:流动性风险": (
                0.0,
                1.0,
                0.0,
            ),
        },
    )


def test_build_and_search_competition_dense(
) -> None:
    chunks = _corpus_chunks()
    provider = _provider()

    index = (
        CompetitionExactDenseIndex.build(
            chunks=chunks,
            provider=provider,
        )
    )

    hits = index.search(
        query="资本充足率",
        provider=provider,
        top_k=2,
    )

    assert len(hits) == 2

    assert [
        hit.chunk_id
        for hit in hits
    ] == [
        "chunk_capital_main",
        "chunk_capital_supplement",
    ]

    assert [
        hit.rank
        for hit in hits
    ] == [
        1,
        2,
    ]

    assert hits[0].score == pytest.approx(
        1.0
    )

    assert (
        hits[0].retriever_type
        == "dense"
    )

    assert (
        hits[0].score_type
        == "cosine_similarity"
    )


def test_search_supports_source_filter(
) -> None:
    chunks = _corpus_chunks()
    provider = _provider()

    index = (
        CompetitionExactDenseIndex.build(
            chunks=chunks,
            provider=provider,
        )
    )

    hits = index.search(
        query="流动性风险",
        provider=provider,
        top_k=5,
        filters=(
            CompetitionRetrievalFilter(
                source_ids=(
                    "src_liquidity",
                ),
            )
        ),
    )

    assert len(hits) == 1

    assert hits[0].source_id == (
        "src_liquidity"
    )

    assert hits[0].chunk_id == (
        "chunk_liquidity"
    )


def test_build_rejects_duplicate_chunk_ids(
) -> None:
    chunk = _chunk(
        chunk_id="duplicate_chunk",
        source_id="src_a",
        text="测试文本",
        chunk_index=0,
    )

    provider = FakeEmbeddingProvider(
        spec=_embedding_spec(),
        vectors={
            "doc:测试文本": (
                1.0,
                0.0,
                0.0,
            ),
        },
    )

    with pytest.raises(
        DuplicateCompetitionDenseChunkError
    ):
        CompetitionExactDenseIndex.build(
            chunks=(
                chunk,
                chunk,
            ),
            provider=provider,
        )


def test_search_rejects_provider_mismatch(
) -> None:
    chunks = _corpus_chunks()
    provider = _provider()

    index = (
        CompetitionExactDenseIndex.build(
            chunks=chunks,
            provider=provider,
        )
    )

    other_provider = (
        FakeEmbeddingProvider(
            spec=_embedding_spec(
                model_version="v2",
            ),
            vectors={},
        )
    )

    with pytest.raises(
        CompetitionDenseProviderMismatchError
    ):
        index.search(
            query="资本充足率",
            provider=other_provider,
        )


def test_search_rejects_invalid_top_k(
) -> None:
    chunks = _corpus_chunks()
    provider = _provider()

    index = (
        CompetitionExactDenseIndex.build(
            chunks=chunks,
            provider=provider,
        )
    )

    with pytest.raises(
        InvalidCompetitionDenseTopKError
    ):
        index.search(
            query="资本充足率",
            provider=provider,
            top_k=0,
        )