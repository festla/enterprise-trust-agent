from __future__ import annotations

import numpy as np
import pytest

from app.rag.embedders import (
    EmbeddingModelDimensionMismatchError,
    InvalidEmbeddingBackendOutputError,
    SentenceTransformerEmbeddingProvider,
)
from app.schemas.embedding import EmbeddingSpec


class FakeSentenceTransformerBackend:
    def __init__(
        self,
        *,
        dimension: int = 3,
    ) -> None:
        self.dimension = dimension
        self.max_seq_length = 999
        self.calls: list[
            tuple[list[str], bool]
        ] = []

    def get_embedding_dimension(
        self,
    ) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append(
            (
                list(sentences),
                normalize_embeddings,
            )
        )

        vectors = []

        for index, _ in enumerate(
            sentences,
            start=1,
        ):
            vectors.append(
                [
                    float(index),
                    1.0,
                    0.0,
                ]
            )

        return np.asarray(
            vectors,
            dtype=np.float64,
        )


def build_spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        provider="sentence_transformers",
        model_name="test/model",
        model_version="test_revision",
        dimension=3,
        normalize_embeddings=True,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_sequence_length=128,
    )


def test_provider_sets_max_sequence_length(
) -> None:
    backend = (
        FakeSentenceTransformerBackend()
    )

    SentenceTransformerEmbeddingProvider(
        spec=build_spec(),
        backend=backend,
    )

    assert backend.max_seq_length == 128


def test_embed_documents_returns_float32(
) -> None:
    backend = (
        FakeSentenceTransformerBackend()
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            spec=build_spec(),
            backend=backend,
        )
    )

    vectors = provider.embed_documents(
        (
            "passage: 文档一",
            "passage: 文档二",
        )
    )

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32

    assert backend.calls[0][0] == [
        "passage: 文档一",
        "passage: 文档二",
    ]

    assert backend.calls[0][1] is True


def test_embed_query_returns_one_vector(
) -> None:
    backend = (
        FakeSentenceTransformerBackend()
    )

    provider = (
        SentenceTransformerEmbeddingProvider(
            spec=build_spec(),
            backend=backend,
        )
    )

    vector = provider.embed_query(
        "query: 用户问题"
    )

    assert vector.shape == (3,)
    assert vector.dtype == np.float32


def test_reject_model_dimension_mismatch(
) -> None:
    backend = (
        FakeSentenceTransformerBackend(
            dimension=4
        )
    )

    with pytest.raises(
        EmbeddingModelDimensionMismatchError,
        match="实际向量维度",
    ):
        SentenceTransformerEmbeddingProvider(
            spec=build_spec(),
            backend=backend,
        )


def test_reject_invalid_backend_shape(
) -> None:
    class InvalidBackend(
        FakeSentenceTransformerBackend
    ):
        def encode(
            self,
            sentences: list[str],
            **kwargs: object,
        ) -> object:
            return np.zeros(
                (len(sentences), 2),
                dtype=np.float32,
            )

    provider = (
        SentenceTransformerEmbeddingProvider(
            spec=build_spec(),
            backend=InvalidBackend(),
        )
    )

    with pytest.raises(
        InvalidEmbeddingBackendOutputError,
        match="输出形状",
    ):
        provider.embed_documents(
            ("文档一",)
        )