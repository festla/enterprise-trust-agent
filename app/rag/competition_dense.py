from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.rag.embedding import (
    EmbeddingProvider,
    FloatArray,
    normalize_embedding_matrix,
    normalize_query_vector,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_retrieval import (
    CompetitionDenseHit,
    CompetitionRetrievalFilter,
)
from app.schemas.embedding import (
    EmbeddingSpec,
)


class CompetitionDenseError(
    ValueError
):
    """Competition Dense Retriever 基础异常。"""


class EmptyCompetitionDenseIndexError(
    CompetitionDenseError
):
    """不能使用空 Chunk 集合构建 Dense Index。"""


class DuplicateCompetitionDenseChunkError(
    CompetitionDenseError
):
    """Dense Index 中出现重复 chunk_id。"""


class CompetitionDenseProviderMismatchError(
    CompetitionDenseError
):
    """查询 Provider 与 Dense Index 配置不一致。"""


class InvalidCompetitionDenseTopKError(
    CompetitionDenseError
):
    """Dense top_k 参数无效。"""


class EmptyCompetitionDenseQueryError(
    CompetitionDenseError
):
    """Dense 查询不能为空。"""


def _matches_filter(
    *,
    chunk: CompetitionTextChunk,
    filters: CompetitionRetrievalFilter,
) -> bool:
    if (
        filters.source_ids
        and chunk.source_id
        not in filters.source_ids
    ):
        return False

    if (
        filters.doc_ids
        and chunk.doc_id
        not in filters.doc_ids
    ):
        return False

    if (
        filters.source_types
        and chunk.source_type
        not in filters.source_types
    ):
        return False

    if (
        filters.chunk_types
        and chunk.chunk_type
        not in filters.chunk_types
    ):
        return False

    return True


@dataclass(
    frozen=True,
    slots=True,
)
class CompetitionExactDenseIndex:
    """
    Competition Corpus 的精确 Dense Index。

    当前 Frozen Dev Corpus 只有约 2171 chunks，
    因此 6B3 Baseline 阶段直接使用精确余弦扫描，
    暂时没有引入 FAISS / ANN。

    这样可以避免 ANN 近似误差干扰 Dense Baseline。
    """

    chunks: tuple[
        CompetitionTextChunk,
        ...,
    ]

    vectors: FloatArray

    embedding_spec: EmbeddingSpec

    @classmethod
    def build(
        cls,
        *,
        chunks: tuple[
            CompetitionTextChunk,
            ...,
        ],
        provider: EmbeddingProvider,
    ) -> CompetitionExactDenseIndex:
        if not chunks:
            raise (
                EmptyCompetitionDenseIndexError(
                    "至少需要一个 Chunk 才能"
                    "构建 Competition Dense Index"
                )
            )

        chunk_ids = tuple(
            chunk.chunk_id
            for chunk in chunks
        )

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise (
                DuplicateCompetitionDenseChunkError(
                    "Competition Dense 输入"
                    "包含重复 chunk_id"
                )
            )

        embedding_inputs = tuple(
            (
                provider.spec.document_prefix
                + chunk.text
            )
            for chunk in chunks
        )

        raw_vectors = (
            provider.embed_documents(
                embedding_inputs
            )
        )

        vectors = (
            normalize_embedding_matrix(
                raw_vectors,
                expected_rows=len(chunks),
                expected_dimension=(
                    provider.spec.dimension
                ),
            )
        )

        # 防止 Dense Index 构建完成后，
        # 外部代码原地修改向量矩阵。
        vectors.setflags(
            write=False
        )

        return cls(
            chunks=chunks,
            vectors=vectors,
            embedding_spec=provider.spec,
        )

    def search(
        self,
        *,
        query: str,
        provider: EmbeddingProvider,
        top_k: int = 5,
        filters: (
            CompetitionRetrievalFilter
            | None
        ) = None,
    ) -> tuple[
        CompetitionDenseHit,
        ...,
    ]:
        if not query.strip():
            raise (
                EmptyCompetitionDenseQueryError(
                    "Dense 检索问题不能为空"
                )
            )

        if top_k < 1:
            raise (
                InvalidCompetitionDenseTopKError(
                    "top_k 必须大于等于 1"
                )
            )

        if (
            provider.spec
            != self.embedding_spec
        ):
            raise (
                CompetitionDenseProviderMismatchError(
                    "查询 Provider 的 "
                    "EmbeddingSpec 与 "
                    "Dense Index 不一致"
                )
            )

        active_filters = (
            filters
            if filters is not None
            else CompetitionRetrievalFilter()
        )

        candidate_indices = tuple(
            position
            for position, chunk
            in enumerate(self.chunks)
            if _matches_filter(
                chunk=chunk,
                filters=active_filters,
            )
        )

        if not candidate_indices:
            return ()

        query_input = (
            provider.spec.query_prefix
            + query
        )

        raw_query_vector = (
            provider.embed_query(
                query_input
            )
        )

        query_vector = (
            normalize_query_vector(
                raw_query_vector,
                expected_dimension=(
                    self.embedding_spec.dimension
                ),
            )
        )

        candidate_matrix = (
            self.vectors[
                list(candidate_indices)
            ]
        )

        # 文档向量和 Query 向量均已 L2 Normalize，
        # 因此点积就是 Cosine Similarity。
        scores = (
            candidate_matrix
            @ query_vector
        )

        ranked_candidates = sorted(
            zip(
                candidate_indices,
                scores,
                strict=True,
            ),
            key=lambda item: (
                -float(item[1]),
                self.chunks[
                    item[0]
                ].chunk_id,
            ),
        )

        selected = (
            ranked_candidates[
                :top_k
            ]
        )

        hits: list[
            CompetitionDenseHit
        ] = []

        for rank, (
            chunk_position,
            raw_score,
        ) in enumerate(
            selected,
            start=1,
        ):
            # 浮点计算可能产生极小的
            # 1.0000001 / -1.0000001。
            score = float(
                np.clip(
                    raw_score,
                    -1.0,
                    1.0,
                )
            )

            hits.append(
                CompetitionDenseHit(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[
                        chunk_position
                    ],
                )
            )

        return tuple(hits)