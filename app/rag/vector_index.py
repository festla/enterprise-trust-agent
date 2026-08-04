from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.rag.embedding import (
    EmbeddingProvider,
    FloatArray,
    normalize_embedding_matrix,
    normalize_query_vector,
)
from app.schemas.chunk import Chunk
from app.schemas.embedding import (
    EmbeddingSpec,
)
from app.schemas.retrieval import (
    RetrievalFilter,
    RetrievalHit,
)
from app.rag.retrieval_filtering import (
    matches_retrieval_filter,
)


class VectorIndexError(ValueError):
    """精确向量索引基础异常。"""


class EmptyVectorIndexError(
    VectorIndexError
):
    """不能使用空 Chunk 集合构建索引。"""


class DuplicateIndexedChunkError(
    VectorIndexError
):
    """索引中出现重复 chunk_id 。"""


class EmbeddingSpecMismatchError(
    VectorIndexError
):
    """查询 Provider 与索引模型配置不一致。"""


class InvalidTopKError(
    VectorIndexError
):
    """top_k 参数无效。"""



@dataclass(frozen=True, slots=True)
class ExactVectorIndex:
    """使用精确余弦相似度扫描的内存索引。"""

    chunks: tuple[Chunk, ...]
    vectors: FloatArray
    embedding_spec: EmbeddingSpec

    @classmethod
    def build(
        cls,
        *,
        chunks: tuple[Chunk, ...],
        provider: EmbeddingProvider,
    ) -> ExactVectorIndex:
        """为给定 Chunk 生成归一化向量矩阵。"""

        if not chunks:
            raise EmptyVectorIndexError(
                "至少需要一个 Chunk 才能构建索引"
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise DuplicateIndexedChunkError(
                "索引输入包含重复 chunk_id"
            )

        embedding_inputs = tuple(
            provider.spec.document_prefix
            + chunk.text
            for chunk in chunks
        )

        raw_vectors = (
            provider.embed_documents(
                embedding_inputs
            )
        )

        vectors = normalize_embedding_matrix(
            raw_vectors,
            expected_rows=len(chunks),
            expected_dimension=(
                provider.spec.dimension
            ),
        )

        # 防止外部调用者原地修改索引矩阵。
        vectors.setflags(write=False)

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
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """对过滤后的候选 Chunk 执行精确余弦检索。"""

        if not query.strip():
            raise VectorIndexError(
                "检索问题不能为空"
            )

        if top_k < 1:
            raise InvalidTopKError(
                "top_k 必须大于等于 1"
            )

        if provider.spec != self.embedding_spec:
            raise EmbeddingSpecMismatchError(
                "查询 Provider 的 EmbeddingSpec "
                "与索引不一致"
            )

        active_filters = (
            filters
            if filters is not None
            else RetrievalFilter()
        )

        candidate_indices = [
            index
            for index, chunk
            in enumerate(self.chunks)
            if matches_retrieval_filter(
                chunk=chunk,
                filters=active_filters,
            )
        ]

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

        query_vector = normalize_query_vector(
            raw_query_vector,
            expected_dimension=(
                self.embedding_spec.dimension
            ),
        )

        candidate_matrix = self.vectors[
            candidate_indices
        ]

        # 查询向量与所有候选 Chunk 向量做点积
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
                self.chunks[item[0]].chunk_id,
            ),
        )

        selected = ranked_candidates[
            :top_k
        ]

        hits: list[RetrievalHit] = []

        for rank, (
            chunk_position,
            raw_score,
        ) in enumerate(
            selected,
            start=1,
        ):
            score = float(
                np.clip(
                    raw_score,
                    -1.0,
                    1.0,
                )
            )

            hits.append(
                RetrievalHit.from_chunk(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[
                        chunk_position
                    ],
                )
            )

        return tuple(hits)