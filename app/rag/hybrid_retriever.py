from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from app.rag.bm25 import (
    ExactBM25Index,
)
from app.rag.embedding import (
    EmbeddingProvider,
)
from app.rag.rrf import (
    reciprocal_rank_fusion,
)
from app.rag.tokenization import (
    BM25Tokenizer,
)
from app.rag.vector_index import (
    ExactVectorIndex,
)
from app.schemas.chunk import Chunk
from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
    RRFConfig,
)
from app.schemas.retrieval import (
    RetrievalFilter,
)


class HybridRetrieverError(ValueError):
    """Hybrid Retriever 基础异常。"""


class HybridIndexMismatchError(
    HybridRetrieverError
):
    """Dense 与 BM25 索引来源不一致。"""


class HybridProviderMismatchError(
    HybridRetrieverError
):
    """Embedding Provider 与 Dense Index 不一致。"""


class HybridTokenizerMismatchError(
    HybridRetrieverError
):
    """Tokenizer 与 BM25 Index 不一致。"""


class HybridCandidateMismatchError(
    HybridRetrieverError
):
    """相同过滤条件下两路候选状态不一致。"""


class InvalidHybridQueryError(
    HybridRetrieverError
):
    """Hybrid 查询为空。"""


class InvalidHybridTopKError(
    HybridRetrieverError
):
    """Hybrid top_k 参数无效。"""


def _chunk_identity(
    chunk: Chunk,
) -> dict[str, object]:
    """提取 Chunk 的完整可比较身份。"""

    return chunk.model_dump(
        mode="json"
    )


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """对同一语义查询执行 Dense、BM25 和 RRF。"""

    dense_index: ExactVectorIndex

    bm25_index: ExactBM25Index

    provider: EmbeddingProvider

    tokenizer: BM25Tokenizer

    config: RRFConfig = field(
        default_factory=RRFConfig,
    )

    def __post_init__(self) -> None:
        """在首次检索前验证两套索引和运行配置。"""

        if (
            self.provider.spec
            != self.dense_index.embedding_spec
        ):
            raise HybridProviderMismatchError(
                "Embedding Provider 的配置"
                "与 Dense Index 不一致"
            )

        if (
            self.tokenizer.spec
            != self.bm25_index.tokenizer_spec
        ):
            raise HybridTokenizerMismatchError(
                "TokenizerSpec 与 "
                "BM25 Index 不一致"
            )

        dense_chunks = (
            self.dense_index.chunks
        )

        bm25_chunks = (
            self.bm25_index.chunks
        )

        if (
            len(dense_chunks)
            != len(bm25_chunks)
        ):
            raise HybridIndexMismatchError(
                "Dense 与 BM25 Index 的 "
                "Chunk 数量不一致"
            )

        for position, (
            dense_chunk,
            bm25_chunk,
        ) in enumerate(
            zip(
                dense_chunks,
                bm25_chunks,
                strict=True,
            )
        ):
            if (
                _chunk_identity(dense_chunk)
                != _chunk_identity(bm25_chunk)
            ):
                raise HybridIndexMismatchError(
                    "Dense 与 BM25 Index 的 "
                    "Chunk 来源或顺序不一致："
                    f"position={position}"
                )

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> tuple[
        HybridRetrievalHit,
        ...,
    ]:
        """执行双路召回并通过 RRF 输出 Hybrid 排名。"""

        if not query.strip():
            raise InvalidHybridQueryError(
                "Hybrid 检索问题不能为空"
            )

        if top_k < 1:
            raise InvalidHybridTopKError(
                "top_k 必须大于等于 1"
            )

        dense_hits = (
            self.dense_index.search(
                query=query,
                provider=self.provider,
                top_k=(
                    self.config
                    .dense_candidate_count
                ),
                filters=filters,
            )
        )

        bm25_hits = (
            self.bm25_index.search(
                query=query,
                tokenizer=self.tokenizer,
                top_k=(
                    self.config
                    .bm25_candidate_count
                ),
                filters=filters,
            )
        )

        # Metadata Filter 没有候选属于正常情况。
        if not dense_hits and not bm25_hits:
            return ()

        # 两个索引来自同一 ChunkDataset，
        # 因此相同过滤条件下不应只有一路有候选。
        if not dense_hits or not bm25_hits:
            raise HybridCandidateMismatchError(
                "相同过滤条件下 Dense 与 BM25 "
                "的候选状态不一致"
            )

        return reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            config=self.config,
            top_k=top_k,
        )