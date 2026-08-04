from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import log1p
from types import MappingProxyType

from app.rag.retrieval_filtering import (
    matches_retrieval_filter,
)
from app.rag.tokenization import (
    BM25Tokenizer,
)
from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
)
from app.schemas.chunk import Chunk
from app.schemas.retrieval import (
    RetrievalFilter,
    RetrievalHit,
)


class BM25IndexError(ValueError):
    """BM25 内存索引基础异常。"""


class EmptyBM25IndexError(
    BM25IndexError
):
    """不能使用空 Chunk 集合构建索引。"""


class DuplicateBM25ChunkError(
    BM25IndexError
):
    """BM25 索引中出现重复 chunk_id。"""


class EmptyBM25DocumentError(
    BM25IndexError
):
    """Chunk 分词后没有任何有效 Token。"""


class EmptyBM25QueryError(
    BM25IndexError
):
    """查询分词后没有任何有效 Token。"""


class BM25TokenizerSpecMismatchError(
    BM25IndexError
):
    """查询 Tokenizer 与索引配置不一致。"""


class InvalidBM25TopKError(
    BM25IndexError
):
    """BM25 top_k 参数无效。"""


@dataclass(frozen=True, slots=True)
class ExactBM25Index:
    """使用精确 BM25 计算的内存词法索引。"""

    chunks: tuple[Chunk, ...]

    term_frequencies: tuple[
        Mapping[str, int],
        ...,
    ]

    document_frequencies: Mapping[
        str,
        int,
    ]

    document_lengths: tuple[int, ...]

    average_document_length: float

    tokenizer_spec: BM25TokenizerSpec

    config: BM25Config

    @classmethod
    def build(
        cls,
        *,
        chunks: tuple[Chunk, ...],
        tokenizer: BM25Tokenizer,
        config: BM25Config | None = None,
    ) -> ExactBM25Index:
        """为 Chunk 集合构建确定性 BM25 内存索引。"""

        if not chunks:
            raise EmptyBM25IndexError(
                "至少需要一个 Chunk 才能构建 "
                "BM25 索引"
            )

        chunk_ids = tuple(
            chunk.chunk_id
            for chunk in chunks
        )

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):
            raise DuplicateBM25ChunkError(
                "BM25 索引输入包含重复 chunk_id"
            )

        active_config = (
            config
            if config is not None
            else BM25Config()
        )

        term_frequency_records: list[
            Mapping[str, int]
        ] = []

        document_lengths: list[int] = []

        document_frequency_counter: Counter[
            str
        ] = Counter()

        for chunk in chunks:
            tokens = tokenizer.tokenize(
                chunk.text
            )

            if not tokens:
                raise EmptyBM25DocumentError(
                    "Chunk 分词后没有有效 Token："
                    f"{chunk.chunk_id}"
                )

            term_frequency = Counter(tokens)

            term_frequency_records.append(
                MappingProxyType(
                    dict(term_frequency)
                )
            )

            document_lengths.append(
                len(tokens)
            )

            document_frequency_counter.update(
                term_frequency.keys()
            )

        average_document_length = (
            sum(document_lengths)
            / len(document_lengths)
        )

        return cls(
            chunks=chunks,
            term_frequencies=tuple(
                term_frequency_records
            ),
            document_frequencies=(
                MappingProxyType(
                    dict(
                        document_frequency_counter
                    )
                )
            ),
            document_lengths=tuple(
                document_lengths
            ),
            average_document_length=(
                average_document_length
            ),
            tokenizer_spec=tokenizer.spec,
            config=active_config,
        )

    def search(
        self,
        *,
        query: str,
        tokenizer: BM25Tokenizer,
        top_k: int = 5,
        filters: RetrievalFilter | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """对过滤后的候选 Chunk 执行 BM25 排名。"""

        if not query.strip():
            raise BM25IndexError(
                "检索问题不能为空"
            )

        if top_k < 1:
            raise InvalidBM25TopKError(
                "top_k 必须大于等于 1"
            )

        if tokenizer.spec != self.tokenizer_spec:
            raise BM25TokenizerSpecMismatchError(
                "查询 TokenizerSpec 与 "
                "BM25 索引不一致"
            )

        query_tokens = tokenizer.tokenize(
            query
        )

        if not query_tokens:
            raise EmptyBM25QueryError(
                "查询分词后没有有效 Token"
            )

        if self.config.unique_query_terms:
            query_terms = tuple(
                dict.fromkeys(query_tokens)
            )
        else:
            query_terms = query_tokens

        active_filters = (
            filters
            if filters is not None
            else RetrievalFilter()
        )

        candidate_indices = tuple(
            index
            for index, chunk
            in enumerate(self.chunks)
            if matches_retrieval_filter(
                chunk=chunk,
                filters=active_filters,
            )
        )

        if not candidate_indices:
            return ()

        document_count = len(self.chunks)
        k1 = self.config.k1
        b = self.config.b

        scored_candidates: list[
            tuple[int, float]
        ] = []

        for chunk_position in candidate_indices:
            term_frequency = (
                self.term_frequencies[
                    chunk_position
                ]
            )

            document_length = (
                self.document_lengths[
                    chunk_position
                ]
            )

            length_ratio = (
                document_length
                / self.average_document_length
            )

            score = 0.0

            for term in query_terms:
                term_count = (
                    term_frequency.get(
                        term,
                        0,
                    )
                )

                if term_count == 0:
                    continue

                document_frequency = (
                    self.document_frequencies[
                        term
                    ]
                )

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
                        * length_ratio
                    )
                )

                score += (
                    idf
                    * (
                        term_count
                        * (k1 + 1)
                    )
                    / denominator
                )

            scored_candidates.append(
                (
                    chunk_position,
                    score,
                )
            )

        ranked_candidates = sorted(
            scored_candidates,
            key=lambda item: (
                -item[1],
                self.chunks[
                    item[0]
                ].chunk_id,
            ),
        )

        selected = ranked_candidates[
            :top_k
        ]

        hits: list[RetrievalHit] = []

        for rank, (
            chunk_position,
            score,
        ) in enumerate(
            selected,
            start=1,
        ):
            hits.append(
                RetrievalHit.from_chunk(
                    rank=rank,
                    score=score,
                    chunk=self.chunks[
                        chunk_position
                    ],
                    retriever_type="bm25",
                    score_type="bm25",
                )
            )

        return tuple(hits)