from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import log1p
from types import MappingProxyType

from app.rag.tokenization import (
    BM25Tokenizer,
)
from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
)
from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
    CompetitionRetrievalFilter,
)


class CompetitionBM25Error(
    ValueError
):
    """Competition BM25 基础异常。"""


class EmptyCompetitionBM25IndexError(
    CompetitionBM25Error
):
    """不能使用空 Chunk 集合构建索引。"""


class DuplicateCompetitionBM25ChunkError(
    CompetitionBM25Error
):
    """索引中出现重复 chunk_id。"""


class EmptyCompetitionBM25DocumentError(
    CompetitionBM25Error
):
    """Chunk 分词后没有有效 Token。"""


class EmptyCompetitionBM25QueryError(
    CompetitionBM25Error
):
    """查询分词后没有有效 Token。"""


class CompetitionBM25TokenizerMismatchError(
    CompetitionBM25Error
):
    """查询分词器与索引不一致。"""


class InvalidCompetitionBM25TopKError(
    CompetitionBM25Error
):
    """top_k 参数无效。"""


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


@dataclass(frozen=True, slots=True)
class CompetitionExactBM25Index:
    chunks: tuple[
        CompetitionTextChunk,
        ...,
    ]

    term_frequencies: tuple[
        Mapping[str, int],
        ...,
    ]

    document_frequencies: Mapping[
        str,
        int,
    ]

    document_lengths: tuple[
        int,
        ...,
    ]

    average_document_length: float

    tokenizer_spec: BM25TokenizerSpec

    config: BM25Config

    @classmethod
    def build(
        cls,
        *,
        chunks: tuple[
            CompetitionTextChunk,
            ...,
        ],
        tokenizer: BM25Tokenizer,
        config: BM25Config | None = None,
    ) -> CompetitionExactBM25Index:
        if not chunks:
            raise (
                EmptyCompetitionBM25IndexError(
                    "至少需要一个 Chunk 才能"
                    "构建 Competition BM25"
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
                DuplicateCompetitionBM25ChunkError(
                    "Competition BM25 输入"
                    "包含重复 chunk_id"
                )
            )

        active_config = (
            config
            if config is not None
            else BM25Config()
        )

        term_frequency_records: list[
            Mapping[str, int]
        ] = []

        document_lengths: list[
            int
        ] = []

        document_frequency_counter: Counter[
            str
        ] = Counter()

        for chunk in chunks:
            tokens = tokenizer.tokenize(
                chunk.text
            )

            if not tokens:
                raise (
                    EmptyCompetitionBM25DocumentError(
                        "Chunk 分词后没有有效 Token："
                        f"{chunk.chunk_id}"
                    )
                )

            term_frequency = Counter(
                tokens
            )

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
            tokenizer_spec=(
                tokenizer.spec
            ),
            config=active_config,
        )

    def search(
        self,
        *,
        query: str,
        tokenizer: BM25Tokenizer,
        top_k: int = 5,
        filters: (
            CompetitionRetrievalFilter
            | None
        ) = None,
    ) -> tuple[
        CompetitionBM25Hit,
        ...,
    ]:
        if not query.strip():
            raise CompetitionBM25Error(
                "检索问题不能为空"
            )

        if top_k < 1:
            raise (
                InvalidCompetitionBM25TopKError(
                    "top_k 必须大于等于 1"
                )
            )

        if (
            tokenizer.spec
            != self.tokenizer_spec
        ):
            raise (
                CompetitionBM25TokenizerMismatchError(
                    "查询 TokenizerSpec 与"
                    "索引配置不一致"
                )
            )

        query_tokens = tokenizer.tokenize(
            query
        )

        if not query_tokens:
            raise (
                EmptyCompetitionBM25QueryError(
                    "查询分词后没有有效 Token"
                )
            )

        if self.config.unique_query_terms:
            query_terms = tuple(
                dict.fromkeys(
                    query_tokens
                )
            )
        else:
            query_terms = query_tokens

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

        document_count = len(
            self.chunks
        )

        k1 = self.config.k1
        b = self.config.b

        scored_candidates: list[
            tuple[int, float]
        ] = []

        for position in candidate_indices:
            term_frequency = (
                self.term_frequencies[
                    position
                ]
            )

            document_length = (
                self.document_lengths[
                    position
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

            # 可信检索要求：
            # 没有任何词法匹配时不返回伪证据。
            if score > 0:
                scored_candidates.append(
                    (
                        position,
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

        return tuple(
            CompetitionBM25Hit(
                rank=rank,
                score=score,
                chunk=self.chunks[
                    position
                ],
            )
            for rank, (
                position,
                score,
            )
            in enumerate(
                selected,
                start=1,
            )
        )