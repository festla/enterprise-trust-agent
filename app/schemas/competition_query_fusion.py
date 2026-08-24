from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.competition_chunk import (
    CompetitionChunkType,
    CompetitionTextChunk,
)


CompetitionBM25QuerySource = Literal[
    "question_only",
    "question_with_options",
]


class CompetitionBM25QueryFusionConfig(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    fusion_method: Literal[
        "reciprocal_rank_fusion"
    ] = "reciprocal_rank_fusion"

    fusion_version: Literal[
        "competition_bm25_query_rrf_v1"
    ] = "competition_bm25_query_rrf_v1"

    rank_constant: int = Field(
        default=60,
        ge=1,
    )

    question_only_candidate_count: int = Field(
        default=20,
        ge=1,
    )

    question_with_options_candidate_count: int = Field(
        default=20,
        ge=1,
    )


class CompetitionBM25QueryFusionHit(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1

    rank: int = Field(
        ge=1,
    )

    score: float = Field(
        gt=0,
    )

    retriever_type: Literal[
        "bm25_query_rrf"
    ] = "bm25_query_rrf"

    score_type: Literal[
        "rrf"
    ] = "rrf"

    rrf_score: float = Field(
        gt=0,
    )

    question_only_rank: int | None = Field(
        default=None,
        ge=1,
    )

    question_with_options_rank: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )

    source_queries: tuple[
        CompetitionBM25QuerySource,
        ...,
    ]

    chunk: CompetitionTextChunk

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_id(self) -> str:
        return self.chunk.source_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def chunk_type(
        self,
    ) -> CompetitionChunkType:
        return self.chunk.chunk_type

    @model_validator(
        mode="after"
    )
    def validate_fusion_fields(
        self,
    ) -> Self:
        expected_sources: list[
            CompetitionBM25QuerySource
        ] = []

        if self.question_only_rank is not None:
            expected_sources.append(
                "question_only"
            )

        if (
            self.question_with_options_rank
            is not None
        ):
            expected_sources.append(
                "question_with_options"
            )

        if not expected_sources:
            raise ValueError(
                "融合 Hit 至少需要一个"
                "来源查询排名"
            )

        if (
            self.source_queries
            != tuple(expected_sources)
        ):
            raise ValueError(
                "source_queries 与"
                "来源查询排名不一致"
            )

        if (
            abs(
                self.score
                - self.rrf_score
            )
            > 1e-12
        ):
            raise ValueError(
                "融合 score 必须等于 rrf_score"
            )

        return self

    @classmethod
    def from_source_hit(
        cls,
        *,
        rank: int,
        rrf_score: float,
        chunk: CompetitionTextChunk,
        question_only_rank: int | None,
        question_with_options_rank: int | None,
    ) -> CompetitionBM25QueryFusionHit:
        source_queries: list[
            CompetitionBM25QuerySource
        ] = []

        if question_only_rank is not None:
            source_queries.append(
                "question_only"
            )

        if question_with_options_rank is not None:
            source_queries.append(
                "question_with_options"
            )

        return cls(
            rank=rank,
            score=rrf_score,
            rrf_score=rrf_score,
            question_only_rank=(
                question_only_rank
            ),
            question_with_options_rank=(
                question_with_options_rank
            ),
            source_queries=tuple(
                source_queries
            ),
            chunk=chunk,
        )