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


CompetitionBM25OptionAnchoredLayer = Literal[
    "consensus",
    "option_anchor",
    "question_backfill",
]


class CompetitionBM25OptionAnchoredMergeConfig(
    BaseModel
):
    """
    6B1d：带选项结果保底的分层合并配置。

    核心原则：

    1. question_with_options 的 Top-K 作为 Anchor。
    2. Anchor 内同时被 question_only 命中的 Chunk
       提升到前面。
    3. Anchor 中其余 Chunk 保持原始 option rank。
    4. 只有 option 结果不足 top_k 时，
       才允许 question_only 候选回填。

    因此当 option hits 数量 >= top_k 时，
    最终 Top-K Chunk 集合严格等于
    question_with_options 的 Top-K Chunk 集合。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    fusion_method: Literal[
        "option_anchored_layered_merge"
    ] = "option_anchored_layered_merge"

    fusion_version: Literal[
        "competition_bm25_option_anchored_merge_v1"
    ] = "competition_bm25_option_anchored_merge_v1"

    question_candidate_count: int = Field(
        default=20,
        ge=1,
    )

    promotion_question_rank_limit: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )


class CompetitionBM25OptionAnchoredMergeHit(
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
        "bm25_option_anchored_merge"
    ] = "bm25_option_anchored_merge"

    score_type: Literal[
        "option_anchored_layer"
    ] = "option_anchored_layer"

    layer: CompetitionBM25OptionAnchoredLayer

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
    def validate_layer_fields(
        self,
    ) -> Self:
        if self.layer == "consensus":
            if (
                self.question_only_rank is None
                or self.question_with_options_rank
                is None
            ):
                raise ValueError(
                    "consensus Hit 必须同时具有"
                    "两路查询排名"
                )

        elif self.layer == "option_anchor":
            if (
                self.question_with_options_rank
                is None
            ):
                raise ValueError(
                    "option_anchor Hit 必须具有"
                    " question_with_options_rank"
                )

        elif self.layer == "question_backfill":
            if self.question_only_rank is None:
                raise ValueError(
                    "question_backfill Hit 必须具有"
                    " question_only_rank"
                )

            if (
                self.question_with_options_rank
                is not None
            ):
                raise ValueError(
                    "question_backfill Hit 不应具有"
                    " question_with_options_rank"
                )

        return self