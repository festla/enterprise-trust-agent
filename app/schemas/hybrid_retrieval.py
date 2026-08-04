from __future__ import annotations

from typing import (
    Literal,
    Self,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.schemas.retrieval import (
    RetrievalHit,
)


SourceRetriever = Literal[
    "dense",
    "bm25",
]

class RRFConfig(BaseModel):
    """Reciprocal Rank Fusion 的确定性配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    fusion_method: Literal[
        "reciprocal_rank_fusion"
    ] = "reciprocal_rank_fusion"


    fusion_version: Literal[
        "rrf_v1"
    ] = "rrf_v1"

    rank_constant: int = Field(
        default=60,
        ge=1,
    )

    dense_candidate_count: int = Field(
        default=50,
        ge=1,
    )

    bm25_candidate_count: int = Field(
        default=50,
        ge=1,
    )


class HybridRetrievalHit(RetrievalHit):
    """经过 RRF 融合后的可审计检索结果。"""

    retriever_type: Literal[
        "hybrid_rrf"
    ] = "hybrid_rrf"

    score_type: Literal[
        "rrf"
    ] = "rrf"

    dense_rank: int | None = Field(
        default=None,
        ge=1,
    )

    bm25_rank: int | None = Field(
        default=None,
        ge=1,
    )

    rrf_score: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    source_retrievers: tuple[
        SourceRetriever,
        ...,
    ]


    @classmethod
    def from_source_hit(
        cls,
        *,
        rank: int,
        rrf_score: float,
        source_hit: RetrievalHit,
        dense_rank: int | None,
        bm25_rank: int | None,
    ) -> HybridRetrievalHit:
        """由任一路原始 Hit 构造融合结果。"""

        source_retrievers: list[
            SourceRetriever
        ] = []

        if dense_rank is not None:
            source_retrievers.append(
                "dense"
            )

        if bm25_rank is not None:
            source_retrievers.append(
                "bm25"
            )

        source_values = source_hit.model_dump(
            exclude={
                "rank",
                "retriever_type",
                "score_type",
                "score",
            }
        )

        return cls(
            rank=rank,
            retriever_type="hybrid_rrf",
            score_type="rrf",
            score=rrf_score,
            rrf_score=rrf_score,
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
            source_retrievers=tuple(
                source_retrievers
            ),
            **source_values,
        )

    @model_validator(mode="after")
    def validate_fusion_fields(
        self,
    ) -> Self:
        """确保来源排名、来源列表和分数一致。"""

        if (
            self.dense_rank is None
            and self.bm25_rank is None
        ):
            raise ValueError(
                "Hybrid Hit 至少需要一个" \
                "来源检索排名"
            )

        expected_sources: list[
            SourceRetriever
        ] = []

        if self.dense_rank is not None:
            expected_sources.append(
                "dense"
            )

        if self.bm25_rank is not None:
            expected_sources.append(
                "bm25"
            )

        if (
            self.source_retrievers
            != tuple(expected_sources)
        ):
            raise ValueError(
                "source_retrievers 与" \
                "来源排名不一致"
            )

        if (
            abs(
                self.score
                - self.rrf_score
            )
            > 1e-12
        ):
            raise ValueError(
                "Hybrid score 必须等于" \
                "rrf_score"
            )

        return self
    