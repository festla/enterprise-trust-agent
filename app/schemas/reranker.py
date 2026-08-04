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

from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
    SourceRetriever,
)

from app.schemas.retrieval import (
    RetrievalHit,
)


class RerankerSpec(BaseModel):
    """Cross-Encoder Reranker 的确定性模型配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = Field(
        default=1,
        ge=1,
        le=1,
    )

    provider: Literal[
        "sentence_transformers_cross_encoder"
    ] = "sentence_transformers_cross_encoder"
    
    architecture: Literal[
        "cross_encoder"
    ] = "cross_encoder"

    model_name: str = Field(
        default="BAAI/bge-reranker-base",
        min_length=1,
    )

    # 真实实验时应填写固定 revision，
    # 不把可漂移的远程 main 当作稳定版本。
    model_revision: str = Field(
        min_length=1,
    )

    max_length: int = Field(
        default=512,
        ge=8,
    )

    score_type: Literal[
        "cross_encoder_logit"
    ] = "cross_encoder_logit"

    input_template_version: Literal[
        "query_chunk_v1"
    ] = "query_chunk_v1"



class RerankerRuntimeConfig(BaseModel):
    """不改变模型身份的 Reranker 运行配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    batch_size: int = Field(
        default=8,
        ge=1,
    )

    device: str = Field(
        default="cpu",
        min_length=1,
    )

    local_files_only: bool = False

    rerank_candidate_count: int = Field(
        default=50,
        ge=1,
    )

    return_count: int = Field(
        default=5,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_candidate_counts(
        self,
    ) -> Self:
        """最终返回数不能超过送入重排的候选数。"""

        if (
            self.return_count
            > self.rerank_candidate_count
        ):
            raise ValueError(
                "return_count 不能大于 "
                "rerank_candidate_count"
            )

        return self


class RerankedRetrievalHit(RetrievalHit):
    """经过 Cross-Encoder 重排后的可审计结果。"""

    retriever_type: Literal[
        "hybrid_reranker"
    ] = "hybrid_reranker"

    score_type: Literal[
        "cross_encoder_logit"
    ] = "cross_encoder_logit"

    dense_rank: int | None = Field(
        default=None,
        ge=1,
    )

    bm25_rank: int | None = Field(
        default=None,
        ge=1,
    )

    rrf_rank: int = Field(
        ge=1,
    )

    rrf_score: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    reranker_score: float = Field(
        allow_inf_nan=False,
    )

    source_retrievers: tuple[
        SourceRetriever,
        ...,
    ]

    @classmethod
    def from_hybrid_hit(
        cls,
        *,
        rank: int,
        reranker_score: float,
        source_hit: HybridRetrievalHit,
    ) -> RerankedRetrievalHit:
        """由原始 Hybrid Hit 构造最终重排结果。"""

        source_values = source_hit.model_dump(
            exclude={
                "rank",
                "retriever_type",
                "score_type",
                "score",
                "dense_rank",
                "bm25_rank",
                "rrf_score",
                "source_retrievers",
            }
        )

        return cls(
            rank=rank,
            retriever_type=(
                "hybrid_reranker"
            ),
            score_type=(
                "cross_encoder_logit"
            ),
            score=reranker_score,
            reranker_score=reranker_score,
            dense_rank=(
                source_hit.dense_rank
            ),
            bm25_rank=(
                source_hit.bm25_rank
            ),
            rrf_rank=source_hit.rank,
            rrf_score=(
                source_hit.rrf_score
            ),
            source_retrievers=(
                source_hit
                .source_retrievers
            ),
            **source_values,
        )

    @model_validator(mode="after")
    def validate_reranker_fields(
        self,
    ) -> Self:
        """校验重排分数与两路来源字段。"""

        if (
            self.dense_rank is None
            and self.bm25_rank is None
        ):
            raise ValueError(
                "Reranked Hit 至少需要一个"
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
                "source_retrievers 与"
                "来源排名不一致"
            )

        if (
            abs(
                self.score
                - self.reranker_score
            )
            > 1e-12
        ):
            raise ValueError(
                "Reranked Hit 的 score "
                "必须等于 reranker_score"
            )

        return self