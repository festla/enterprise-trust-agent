from __future__ import annotations

from collections.abc import (
    Sequence,
)
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
    RerankerRuntimeConfig,
    RerankerSpec,
)


class RerankingError(ValueError):
    """Reranker 基础异常。"""


class InvalidRerankerQueryError(
    RerankingError
):
    """Reranker 查询为空。"""


class InvalidHybridHitsError(
    RerankingError
):
    """Hybrid 输入结果类型或排名非法。"""


class DuplicateRerankerChunkError(
    RerankingError
):
    """待重排结果包含重复 Chunk。"""


class RerankerOutputCountMismatchError(
    RerankingError
):
    """Provider 返回的分数数量不正确。"""


class InvalidRerankerScoreError(
    RerankingError
):
    """Provider 返回了非法分数。"""


class RerankerProvider(Protocol):
    """真实 Cross-Encoder 与 Fake Provider 的统一接口。"""

    @property
    def spec(self) -> RerankerSpec:
        """返回可审计的模型配置。"""

    def score_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> Sequence[float]:
        """为 query-passage 对返回相关性分数。"""


@dataclass(frozen=True, slots=True)
class _RerankCandidate:
    source_hit: HybridRetrievalHit
    reranker_score: float


def _validate_hybrid_hits(
    hits: Sequence[
        HybridRetrievalHit
    ],
) -> None:
    """验证 RRF 候选排名、类型和唯一性。"""

    actual_ranks = tuple(
        hit.rank
        for hit in hits
    )

    expected_ranks = tuple(
        range(
            1,
            len(hits) + 1,
        )
    )

    if actual_ranks != expected_ranks:
        raise InvalidHybridHitsError(
            "Hybrid Hit 的 rank 必须从 1 "
            "开始连续递增"
        )

    if any(
        hit.retriever_type
        != "hybrid_rrf"
        for hit in hits
    ):
        raise InvalidHybridHitsError(
            "Reranker 只能处理 "
            "hybrid_rrf 检索结果"
        )

    chunk_ids = tuple(
        hit.chunk_id
        for hit in hits
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise DuplicateRerankerChunkError(
            "待重排结果包含重复 chunk_id"
        )


def _normalize_score(
    value: object,
) -> float:
    """将模型输出转换为有限浮点数。"""

    if isinstance(
        value,
        (str, bytes, bool),
    ):
        raise InvalidRerankerScoreError(
            "Reranker Score 必须是数值"
        )

    try:
        score = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidRerankerScoreError(
            "Reranker Score 无法转换为浮点数"
        ) from exc

    if not isfinite(score):
        raise InvalidRerankerScoreError(
            "Reranker Score 必须是有限数值"
        )

    return score


def rerank_hybrid_hits(
    *,
    query: str,
    hits: Sequence[
        HybridRetrievalHit
    ],
    provider: RerankerProvider,
    config: (
        RerankerRuntimeConfig
        | None
    ) = None,
) -> tuple[
    RerankedRetrievalHit,
    ...,
]:
    """使用 Provider 对 Hybrid 候选重新排序。"""

    if not query.strip():
        raise InvalidRerankerQueryError(
            "Reranker 查询不能为空"
        )

    _validate_hybrid_hits(
        hits
    )

    # 元数据过滤后没有候选属于正常情况。
    if not hits:
        return ()

    active_config = (
        config
        if config is not None
        else RerankerRuntimeConfig()
    )

    candidates = tuple(
        hits[
            :active_config
            .rerank_candidate_count
        ]
    )

    pairs = tuple(
        (
            query,
            hit.text,
        )
        for hit in candidates
    )

    raw_scores = tuple(
        provider.score_pairs(
            pairs
        )
    )

    if (
        len(raw_scores)
        != len(candidates)
    ):
        raise (
            RerankerOutputCountMismatchError(
                "Provider 返回的分数数量"
                "与候选数量不一致："
                f"expected={len(candidates)}, "
                f"actual={len(raw_scores)}"
            )
        )

    rerank_candidates: list[
        _RerankCandidate
    ] = []

    for source_hit, raw_score in zip(
        candidates,
        raw_scores,
        strict=True,
    ):
        rerank_candidates.append(
            _RerankCandidate(
                source_hit=source_hit,
                reranker_score=(
                    _normalize_score(
                        raw_score
                    )
                ),
            )
        )

    ranked_candidates = sorted(
        rerank_candidates,
        key=lambda candidate: (
            -candidate.reranker_score,
            candidate.source_hit.rank,
            candidate.source_hit.chunk_id,
        ),
    )

    selected = ranked_candidates[
        :active_config.return_count
    ]

    return tuple(
        RerankedRetrievalHit.from_hybrid_hit(
            rank=rank,
            reranker_score=(
                candidate.reranker_score
            ),
            source_hit=(
                candidate.source_hit
            ),
        )
        for rank, candidate
        in enumerate(
            selected,
            start=1,
        )
    )