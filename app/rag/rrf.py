from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
    RRFConfig,
)
from app.schemas.retrieval import (
    RetrievalHit,
)


class RRFError(ValueError):
    """RRF 融合基础异常。"""


class EmptyRRFInputError(
    RRFError
):
    """RRF 缺少一路或多路召回结果。"""


class InvalidRRFTopKError(
    RRFError
):
    """RRF 输出 top_k 参数无效。"""


class InvalidRetrieverHitsError(
    RRFError
):
    """输入 Hit 的类型或排名不合法。"""


class DuplicateRRFChunkError(
    RRFError
):
    """同一路检索结果包含重复 Chunk。"""


class RRFSourceMismatchError(
    RRFError
):
    """Dense 与 BM25 结果来源不一致。"""


@dataclass(frozen=True, slots=True)
class _FusionCandidate:
    source_hit: RetrievalHit
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float


def _validate_hits(
    *,
    hits: Sequence[RetrievalHit],
    expected_retriever: str,
) -> None:
    """检查单路召回的类型、排名和唯一性。"""

    actual_ranks = tuple(
        hit.rank
        for hit in hits
    )

    expected_ranks = tuple(
        range(
            1,
            len(hits) + 1
        )
    )

    if actual_ranks != expected_ranks:
        raise InvalidRetrieverHitsError(
            "输入 Hit 的 rank 必须从 1" \
            "开始连续递增"
        )

    if any(
        hit.retriever_type
        != expected_retriever
        for hit in hits
    ):
        raise InvalidRetrieverHitsError(
            "输入 Hit 的 retriever_type " \
            "与召回来源不一致"
        )

    chunk_ids = tuple(
        hit.chunk_id
        for hit in hits
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise DuplicateRRFChunkError(
            "同一路检索结果包含重复 chunk_id"
        )


def _source_identity(
    hit: RetrievalHit,
) -> tuple[object, ...]:
    """提取融合时必须一致的来源身份。"""

    return (
        hit.chunk_id,
        hit.chunk_dataset_id,
        hit.company_id,
        hit.report_id,
        hit.fiscal_year,
        hit.report_type,
        hit.document_id,
        hit.page_id,
        hit.pdf_page,
        hit.printed_page,
        hit.strategy,
        hit.source_start_char,
        hit.source_end_char,
        hit.text,
    )


def reciprocal_rank_fusion(
    *,
    dense_hits: Sequence[RetrievalHit],
    bm25_hits: Sequence[RetrievalHit],
    config: RRFConfig | None = None,
    top_k: int = 5,
) -> tuple[HybridRetrievalHit, ...]:
    """将 Dense 和 BM25 排名通过 RRF 融合。"""

    if top_k < 1:
        raise InvalidRRFTopKError(
            "top_k 必须大于等于 1"
        )

    if not dense_hits or not bm25_hits:
        raise EmptyRRFInputError(
            "RRF 必须同时接收 Dense "
            "和 BM25 召回结果"
        )

    _validate_hits(
        hits=dense_hits,
        expected_retriever="dense",
    )

    _validate_hits(
        hits=bm25_hits,
        expected_retriever="bm25",
    )

    active_config = (
        config
        if config is not None
        else RRFConfig()
    )

    dense_candidates = tuple(
        dense_hits[
            :active_config
            .dense_candidate_count
        ]
    )

    bm25_candidates = tuple(
        bm25_hits[
            :active_config
            .bm25_candidate_count
        ]
    )

    records: dict[
        str,
        dict[str, object],
    ] = {}

    for hit in dense_candidates:
        records[hit.chunk_id] = {
            "source_hit": hit,
            "dense_rank": hit.rank,
            "bm25_rank": None,
        }

    for hit in bm25_candidates:
        existing = records.get(
            hit.chunk_id
        )

        if existing is None:
            records[hit.chunk_id] = {
                "source_hit": hit,
                "dense_rank": None,
                "bm25_rank": hit.rank,
            }

            continue

        existing_source_hit = (
            existing["source_hit"]
        )

        assert isinstance(
            existing_source_hit,
            RetrievalHit,
        )

        if (
            _source_identity(
                existing_source_hit
            )
            != _source_identity(hit)
        ):
            raise RRFSourceMismatchError(
                "相同 chunk_id 在 Dense 与 "
                "BM25 中的来源元数据不一致"
            )

        existing["bm25_rank"] = hit.rank

    fusion_candidates: list[
        _FusionCandidate
    ] = []

    rank_constant = (
        active_config.rank_constant
    )

    for values in records.values():
        source_hit = values[
            "source_hit"
        ]

        dense_rank = values[
            "dense_rank"
        ]

        bm25_rank = values[
            "bm25_rank"
        ]

        assert isinstance(
            source_hit,
            RetrievalHit,
        )

        assert (
            dense_rank is None
            or isinstance(dense_rank, int)
        )

        assert (
            bm25_rank is None
            or isinstance(bm25_rank, int)
        )

        score = 0.0

        if dense_rank is not None:
            score += 1.0 / (
                rank_constant
                + dense_rank
            )

        if bm25_rank is not None:
            score += 1.0 / (
                rank_constant
                + bm25_rank
            )

        fusion_candidates.append(
            _FusionCandidate(
                source_hit=source_hit,
                dense_rank=dense_rank,
                bm25_rank=bm25_rank,
                rrf_score=score,
            )
        )

    ranked_candidates = sorted(
        fusion_candidates,
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.source_hit.chunk_id,
        ),
    )

    selected = ranked_candidates[
        :top_k
    ]

    return tuple(
        HybridRetrievalHit.from_source_hit(
            rank=rank,
            rrf_score=(
                candidate.rrf_score
            ),
            source_hit=(
                candidate.source_hit
            ),
            dense_rank=(
                candidate.dense_rank
            ),
            bm25_rank=(
                candidate.bm25_rank
            ),
        )
        for rank, candidate
        in enumerate(
            selected,
            start=1,
        )
    )