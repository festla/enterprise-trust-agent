from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.competition_query_fusion import (
    CompetitionBM25QueryFusionConfig,
    CompetitionBM25QueryFusionHit,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)


class CompetitionBM25QueryFusionError(
    ValueError
):
    """双查询 BM25 融合基础异常。"""


class EmptyCompetitionBM25QueryFusionError(
    CompetitionBM25QueryFusionError
):
    """两路查询都没有候选。"""


class InvalidCompetitionBM25FusionTopKError(
    CompetitionBM25QueryFusionError
):
    """融合 top_k 无效。"""


class InvalidCompetitionBM25FusionHitsError(
    CompetitionBM25QueryFusionError
):
    """输入 Hit 的排名或类型无效。"""


class DuplicateCompetitionBM25FusionChunkError(
    CompetitionBM25QueryFusionError
):
    """同一路查询包含重复 Chunk。"""


class CompetitionBM25FusionChunkMismatchError(
    CompetitionBM25QueryFusionError
):
    """相同 Chunk ID 的内容身份不一致。"""


@dataclass(frozen=True, slots=True)
class _FusionCandidate:
    source_hit: CompetitionBM25Hit
    question_only_rank: int | None
    question_with_options_rank: int | None
    rrf_score: float


def _validate_hits(
    hits: Sequence[CompetitionBM25Hit],
    *,
    query_name: str,
) -> None:
    ranks = tuple(
        hit.rank
        for hit in hits
    )

    expected_ranks = tuple(
        range(1, len(hits) + 1)
    )

    if ranks != expected_ranks:
        raise InvalidCompetitionBM25FusionHitsError(
            f"{query_name} 的 rank 必须"
            "从 1 开始连续递增"
        )

    if any(
        hit.retriever_type != "bm25"
        for hit in hits
    ):
        raise InvalidCompetitionBM25FusionHitsError(
            f"{query_name} 包含非 BM25 Hit"
        )

    chunk_ids = tuple(
        hit.chunk_id
        for hit in hits
    )

    if (
        len(chunk_ids)
        != len(set(chunk_ids))
    ):
        raise DuplicateCompetitionBM25FusionChunkError(
            f"{query_name} 包含重复 chunk_id"
        )


def _chunk_identity(
    hit: CompetitionBM25Hit,
) -> dict[str, object]:
    return hit.chunk.model_dump(
        mode="json"
    )


def fuse_competition_bm25_queries(
    *,
    question_only_hits: Sequence[
        CompetitionBM25Hit
    ],
    question_with_options_hits: Sequence[
        CompetitionBM25Hit
    ],
    config: (
        CompetitionBM25QueryFusionConfig
        | None
    ) = None,
    top_k: int = 10,
) -> tuple[
    CompetitionBM25QueryFusionHit,
    ...,
]:
    if top_k < 1:
        raise InvalidCompetitionBM25FusionTopKError(
            "top_k 必须大于等于 1"
        )

    if (
        not question_only_hits
        and not question_with_options_hits
    ):
        raise EmptyCompetitionBM25QueryFusionError(
            "两路 BM25 查询都没有候选"
        )

    _validate_hits(
        question_only_hits,
        query_name="question_only",
    )

    _validate_hits(
        question_with_options_hits,
        query_name="question_with_options",
    )

    active_config = (
        config
        if config is not None
        else CompetitionBM25QueryFusionConfig()
    )

    question_candidates = tuple(
        question_only_hits[
            :active_config
            .question_only_candidate_count
        ]
    )

    option_candidates = tuple(
        question_with_options_hits[
            :active_config
            .question_with_options_candidate_count
        ]
    )

    records: dict[
        str,
        dict[str, object],
    ] = {}

    for hit in question_candidates:
        records[hit.chunk_id] = {
            "source_hit": hit,
            "question_only_rank": hit.rank,
            "question_with_options_rank": None,
        }

    for hit in option_candidates:
        existing = records.get(
            hit.chunk_id
        )

        if existing is None:
            records[hit.chunk_id] = {
                "source_hit": hit,
                "question_only_rank": None,
                "question_with_options_rank": (
                    hit.rank
                ),
            }
            continue

        source_hit = existing[
            "source_hit"
        ]

        assert isinstance(
            source_hit,
            CompetitionBM25Hit,
        )

        if (
            _chunk_identity(source_hit)
            != _chunk_identity(hit)
        ):
            raise CompetitionBM25FusionChunkMismatchError(
                "相同 chunk_id 在两路查询中的"
                " Chunk 内容不一致"
            )

        existing[
            "question_with_options_rank"
        ] = hit.rank

    candidates = []
    rank_constant = active_config.rank_constant

    for raw in records.values():
        source_hit = raw["source_hit"]
        question_rank = raw[
            "question_only_rank"
        ]
        option_rank = raw[
            "question_with_options_rank"
        ]

        assert isinstance(
            source_hit,
            CompetitionBM25Hit,
        )
        assert (
            question_rank is None
            or isinstance(question_rank, int)
        )
        assert (
            option_rank is None
            or isinstance(option_rank, int)
        )

        score = 0.0

        if question_rank is not None:
            score += 1.0 / (
                rank_constant + question_rank
            )

        if option_rank is not None:
            score += 1.0 / (
                rank_constant + option_rank
            )

        candidates.append(
            _FusionCandidate(
                source_hit=source_hit,
                question_only_rank=(
                    question_rank
                ),
                question_with_options_rank=(
                    option_rank
                ),
                rrf_score=score,
            )
        )

    def sort_key(
        candidate: _FusionCandidate,
    ) -> tuple[float, int, int, int, str]:
        missing_rank = 10**12

        question_rank = (
            candidate.question_only_rank
            if candidate.question_only_rank
            is not None
            else missing_rank
        )

        option_rank = (
            candidate
            .question_with_options_rank
            if candidate
            .question_with_options_rank
            is not None
            else missing_rank
        )

        return (
            -candidate.rrf_score,
            min(question_rank, option_rank),
            question_rank,
            option_rank,
            candidate.source_hit.chunk_id,
        )

    ordered = tuple(
        sorted(
            candidates,
            key=sort_key,
        )
    )

    return tuple(
        CompetitionBM25QueryFusionHit
        .from_source_hit(
            rank=rank,
            rrf_score=candidate.rrf_score,
            chunk=candidate.source_hit.chunk,
            question_only_rank=(
                candidate.question_only_rank
            ),
            question_with_options_rank=(
                candidate
                .question_with_options_rank
            ),
        )
        for rank, candidate
        in enumerate(
            ordered[:top_k],
            start=1,
        )
    )