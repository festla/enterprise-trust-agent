from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeConfig,
    CompetitionBM25OptionAnchoredMergeHit,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)


class CompetitionBM25OptionAnchoredMergeError(
    ValueError
):
    """Option-Anchored Merge 基础异常。"""


class EmptyCompetitionBM25OptionAnchoredMergeError(
    CompetitionBM25OptionAnchoredMergeError
):
    """两路查询都没有候选。"""


class InvalidCompetitionBM25OptionAnchoredTopKError(
    CompetitionBM25OptionAnchoredMergeError
):
    """top_k 无效。"""


class InvalidCompetitionBM25OptionAnchoredHitsError(
    CompetitionBM25OptionAnchoredMergeError
):
    """输入 BM25 Hits 无效。"""


class DuplicateCompetitionBM25OptionAnchoredChunkError(
    CompetitionBM25OptionAnchoredMergeError
):
    """同一路查询包含重复 Chunk。"""


class CompetitionBM25OptionAnchoredChunkMismatchError(
    CompetitionBM25OptionAnchoredMergeError
):
    """相同 Chunk ID 的 Chunk 内容不一致。"""


@dataclass(frozen=True, slots=True)
class _AnchorCandidate:
    hit: CompetitionBM25Hit
    question_only_rank: int | None
    question_with_options_rank: int
    is_promoted_consensus: bool


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
        raise InvalidCompetitionBM25OptionAnchoredHitsError(
            f"{query_name} 的 rank 必须"
            "从 1 开始连续递增"
        )

    if any(
        hit.retriever_type != "bm25"
        for hit in hits
    ):
        raise InvalidCompetitionBM25OptionAnchoredHitsError(
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
        raise DuplicateCompetitionBM25OptionAnchoredChunkError(
            f"{query_name} 包含重复 chunk_id"
        )


def _chunk_identity(
    hit: CompetitionBM25Hit,
) -> dict[str, object]:
    return hit.chunk.model_dump(
        mode="json"
    )


def _synthetic_score(
    *,
    layer: str,
    question_rank: int | None,
    option_rank: int | None,
) -> float:
    """
    score 只用于结果记录。

    真正排序由显式 layer/rank 规则决定，
    不依赖这个浮点值。
    """

    if layer == "consensus":
        assert question_rank is not None
        assert option_rank is not None

        return (
            3.0
            + 1.0 / question_rank
            + 1.0 / option_rank
        )

    if layer == "option_anchor":
        assert option_rank is not None

        return (
            2.0
            + 1.0 / option_rank
        )

    assert layer == "question_backfill"
    assert question_rank is not None

    return (
        1.0
        + 1.0 / question_rank
    )


def merge_competition_bm25_option_anchored(
    *,
    question_only_hits: Sequence[
        CompetitionBM25Hit
    ],
    question_with_options_hits: Sequence[
        CompetitionBM25Hit
    ],
    config: (
        CompetitionBM25OptionAnchoredMergeConfig
        | None
    ) = None,
    top_k: int = 10,
) -> tuple[
    CompetitionBM25OptionAnchoredMergeHit,
    ...,
]:
    """
    带选项结果保底的分层合并。

    排序：

    Layer 1:
        question_with_options Top-K 中，
        同时被 question_only 命中的候选。

    Layer 2:
        question_with_options Top-K
        中剩余候选。

    Layer 3:
        当 option 候选不足 top_k 时，
        使用 question_only 候选回填。

    关键不变量：

    若 len(question_with_options_hits) >= top_k，
    则最终 Top-K 的 chunk_id 集合严格等于：

        question_with_options_hits[:top_k]

    只允许改变其内部顺序。
    """

    if top_k < 1:
        raise InvalidCompetitionBM25OptionAnchoredTopKError(
            "top_k 必须大于等于 1"
        )

    if (
        not question_only_hits
        and not question_with_options_hits
    ):
        raise EmptyCompetitionBM25OptionAnchoredMergeError(
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
        else CompetitionBM25OptionAnchoredMergeConfig()
    )

    question_candidates = tuple(
        question_only_hits[
            :active_config.question_candidate_count
        ]
    )

    question_by_chunk_id = {
        hit.chunk_id: hit
        for hit in question_candidates
    }

    #
    # 注意这里直接固定为 option Top-K。
    #
    # 这是 6B1d 最重要的设计：
    # 不让 question-only 新候选把 option Top-K
    # 中的候选挤出去。
    #
    option_anchor_hits = tuple(
        question_with_options_hits[:top_k]
    )

    anchor_candidates: list[
        _AnchorCandidate
    ] = []

    for option_hit in option_anchor_hits:
        question_hit = question_by_chunk_id.get(
            option_hit.chunk_id
        )

        if question_hit is not None:
            if (
                _chunk_identity(question_hit)
                != _chunk_identity(option_hit)
            ):
                raise (
                    CompetitionBM25OptionAnchoredChunkMismatchError(
                        "相同 chunk_id 在两路查询中的"
                        " Chunk 内容不一致"
                    )
                )

            question_rank: int | None = (
                question_hit.rank
            )
        else:
            question_rank = None

        rank_limit = (
            active_config
            .promotion_question_rank_limit
        )

        promoted = (
            question_rank is not None
            and (
                rank_limit is None
                or question_rank <= rank_limit
            )
        )

        anchor_candidates.append(
            _AnchorCandidate(
                hit=option_hit,
                question_only_rank=question_rank,
                question_with_options_rank=(
                    option_hit.rank
                ),
                is_promoted_consensus=promoted,
            )
        )

    consensus_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in anchor_candidates
                if candidate.is_promoted_consensus
            ),
            key=lambda candidate: (
                min(
                    (
                        candidate.question_only_rank
                        if candidate.question_only_rank
                        is not None
                        else 10**12
                    ),
                    candidate
                    .question_with_options_rank,
                ),
                candidate
                .question_with_options_rank,
                (
                    candidate.question_only_rank
                    if candidate.question_only_rank
                    is not None
                    else 10**12
                ),
                candidate.hit.chunk_id,
            ),
        )
    )

    option_only_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in anchor_candidates
                if not candidate.is_promoted_consensus
            ),
            key=lambda candidate: (
                candidate
                .question_with_options_rank,
                candidate.hit.chunk_id,
            ),
        )
    )

    ordered_hits: list[
        CompetitionBM25OptionAnchoredMergeHit
    ] = []

    for candidate in consensus_candidates:
        ordered_hits.append(
            CompetitionBM25OptionAnchoredMergeHit(
                rank=len(ordered_hits) + 1,
                score=_synthetic_score(
                    layer="consensus",
                    question_rank=(
                        candidate.question_only_rank
                    ),
                    option_rank=(
                        candidate
                        .question_with_options_rank
                    ),
                ),
                layer="consensus",
                question_only_rank=(
                    candidate.question_only_rank
                ),
                question_with_options_rank=(
                    candidate
                    .question_with_options_rank
                ),
                chunk=candidate.hit.chunk,
            )
        )

    for candidate in option_only_candidates:
        layer = "option_anchor"

        ordered_hits.append(
            CompetitionBM25OptionAnchoredMergeHit(
                rank=len(ordered_hits) + 1,
                score=_synthetic_score(
                    layer=layer,
                    question_rank=(
                        candidate.question_only_rank
                    ),
                    option_rank=(
                        candidate
                        .question_with_options_rank
                    ),
                ),
                layer=layer,
                question_only_rank=(
                    candidate.question_only_rank
                ),
                question_with_options_rank=(
                    candidate
                    .question_with_options_rank
                ),
                chunk=candidate.hit.chunk,
            )
        )

    #
    # 如果 option 本身不足 top_k，
    # 才允许 question-only 候选补齐。
    #
    if len(ordered_hits) < top_k:
        existing_chunk_ids = {
            hit.chunk_id
            for hit in ordered_hits
        }

        for question_hit in question_candidates:
            if len(ordered_hits) >= top_k:
                break

            if (
                question_hit.chunk_id
                in existing_chunk_ids
            ):
                continue

            ordered_hits.append(
                CompetitionBM25OptionAnchoredMergeHit(
                    rank=len(ordered_hits) + 1,
                    score=_synthetic_score(
                        layer="question_backfill",
                        question_rank=(
                            question_hit.rank
                        ),
                        option_rank=None,
                    ),
                    layer="question_backfill",
                    question_only_rank=(
                        question_hit.rank
                    ),
                    question_with_options_rank=None,
                    chunk=question_hit.chunk,
                )
            )

            existing_chunk_ids.add(
                question_hit.chunk_id
            )

    return tuple(
        ordered_hits[:top_k]
    )