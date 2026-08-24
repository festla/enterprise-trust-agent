from __future__ import annotations

from dataclasses import dataclass

from app.schemas.competition_gold import (
    CompetitionGoldChunkRole,
    CompetitionGoldEvidenceMode,
    CompetitionGoldFactRecord,
)
from typing import Protocol

class CompetitionGoldRetrievalHit(
    Protocol
):
    """
    Gold Retrieval Eval 所需的最小 Hit 接口。

    BM25 / Dense / Hybrid 只要能够提供：
        rank
        chunk_id
    就可以共享同一套 Gold 评测逻辑。
    """

    rank: int

    @property
    def chunk_id(
        self,
    ) -> str:
        ...

@dataclass(frozen=True, slots=True)
class CompetitionGoldChunkMatch:
    chunk_id: str
    role: CompetitionGoldChunkRole
    rank: int | None

    def hit_at(
        self,
        cutoff: int,
    ) -> bool:
        _validate_cutoff(cutoff)

        return (
            self.rank is not None
            and self.rank <= cutoff
        )


@dataclass(frozen=True, slots=True)
class CompetitionGoldFactRetrievalResult:
    case_id: str
    fact_index: int
    fact: str
    expected_source_id: str
    evidence_mode: CompetitionGoldEvidenceMode
    retrieved_count: int
    chunk_matches: tuple[
        CompetitionGoldChunkMatch,
        ...,
    ]

    @property
    def gold_chunk_count(self) -> int:
        return len(self.chunk_matches)

    @property
    def direct_chunk_count(self) -> int:
        return sum(
            match.role == "direct"
            for match in self.chunk_matches
        )

    @property
    def first_direct_rank(
        self,
    ) -> int | None:
        ranks = tuple(
            match.rank
            for match in self.chunk_matches
            if (
                match.role == "direct"
                and match.rank is not None
            )
        )

        if not ranks:
            return None

        return min(ranks)

    def matched_gold_chunk_count_at(
        self,
        cutoff: int,
    ) -> int:
        _validate_cutoff(cutoff)

        return sum(
            match.hit_at(cutoff)
            for match in self.chunk_matches
        )

    def direct_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        _validate_cutoff(cutoff)

        return any(
            match.role == "direct"
            and match.hit_at(cutoff)
            for match in self.chunk_matches
        )

    def any_gold_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return (
            self.matched_gold_chunk_count_at(
                cutoff
            )
            > 0
        )

    def complete_gold_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return (
            self.matched_gold_chunk_count_at(
                cutoff
            )
            == self.gold_chunk_count
        )

    def gold_chunk_recall_at(
        self,
        cutoff: int,
    ) -> float:
        return (
            self.matched_gold_chunk_count_at(
                cutoff
            )
            / self.gold_chunk_count
        )

    def direct_reciprocal_rank_at(
        self,
        cutoff: int,
    ) -> float:
        _validate_cutoff(cutoff)

        rank = self.first_direct_rank

        if (
            rank is None
            or rank > cutoff
        ):
            return 0.0

        return 1.0 / rank


@dataclass(frozen=True, slots=True)
class CompetitionGoldRetrievalMetricsAtK:
    cutoff: int
    fact_direct_hit_rate: float
    fact_any_gold_hit_rate: float
    fact_complete_gold_hit_rate: float
    mean_gold_chunk_recall: float
    mean_direct_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class CompetitionGoldRetrievalEvalSummary:
    fact_count: int
    metrics: tuple[
        CompetitionGoldRetrievalMetricsAtK,
        ...,
    ]

    def at(
        self,
        cutoff: int,
    ) -> CompetitionGoldRetrievalMetricsAtK:
        for metric in self.metrics:
            if metric.cutoff == cutoff:
                return metric

        raise KeyError(
            f"没有 cutoff={cutoff} 的指标"
        )


def _validate_cutoff(
    cutoff: int,
) -> None:
    if cutoff < 1:
        raise ValueError(
            "cutoff 必须大于等于 1"
        )


def evaluate_competition_gold_fact(
    *,
    record: CompetitionGoldFactRecord,
    hits: tuple[
        CompetitionGoldRetrievalHit,
        ...,
    ],
) -> CompetitionGoldFactRetrievalResult:
    if record.review_status != "confirmed":
        raise ValueError(
            "Gold 检索评测只接受 confirmed 记录"
        )

    if record.evidence_mode is None:
        raise ValueError(
            "confirmed Gold 记录缺少 evidence_mode"
        )

    ordered_hits = tuple(
        sorted(
            hits,
            key=lambda hit: (
                hit.rank,
                hit.chunk_id,
            ),
        )
    )

    ranks = tuple(
        hit.rank
        for hit in ordered_hits
    )

    if len(ranks) != len(set(ranks)):
        raise ValueError(
            "检索结果包含重复 rank"
        )

    retrieved_chunk_ids = tuple(
        hit.chunk_id
        for hit in ordered_hits
    )

    if (
        len(retrieved_chunk_ids)
        != len(set(retrieved_chunk_ids))
    ):
        raise ValueError(
            "检索结果包含重复 chunk_id"
        )

    rank_by_chunk_id = {
        hit.chunk_id: hit.rank
        for hit in ordered_hits
    }

    chunk_matches = tuple(
        CompetitionGoldChunkMatch(
            chunk_id=reference.chunk_id,
            role=reference.role,
            rank=rank_by_chunk_id.get(
                reference.chunk_id
            ),
        )
        for reference in record.gold_chunks
    )

    return CompetitionGoldFactRetrievalResult(
        case_id=record.case_id,
        fact_index=record.fact_index,
        fact=record.fact,
        expected_source_id=(
            record.expected_source_id
        ),
        evidence_mode=(
            record.evidence_mode
        ),
        retrieved_count=len(
            ordered_hits
        ),
        chunk_matches=chunk_matches,
    )


def summarize_competition_gold_retrieval(
    results: tuple[
        CompetitionGoldFactRetrievalResult,
        ...,
    ],
    *,
    cutoffs: tuple[
        int,
        ...,
    ] = (
        1,
        3,
        5,
        10,
    ),
) -> CompetitionGoldRetrievalEvalSummary:
    if not results:
        raise ValueError(
            "Gold 检索评测结果不能为空"
        )

    fact_keys = tuple(
        (
            result.case_id,
            result.fact_index,
        )
        for result in results
    )

    if (
        len(fact_keys)
        != len(set(fact_keys))
    ):
        raise ValueError(
            "Gold 检索评测结果包含重复 Fact Key"
        )

    normalized_cutoffs = tuple(
        sorted(set(cutoffs))
    )

    if not normalized_cutoffs:
        raise ValueError(
            "cutoffs 不能为空"
        )

    for cutoff in normalized_cutoffs:
        _validate_cutoff(cutoff)

    fact_count = len(results)
    metrics = []

    for cutoff in normalized_cutoffs:
        metrics.append(
            CompetitionGoldRetrievalMetricsAtK(
                cutoff=cutoff,
                fact_direct_hit_rate=(
                    sum(
                        result.direct_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / fact_count
                ),
                fact_any_gold_hit_rate=(
                    sum(
                        result.any_gold_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / fact_count
                ),
                fact_complete_gold_hit_rate=(
                    sum(
                        result.complete_gold_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / fact_count
                ),
                mean_gold_chunk_recall=(
                    sum(
                        result.gold_chunk_recall_at(
                            cutoff
                        )
                        for result in results
                    )
                    / fact_count
                ),
                mean_direct_reciprocal_rank=(
                    sum(
                        result.direct_reciprocal_rank_at(
                            cutoff
                        )
                        for result in results
                    )
                    / fact_count
                ),
            )
        )

    return CompetitionGoldRetrievalEvalSummary(
        fact_count=fact_count,
        metrics=tuple(metrics),
    )