from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from app.schemas.competition import (
    CompetitionQaCase,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)


_EVIDENCE_SEPARATOR_PATTERN = re.compile(
    r"[；;\r\n]+"
)


def normalize_competition_evidence_text(
    value: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    return "".join(
        character
        for character in normalized
        if character.isalnum()
    )


def split_competition_evidence_facts(
    evidence: str,
) -> tuple[str, ...]:
    facts = tuple(
        part.strip()
        for part in (
            _EVIDENCE_SEPARATOR_PATTERN
            .split(evidence)
        )
        if part.strip()
    )

    if not facts:
        raise ValueError(
            "Competition Evidence "
            "没有可评测事实"
        )

    return facts


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceFactMatch:
    fact: str
    normalized_fact: str
    hit_ranks: tuple[int, ...]
    chunk_ids: tuple[str, ...]

    @property
    def covered(self) -> bool:
        return bool(self.hit_ranks)


@dataclass(frozen=True, slots=True)
class CompetitionRetrievalCaseResult:
    case_id: str
    source_type: str
    qa_type: str
    difficulty: str
    expected_source_id: str
    retrieved_count: int
    source_ranks: tuple[int, ...]
    fact_matches: tuple[
        CompetitionEvidenceFactMatch,
        ...,
    ]

    @property
    def fact_count(self) -> int:
        return len(self.fact_matches)

    @property
    def first_source_rank(
        self,
    ) -> int | None:
        if not self.source_ranks:
            return None

        return min(self.source_ranks)

    @property
    def first_evidence_rank(
        self,
    ) -> int | None:
        ranks = tuple(
            rank
            for match in self.fact_matches
            for rank in match.hit_ranks
        )

        if not ranks:
            return None

        return min(ranks)

    def source_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        _validate_cutoff(cutoff)

        return any(
            rank <= cutoff
            for rank in self.source_ranks
        )

    def covered_fact_count_at(
        self,
        cutoff: int,
    ) -> int:
        _validate_cutoff(cutoff)

        return sum(
            any(
                rank <= cutoff
                for rank in match.hit_ranks
            )
            for match in self.fact_matches
        )

    def fact_recall_at(
        self,
        cutoff: int,
    ) -> float:
        return (
            self.covered_fact_count_at(
                cutoff
            )
            / self.fact_count
        )

    def any_fact_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return (
            self.covered_fact_count_at(
                cutoff
            )
            > 0
        )

    def all_facts_hit_at(
        self,
        cutoff: int,
    ) -> bool:
        return (
            self.covered_fact_count_at(
                cutoff
            )
            == self.fact_count
        )

    def reciprocal_rank_at(
        self,
        cutoff: int,
    ) -> float:
        _validate_cutoff(cutoff)

        rank = self.first_evidence_rank

        if (
            rank is None
            or rank > cutoff
        ):
            return 0.0

        return 1.0 / rank


@dataclass(frozen=True, slots=True)
class CompetitionRetrievalMetricsAtK:
    cutoff: int
    source_hit_rate: float
    any_fact_hit_rate: float
    all_facts_hit_rate: float
    mean_fact_recall: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class CompetitionRetrievalEvalSummary:
    case_count: int
    metrics: tuple[
        CompetitionRetrievalMetricsAtK,
        ...,
    ]

    def at(
        self,
        cutoff: int,
    ) -> CompetitionRetrievalMetricsAtK:
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


def evaluate_competition_retrieval_case(
    *,
    case: CompetitionQaCase,
    expected_source_id: str,
    hits: tuple[
        CompetitionBM25Hit,
        ...,
    ],
) -> CompetitionRetrievalCaseResult:
    if case.source_type not in {
        "word",
        "pdf",
    }:
        raise ValueError(
            "检索评测只支持 Word/PDF QA"
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

    if (
        len(ranks)
        != len(set(ranks))
    ):
        raise ValueError(
            "检索结果包含重复 rank"
        )

    source_hits = tuple(
        hit
        for hit in ordered_hits
        if (
            hit.source_id
            == expected_source_id
        )
    )

    source_ranks = tuple(
        hit.rank
        for hit in source_hits
    )

    normalized_chunks = tuple(
        (
            hit,
            normalize_competition_evidence_text(
                hit.chunk.text
            ),
        )
        for hit in source_hits
    )

    fact_matches = []

    for fact in (
        split_competition_evidence_facts(
            case.evidence
        )
    ):
        normalized_fact = (
            normalize_competition_evidence_text(
                fact
            )
        )

        if not normalized_fact:
            raise ValueError(
                "Evidence Fact 归一化后为空"
            )

        matched_hits = tuple(
            hit
            for hit, normalized_chunk
            in normalized_chunks
            if (
                normalized_fact
                in normalized_chunk
            )
        )

        fact_matches.append(
            CompetitionEvidenceFactMatch(
                fact=fact,
                normalized_fact=(
                    normalized_fact
                ),
                hit_ranks=tuple(
                    hit.rank
                    for hit in matched_hits
                ),
                chunk_ids=tuple(
                    hit.chunk_id
                    for hit in matched_hits
                ),
            )
        )

    return CompetitionRetrievalCaseResult(
        case_id=case.case_id,
        source_type=case.source_type,
        qa_type=case.qa_type,
        difficulty=case.difficulty,
        expected_source_id=(
            expected_source_id
        ),
        retrieved_count=len(
            ordered_hits
        ),
        source_ranks=source_ranks,
        fact_matches=tuple(
            fact_matches
        ),
    )


def summarize_competition_retrieval(
    results: tuple[
        CompetitionRetrievalCaseResult,
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
) -> CompetitionRetrievalEvalSummary:
    if not results:
        raise ValueError(
            "检索评测结果不能为空"
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

    case_count = len(results)
    metrics = []

    for cutoff in normalized_cutoffs:
        metrics.append(
            CompetitionRetrievalMetricsAtK(
                cutoff=cutoff,
                source_hit_rate=(
                    sum(
                        result.source_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / case_count
                ),
                any_fact_hit_rate=(
                    sum(
                        result.any_fact_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / case_count
                ),
                all_facts_hit_rate=(
                    sum(
                        result.all_facts_hit_at(
                            cutoff
                        )
                        for result in results
                    )
                    / case_count
                ),
                mean_fact_recall=(
                    sum(
                        result.fact_recall_at(
                            cutoff
                        )
                        for result in results
                    )
                    / case_count
                ),
                mean_reciprocal_rank=(
                    sum(
                        result.reciprocal_rank_at(
                            cutoff
                        )
                        for result in results
                    )
                    / case_count
                ),
            )
        )

    return CompetitionRetrievalEvalSummary(
        case_count=case_count,
        metrics=tuple(metrics),
    )