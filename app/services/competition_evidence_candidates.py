from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.schemas.competition_chunk import (
    CompetitionTextChunk,
)
from app.services.competition_retrieval_eval import (
    normalize_competition_evidence_text,
)


@dataclass(frozen=True, slots=True)
class CompetitionEvidenceCandidate:
    rank: int
    chunk: CompetitionTextChunk
    normalized_fact_char_count: int
    normalized_chunk_char_count: int
    unigram_overlap_count: int
    unigram_recall: float
    bigram_overlap_count: int
    bigram_precision: float
    bigram_recall: float
    bigram_f1: float
    sequence_ratio: float


def _character_ngrams(
    value: str,
    *,
    size: int,
) -> Counter[str]:
    if size < 1:
        raise ValueError(
            "N-gram size 必须大于等于 1"
        )

    if not value:
        return Counter()

    if len(value) < size:
        return Counter(
            {
                value: 1,
            }
        )

    return Counter(
        value[
            index: index + size
        ]
        for index in range(
            len(value) - size + 1
        )
    )


def _overlap_count(
    left: Counter[str],
    right: Counter[str],
) -> int:
    return sum(
        (
            left
            & right
        ).values()
    )


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _f1(
    *,
    precision: float,
    recall: float,
) -> float:
    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / denominator
    )


def _score_chunk(
    *,
    normalized_fact: str,
    chunk: CompetitionTextChunk,
) -> dict[str, object]:
    normalized_chunk = (
        normalize_competition_evidence_text(
            chunk.text
        )
    )

    fact_unigrams = _character_ngrams(
        normalized_fact,
        size=1,
    )

    chunk_unigrams = _character_ngrams(
        normalized_chunk,
        size=1,
    )

    unigram_overlap = _overlap_count(
        fact_unigrams,
        chunk_unigrams,
    )

    fact_bigrams = _character_ngrams(
        normalized_fact,
        size=2,
    )

    chunk_bigrams = _character_ngrams(
        normalized_chunk,
        size=2,
    )

    bigram_overlap = _overlap_count(
        fact_bigrams,
        chunk_bigrams,
    )

    bigram_precision = _safe_ratio(
        bigram_overlap,
        sum(chunk_bigrams.values()),
    )

    bigram_recall = _safe_ratio(
        bigram_overlap,
        sum(fact_bigrams.values()),
    )

    return {
        "chunk": chunk,
        "normalized_fact_char_count": (
            len(normalized_fact)
        ),
        "normalized_chunk_char_count": (
            len(normalized_chunk)
        ),
        "unigram_overlap_count": (
            unigram_overlap
        ),
        "unigram_recall": _safe_ratio(
            unigram_overlap,
            sum(fact_unigrams.values()),
        ),
        "bigram_overlap_count": (
            bigram_overlap
        ),
        "bigram_precision": (
            bigram_precision
        ),
        "bigram_recall": bigram_recall,
        "bigram_f1": _f1(
            precision=bigram_precision,
            recall=bigram_recall,
        ),
        "sequence_ratio": (
            SequenceMatcher(
                None,
                normalized_fact,
                normalized_chunk,
                autojunk=False,
            ).ratio()
        ),
    }


def rank_competition_evidence_candidates(
    *,
    fact: str,
    expected_source_id: str,
    chunks: tuple[
        CompetitionTextChunk,
        ...,
    ],
    top_k: int = 5,
) -> tuple[
    CompetitionEvidenceCandidate,
    ...,
]:
    if top_k < 1:
        raise ValueError(
            "top_k 必须大于等于 1"
        )

    normalized_fact = (
        normalize_competition_evidence_text(
            fact
        )
    )

    if not normalized_fact:
        raise ValueError(
            "Evidence Fact 归一化后为空"
        )

    source_chunks = tuple(
        chunk
        for chunk in chunks
        if chunk.source_id == expected_source_id
    )

    if not source_chunks:
        raise ValueError(
            "Corpus 中没有正确来源的 Chunk: "
            f"{expected_source_id}"
        )

    scored = [
        _score_chunk(
            normalized_fact=normalized_fact,
            chunk=chunk,
        )
        for chunk in source_chunks
    ]

    scored.sort(
        key=lambda item: (
            -float(
                item["bigram_recall"]
            ),
            -float(
                item["bigram_f1"]
            ),
            -float(
                item["unigram_recall"]
            ),
            -float(
                item["sequence_ratio"]
            ),
            item["chunk"].chunk_index,
            item["chunk"].chunk_id,
        )
    )

    return tuple(
        CompetitionEvidenceCandidate(
            rank=rank,
            **payload,
        )
        for rank, payload in enumerate(
            scored[:top_k],
            start=1,
        )
    )