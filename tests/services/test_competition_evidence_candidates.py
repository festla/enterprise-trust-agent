from __future__ import annotations

import hashlib

import pytest

from app.schemas.competition_chunk import (
    CompetitionChunkSourceSpan,
    CompetitionTextChunk,
)
from app.services.competition_evidence_candidates import (
    rank_competition_evidence_candidates,
)


SOURCE_ID = "src_0123456789abcdef"


def _chunk(
    *,
    text: str,
    chunk_index: int,
    source_id: str = SOURCE_ID,
) -> CompetitionTextChunk:
    doc_id = (
        f"doc_{source_id}_"
        "0123456789abcdef01234567"
    )

    return CompetitionTextChunk(
        chunk_id=(
            f"chunk:{source_id}:"
            f"{chunk_index:05d}"
        ),
        source_id=source_id,
        doc_id=doc_id,
        source_type="pdf",
        chunk_index=chunk_index,
        chunk_type="text",
        source_spans=(
            CompetitionChunkSourceSpan(
                block_id=(
                    f"block:{source_id}:"
                    f"{chunk_index:05d}"
                ),
                block_index=chunk_index,
                start_char=0,
                end_char=len(text),
            ),
        ),
        text=text,
        char_count=len(text),
        text_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
    )


def test_ranks_strongest_lexical_candidate_first(
) -> None:
    fact = (
        "核心一级资本工具"
        "没有到期日"
    )

    candidates = (
        rank_competition_evidence_candidates(
            fact=fact,
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="无关的流动性监管要求。",
                    chunk_index=0,
                ),
                _chunk(
                    text=(
                        "核心一级资本工具"
                        "没有到期日，"
                        "不得设置到期时间。"
                    ),
                    chunk_index=1,
                ),
                _chunk(
                    text=(
                        "核心一级资本工具"
                        "应当直接发行且实缴。"
                    ),
                    chunk_index=2,
                ),
            ),
            top_k=3,
        )
    )

    assert candidates[0].rank == 1
    assert (
        candidates[0].chunk.chunk_index
        == 1
    )
    assert (
        candidates[0].bigram_recall
        == 1.0
    )

    assert (
        candidates[0].bigram_recall
        >= candidates[1].bigram_recall
    )


def test_filters_candidates_by_expected_source(
) -> None:
    wrong_source_id = (
        "src_fedcba9876543210"
    )

    candidates = (
        rank_competition_evidence_candidates(
            fact="资本充足率监管要求",
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="资本监管相关内容",
                    chunk_index=0,
                ),
                _chunk(
                    text="资本充足率监管要求",
                    chunk_index=0,
                    source_id=wrong_source_id,
                ),
            ),
        )
    )

    assert candidates
    assert all(
        candidate.chunk.source_id
        == SOURCE_ID
        for candidate in candidates
    )


def test_uses_chunk_order_for_stable_ties(
) -> None:
    candidates = (
        rank_competition_evidence_candidates(
            fact="相同监管事实",
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="相同监管事实",
                    chunk_index=2,
                ),
                _chunk(
                    text="相同监管事实",
                    chunk_index=0,
                ),
                _chunk(
                    text="相同监管事实",
                    chunk_index=1,
                ),
            ),
            top_k=3,
        )
    )

    assert [
        candidate.chunk.chunk_index
        for candidate in candidates
    ] == [
        0,
        1,
        2,
    ]

    assert [
        candidate.rank
        for candidate in candidates
    ] == [
        1,
        2,
        3,
    ]


def test_supports_single_character_fact(
) -> None:
    candidates = (
        rank_competition_evidence_candidates(
            fact="甲",
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="乙",
                    chunk_index=0,
                ),
                _chunk(
                    text="甲",
                    chunk_index=1,
                ),
            ),
            top_k=2,
        )
    )

    assert (
        candidates[0].chunk.chunk_index
        == 1
    )
    assert candidates[0].bigram_recall == 1.0
    assert candidates[0].unigram_recall == 1.0


def test_rejects_invalid_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        rank_competition_evidence_candidates(
            fact="监管事实",
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="监管事实",
                    chunk_index=0,
                ),
            ),
            top_k=0,
        )


def test_rejects_missing_expected_source(
) -> None:
    with pytest.raises(
        ValueError,
        match="没有正确来源",
    ):
        rank_competition_evidence_candidates(
            fact="监管事实",
            expected_source_id=SOURCE_ID,
            chunks=(
                _chunk(
                    text="监管事实",
                    chunk_index=0,
                    source_id=(
                        "src_fedcba9876543210"
                    ),
                ),
            ),
        )