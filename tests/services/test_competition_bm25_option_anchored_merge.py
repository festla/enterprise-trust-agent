from __future__ import annotations

import pytest

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_option_anchored_merge import (
    CompetitionBM25OptionAnchoredMergeConfig,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_bm25_option_anchored_merge import (
    CompetitionBM25OptionAnchoredChunkMismatchError,
    DuplicateCompetitionBM25OptionAnchoredChunkError,
    EmptyCompetitionBM25OptionAnchoredMergeError,
    InvalidCompetitionBM25OptionAnchoredHitsError,
    InvalidCompetitionBM25OptionAnchoredTopKError,
    merge_competition_bm25_option_anchored,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _hit(
    *,
    source_number: int,
    text: str,
    rank: int,
) -> CompetitionBM25Hit:
    source_id = (
        f"src_{source_number:016x}"
    )

    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title="Option Anchored Merge 测试来源",
        source_type="word",
        relative_path=(
            f"word/{source_id}.docx"
        ),
        sha256=(
            f"{source_number:x}"[-1]
            * 64
        ),
    )

    document = CompetitionTextDocument(
        source=source,
        blocks=(
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:"
                    "00000"
                ),
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type="word",
                block_index=0,
                block_type="paragraph",
                paragraph_index=0,
                text=text,
            ),
        ),
    )

    chunk = build_competition_text_chunks(
        document
    )[0]

    return CompetitionBM25Hit(
        rank=rank,
        score=10.0 / rank,
        chunk=chunk,
    )


def test_preserves_option_top_k_chunk_set(
) -> None:
    option_hits = tuple(
        _hit(
            source_number=index,
            text=f"option-{index}",
            rank=index,
        )
        for index in range(1, 11)
    )

    #
    # question-only 中命中 option 第 8 名，
    # 使它被提升到头部。
    #
    shared_question_hit = (
        option_hits[7].model_copy(
            update={
                "rank": 1,
                "score": 20.0,
            }
        )
    )

    question_only_extra = _hit(
        source_number=100,
        text="question-only extra",
        rank=2,
    )

    result = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                shared_question_hit,
                question_only_extra,
            ),
            question_with_options_hits=option_hits,
            top_k=10,
        )
    )

    assert len(result) == 10

    assert {
        hit.chunk_id
        for hit in result
    } == {
        hit.chunk_id
        for hit in option_hits
    }

    #
    # question-only extra 不允许挤掉 option anchor。
    #
    assert (
        question_only_extra.chunk_id
        not in {
            hit.chunk_id
            for hit in result
        }
    )

    #
    # 共识 Chunk 被提升。
    #
    assert (
        result[0].chunk_id
        == option_hits[7].chunk_id
    )

    assert result[0].layer == "consensus"


def test_remaining_option_hits_keep_original_order(
) -> None:
    option_first = _hit(
        source_number=201,
        text="option first",
        rank=1,
    )
    option_second = _hit(
        source_number=202,
        text="option second",
        rank=2,
    )
    option_third = _hit(
        source_number=203,
        text="option third",
        rank=3,
    )

    question_shared = (
        option_third.model_copy(
            update={
                "rank": 1,
                "score": 10.0,
            }
        )
    )

    result = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                question_shared,
            ),
            question_with_options_hits=(
                option_first,
                option_second,
                option_third,
            ),
            top_k=3,
        )
    )

    assert tuple(
        hit.chunk_id
        for hit in result
    ) == (
        option_third.chunk_id,
        option_first.chunk_id,
        option_second.chunk_id,
    )

    assert tuple(
        hit.layer
        for hit in result
    ) == (
        "consensus",
        "option_anchor",
        "option_anchor",
    )


def test_question_rank_limit_controls_promotion(
) -> None:
    option_first = _hit(
        source_number=301,
        text="option first",
        rank=1,
    )
    option_second = _hit(
        source_number=302,
        text="option second",
        rank=2,
    )

    shared_question = (
        option_second.model_copy(
            update={
                "rank": 5,
                "score": 2.0,
            }
        )
    )

    result = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                _hit(
                    source_number=303,
                    text="q1",
                    rank=1,
                ),
                _hit(
                    source_number=304,
                    text="q2",
                    rank=2,
                ),
                _hit(
                    source_number=305,
                    text="q3",
                    rank=3,
                ),
                _hit(
                    source_number=306,
                    text="q4",
                    rank=4,
                ),
                shared_question,
            ),
            question_with_options_hits=(
                option_first,
                option_second,
            ),
            config=(
                CompetitionBM25OptionAnchoredMergeConfig(
                    promotion_question_rank_limit=3
                )
            ),
            top_k=2,
        )
    )

    assert tuple(
        hit.chunk_id
        for hit in result
    ) == (
        option_first.chunk_id,
        option_second.chunk_id,
    )

    assert tuple(
        hit.layer
        for hit in result
    ) == (
        "option_anchor",
        "option_anchor",
    )


def test_backfills_question_hits_only_when_option_short(
) -> None:
    option_hit = _hit(
        source_number=401,
        text="only option",
        rank=1,
    )

    question_extra = _hit(
        source_number=402,
        text="question extra",
        rank=1,
    )

    result = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                question_extra,
            ),
            question_with_options_hits=(
                option_hit,
            ),
            top_k=2,
        )
    )

    assert len(result) == 2

    assert result[0].chunk_id == (
        option_hit.chunk_id
    )

    assert result[1].chunk_id == (
        question_extra.chunk_id
    )

    assert result[1].layer == (
        "question_backfill"
    )


def test_allows_question_only_when_option_empty(
) -> None:
    question_hit = _hit(
        source_number=501,
        text="question only",
        rank=1,
    )

    result = (
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                question_hit,
            ),
            question_with_options_hits=(),
            top_k=1,
        )
    )

    assert len(result) == 1
    assert (
        result[0].chunk_id
        == question_hit.chunk_id
    )
    assert result[0].layer == (
        "question_backfill"
    )


def test_rejects_empty_and_invalid_top_k(
) -> None:
    with pytest.raises(
        EmptyCompetitionBM25OptionAnchoredMergeError,
        match="都没有候选",
    ):
        merge_competition_bm25_option_anchored(
            question_only_hits=(),
            question_with_options_hits=(),
        )

    hit = _hit(
        source_number=601,
        text="invalid top k",
        rank=1,
    )

    with pytest.raises(
        InvalidCompetitionBM25OptionAnchoredTopKError,
        match="大于等于 1",
    ):
        merge_competition_bm25_option_anchored(
            question_only_hits=(hit,),
            question_with_options_hits=(),
            top_k=0,
        )


def test_rejects_invalid_rank_and_duplicate(
) -> None:
    first = _hit(
        source_number=701,
        text="first",
        rank=1,
    )

    bad_rank = _hit(
        source_number=702,
        text="bad rank",
        rank=3,
    )

    with pytest.raises(
        InvalidCompetitionBM25OptionAnchoredHitsError,
        match="连续递增",
    ):
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                first,
                bad_rank,
            ),
            question_with_options_hits=(),
        )

    repeated = first.model_copy(
        update={
            "rank": 2,
            "score": 5.0,
        }
    )

    with pytest.raises(
        DuplicateCompetitionBM25OptionAnchoredChunkError,
        match="重复 chunk_id",
    ):
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                first,
                repeated,
            ),
            question_with_options_hits=(),
        )


def test_rejects_chunk_identity_mismatch(
) -> None:
    original = _hit(
        source_number=801,
        text="original",
        rank=1,
    )

    changed_chunk = (
        original.chunk.model_copy(
            update={
                "text": "changed",
            }
        )
    )

    changed_question = (
        original.model_copy(
            update={
                "chunk": changed_chunk,
            }
        )
    )

    with pytest.raises(
        CompetitionBM25OptionAnchoredChunkMismatchError,
        match="内容不一致",
    ):
        merge_competition_bm25_option_anchored(
            question_only_hits=(
                changed_question,
            ),
            question_with_options_hits=(
                original,
            ),
            top_k=1,
        )