from __future__ import annotations

import pytest

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_bm25_query_fusion import (
    CompetitionBM25FusionChunkMismatchError,
    DuplicateCompetitionBM25FusionChunkError,
    EmptyCompetitionBM25QueryFusionError,
    InvalidCompetitionBM25FusionHitsError,
    InvalidCompetitionBM25FusionTopKError,
    fuse_competition_bm25_queries,
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
        title="双查询融合测试来源",
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


def test_overlap_is_boosted_above_single_query_hits(
) -> None:
    question_first = _hit(
        source_number=1,
        text="问题查询第一名。",
        rank=1,
    )
    shared_question = _hit(
        source_number=2,
        text="两路共同命中。",
        rank=2,
    )

    option_first = _hit(
        source_number=3,
        text="选项查询第一名。",
        rank=1,
    )
    shared_option = shared_question.model_copy(
        update={
            "rank": 2,
            "score": 5.0,
        }
    )

    hits = fuse_competition_bm25_queries(
        question_only_hits=(
            question_first,
            shared_question,
        ),
        question_with_options_hits=(
            option_first,
            shared_option,
        ),
        top_k=3,
    )

    assert hits[0].chunk_id == (
        shared_question.chunk_id
    )
    assert hits[0].question_only_rank == 2
    assert (
        hits[0].question_with_options_rank
        == 2
    )
    assert hits[0].source_queries == (
        "question_only",
        "question_with_options",
    )


def test_retains_candidates_from_only_one_query(
) -> None:
    question_hit = _hit(
        source_number=4,
        text="仅问题查询命中。",
        rank=1,
    )
    option_hit = _hit(
        source_number=5,
        text="仅选项查询命中。",
        rank=1,
    )

    hits = fuse_competition_bm25_queries(
        question_only_hits=(question_hit,),
        question_with_options_hits=(
            option_hit,
        ),
        top_k=2,
    )

    by_id = {
        hit.chunk_id: hit
        for hit in hits
    }

    assert by_id[
        question_hit.chunk_id
    ].source_queries == (
        "question_only",
    )

    assert by_id[
        option_hit.chunk_id
    ].source_queries == (
        "question_with_options",
    )


def test_tie_order_is_deterministic(
) -> None:
    first = _hit(
        source_number=6,
        text="第一条。",
        rank=1,
    )
    second = _hit(
        source_number=7,
        text="第二条。",
        rank=1,
    )

    hits = fuse_competition_bm25_queries(
        question_only_hits=(first,),
        question_with_options_hits=(second,),
        top_k=2,
    )

    assert tuple(
        hit.chunk_id
        for hit in hits
    ) == tuple(
        sorted(
            (
                first.chunk_id,
                second.chunk_id,
            )
        )
    )

    assert tuple(
        hit.rank
        for hit in hits
    ) == (1, 2)


def test_allows_one_empty_query_result(
) -> None:
    hit = _hit(
        source_number=8,
        text="只有一路有候选。",
        rank=1,
    )

    result = fuse_competition_bm25_queries(
        question_only_hits=(),
        question_with_options_hits=(hit,),
        top_k=10,
    )

    assert len(result) == 1
    assert result[0].chunk_id == hit.chunk_id
    assert result[0].source_queries == (
        "question_with_options",
    )


def test_rejects_invalid_ranks_and_duplicates(
) -> None:
    first = _hit(
        source_number=9,
        text="第一条。",
        rank=1,
    )
    skipped_rank = _hit(
        source_number=10,
        text="跳号。",
        rank=3,
    )

    with pytest.raises(
        InvalidCompetitionBM25FusionHitsError,
        match="连续递增",
    ):
        fuse_competition_bm25_queries(
            question_only_hits=(
                first,
                skipped_rank,
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
        DuplicateCompetitionBM25FusionChunkError,
        match="重复 chunk_id",
    ):
        fuse_competition_bm25_queries(
            question_only_hits=(
                first,
                repeated,
            ),
            question_with_options_hits=(),
        )


def test_rejects_mismatched_chunk_and_invalid_empty_input(
) -> None:
    original = _hit(
        source_number=11,
        text="原始内容。",
        rank=1,
    )

    changed_chunk = original.chunk.model_copy(
        update={
            "text": "被修改的内容。",
        }
    )

    changed = original.model_copy(
        update={
            "chunk": changed_chunk,
        }
    )

    with pytest.raises(
        CompetitionBM25FusionChunkMismatchError,
        match="内容不一致",
    ):
        fuse_competition_bm25_queries(
            question_only_hits=(original,),
            question_with_options_hits=(changed,),
        )

    with pytest.raises(
        EmptyCompetitionBM25QueryFusionError,
        match="都没有候选",
    ):
        fuse_competition_bm25_queries(
            question_only_hits=(),
            question_with_options_hits=(),
        )

    with pytest.raises(
        InvalidCompetitionBM25FusionTopKError,
        match="大于等于 1",
    ):
        fuse_competition_bm25_queries(
            question_only_hits=(original,),
            question_with_options_hits=(),
            top_k=0,
        )