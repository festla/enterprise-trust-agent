from __future__ import annotations

import pytest

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_gold import (
    CompetitionGoldChunkReference,
    CompetitionGoldFactRecord,
)
from app.schemas.competition_retrieval import (
    CompetitionBM25Hit,
    CompetitionDenseHit,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_gold_retrieval_eval import (
    evaluate_competition_gold_fact,
    summarize_competition_gold_retrieval,
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
        title="Gold 检索评测来源",
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

    chunk = (
        build_competition_text_chunks(
            document
        )[0]
    )

    return CompetitionBM25Hit(
        rank=rank,
        score=10.0 / rank,
        chunk=chunk,
    )


def _record(
    *,
    case_id: str,
    evidence_mode: str,
    references: tuple[
        CompetitionGoldChunkReference,
        ...,
    ],
) -> CompetitionGoldFactRecord:
    return CompetitionGoldFactRecord(
        case_id=case_id,
        fact_index=0,
        fact="测试监管事实。",
        expected_source_id=(
            "src_0000000000000001"
        ),
        review_status="confirmed",
        evidence_mode=evidence_mode,
        gold_chunks=references,
    )


def test_single_chunk_uses_exact_chunk_id(
) -> None:
    unrelated = _hit(
        source_number=1,
        text="无关内容。",
        rank=1,
    )

    direct = _hit(
        source_number=2,
        text="测试监管事实。",
        rank=2,
    )

    result = evaluate_competition_gold_fact(
        record=_record(
            case_id="Q101",
            evidence_mode="single_chunk",
            references=(
                CompetitionGoldChunkReference(
                    chunk_id=direct.chunk_id,
                    role="direct",
                ),
            ),
        ),
        hits=(
            unrelated,
            direct,
        ),
    )

    assert not result.direct_hit_at(1)
    assert result.direct_hit_at(2)
    assert result.complete_gold_hit_at(2)
    assert result.gold_chunk_recall_at(2) == 1.0
    assert (
        result.direct_reciprocal_rank_at(2)
        == 0.5
    )


def test_parent_child_separates_direct_and_complete(
) -> None:
    direct = _hit(
        source_number=3,
        text="直接证据。",
        rank=1,
    )

    parent = _hit(
        source_number=4,
        text="父级上下文。",
        rank=3,
    )

    result = evaluate_competition_gold_fact(
        record=_record(
            case_id="Q102",
            evidence_mode="parent_child",
            references=(
                CompetitionGoldChunkReference(
                    chunk_id=parent.chunk_id,
                    role="parent_context",
                ),
                CompetitionGoldChunkReference(
                    chunk_id=direct.chunk_id,
                    role="direct",
                ),
            ),
        ),
        hits=(
            direct,
            parent,
        ),
    )

    assert result.direct_hit_at(1)
    assert result.any_gold_hit_at(1)
    assert not result.complete_gold_hit_at(1)
    assert result.gold_chunk_recall_at(1) == 0.5

    assert result.complete_gold_hit_at(3)
    assert result.gold_chunk_recall_at(3) == 1.0


def test_multi_chunk_reports_partial_recall(
) -> None:
    direct = _hit(
        source_number=5,
        text="直接证据。",
        rank=2,
    )

    first_support = _hit(
        source_number=6,
        text="补充证据一。",
        rank=4,
    )

    missing_support = _hit(
        source_number=7,
        text="补充证据二。",
        rank=6,
    )

    result = evaluate_competition_gold_fact(
        record=_record(
            case_id="Q103",
            evidence_mode="multi_chunk",
            references=(
                CompetitionGoldChunkReference(
                    chunk_id=direct.chunk_id,
                    role="direct",
                ),
                CompetitionGoldChunkReference(
                    chunk_id=(
                        first_support.chunk_id
                    ),
                    role="supporting",
                ),
                CompetitionGoldChunkReference(
                    chunk_id=(
                        missing_support.chunk_id
                    ),
                    role="supporting",
                ),
            ),
        ),
        hits=(
            direct,
            first_support,
        ),
    )

    assert result.direct_hit_at(4)
    assert result.any_gold_hit_at(4)
    assert not result.complete_gold_hit_at(4)
    assert (
        result.gold_chunk_recall_at(4)
        == pytest.approx(2 / 3)
    )


def test_summary_aggregates_fact_metrics(
) -> None:
    matched = _hit(
        source_number=8,
        text="命中事实。",
        rank=1,
    )

    missing = _hit(
        source_number=9,
        text="未命中事实。",
        rank=2,
    )

    successful = evaluate_competition_gold_fact(
        record=_record(
            case_id="Q104",
            evidence_mode="single_chunk",
            references=(
                CompetitionGoldChunkReference(
                    chunk_id=matched.chunk_id,
                    role="direct",
                ),
            ),
        ),
        hits=(matched,),
    )

    empty = evaluate_competition_gold_fact(
        record=_record(
            case_id="Q105",
            evidence_mode="single_chunk",
            references=(
                CompetitionGoldChunkReference(
                    chunk_id=missing.chunk_id,
                    role="direct",
                ),
            ),
        ),
        hits=(),
    )

    summary = summarize_competition_gold_retrieval(
        (
            successful,
            empty,
        ),
        cutoffs=(1, 3),
    )

    at_one = summary.at(1)

    assert at_one.fact_direct_hit_rate == 0.5
    assert at_one.fact_any_gold_hit_rate == 0.5
    assert (
        at_one.fact_complete_gold_hit_rate
        == 0.5
    )
    assert at_one.mean_gold_chunk_recall == 0.5
    assert (
        at_one.mean_direct_reciprocal_rank
        == 0.5
    )


def test_rejects_pending_gold_record(
) -> None:
    pending = CompetitionGoldFactRecord(
        case_id="Q106",
        fact_index=0,
        fact="尚未审核的事实。",
        expected_source_id=(
            "src_0000000000000001"
        ),
    )

    with pytest.raises(
        ValueError,
        match="只接受 confirmed",
    ):
        evaluate_competition_gold_fact(
            record=pending,
            hits=(),
        )


def test_rejects_duplicate_hit_rank_and_chunk_id(
) -> None:
    first = _hit(
        source_number=10,
        text="第一条。",
        rank=1,
    )

    same_rank = _hit(
        source_number=11,
        text="第二条。",
        rank=1,
    )

    record = _record(
        case_id="Q107",
        evidence_mode="single_chunk",
        references=(
            CompetitionGoldChunkReference(
                chunk_id=first.chunk_id,
                role="direct",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="重复 rank",
    ):
        evaluate_competition_gold_fact(
            record=record,
            hits=(
                first,
                same_rank,
            ),
        )

    repeated_chunk = (
        first.model_copy(
            update={
                "rank": 2,
                "score": 5.0,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="重复 chunk_id",
    ):
        evaluate_competition_gold_fact(
            record=record,
            hits=(
                first,
                repeated_chunk,
            ),
        )

def test_gold_eval_accepts_dense_hits(
) -> None:
    bm25_hit = _hit(
        source_number=1,
        text="Dense 直接证据。",
        rank=1,
    )

    dense_hit = CompetitionDenseHit(
        rank=1,
        score=0.92,
        chunk=bm25_hit.chunk,
    )

    record = _record(
        case_id="Q199",
        evidence_mode="single_chunk",
        references=(
            CompetitionGoldChunkReference(
                chunk_id=(
                    dense_hit.chunk_id
                ),
                role="direct",
            ),
        ),
    )

    result = (
        evaluate_competition_gold_fact(
            record=record,
            hits=(dense_hit,),
        )
    )

    assert result.direct_hit_at(1)
    assert result.complete_gold_hit_at(1)
    assert (
        result.gold_chunk_recall_at(1)
        == 1.0
    )