from __future__ import annotations

import pytest

from app.rag.rrf import (
    DuplicateRRFChunkError,
    EmptyRRFInputError,
    InvalidRetrieverHitsError,
    InvalidRRFTopKError,
    RRFSourceMismatchError,
    reciprocal_rank_fusion,
)
from app.schemas.enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
)
from app.schemas.hybrid_retrieval import (
    RRFConfig,
)
from app.schemas.retrieval import (
    RetrievalHit,
)


REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

CHUNK_DATASET_ID = (
    f"chunk_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)


CHUNK_VALUES = {
    "a": {
        "pdf_page": 10,
        "text": "合并利润表营业收入",
    },
    "b": {
        "pdf_page": 20,
        "text": "合并资产负债表资产总计",
    },
    "c": {
        "pdf_page": 30,
        "text": (
            "经营活动产生的"
            "现金流量净额"
        ),
    },
    "d": {
        "pdf_page": 40,
        "text": "应收账款期末余额",
    },
}


def build_hit(
    *,
    suffix: str,
    rank: int,
    retriever_type: str,
    pdf_page: int | None = None,
    text: str | None = None,
) -> RetrievalHit:
    values = CHUNK_VALUES[suffix]

    active_pdf_page = (
        int(values["pdf_page"])
        if pdf_page is None
        else pdf_page
    )

    active_text = (
        str(values["text"])
        if text is None
        else text
    )

    score_type = (
        "cosine_similarity"
        if retriever_type == "dense"
        else "bm25"
    )

    score = (
        max(
            0.1,
            0.95 - rank * 0.05,
        )
        if retriever_type == "dense"
        else 30.0 - rank
    )

    return RetrievalHit(
        rank=rank,
        retriever_type=retriever_type,
        score_type=score_type,
        score=score,
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{suffix * 24}"
        ),
        chunk_dataset_id=(
            CHUNK_DATASET_ID
        ),
        company_id="midea_group",
        report_id=REPORT_ID,
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=DOCUMENT_ID,
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{active_pdf_page:04d}"
        ),
        pdf_page=active_pdf_page,
        printed_page=active_pdf_page,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        chunk_index=active_pdf_page,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        source_start_char=0,
        source_end_char=len(active_text),
        section_path=(),
        text=active_text,
    )


def build_dense_hits(
) -> tuple[RetrievalHit, ...]:
    return (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="dense",
        ),
        build_hit(
            suffix="b",
            rank=2,
            retriever_type="dense",
        ),
        build_hit(
            suffix="c",
            rank=3,
            retriever_type="dense",
        ),
    )


def build_bm25_hits(
) -> tuple[RetrievalHit, ...]:
    return (
        build_hit(
            suffix="b",
            rank=1,
            retriever_type="bm25",
        ),
        build_hit(
            suffix="d",
            rank=2,
            retriever_type="bm25",
        ),
        build_hit(
            suffix="a",
            rank=3,
            retriever_type="bm25",
        ),
    )


def test_rank_candidates_by_rrf_score(
) -> None:
    hits = reciprocal_rank_fusion(
        dense_hits=build_dense_hits(),
        bm25_hits=build_bm25_hits(),
        top_k=4,
    )

    assert [
        hit.chunk_id
        for hit in hits
    ] == [
        (
            f"chunk_{REPORT_ID}_"
            f"{'b' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'d' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'c' * 24}"
        ),
    ]

    assert [
        hit.rank
        for hit in hits
    ] == [1, 2, 3, 4]


def test_rrf_score_matches_manual_formula(
) -> None:
    hits = reciprocal_rank_fusion(
        dense_hits=build_dense_hits(),
        bm25_hits=build_bm25_hits(),
        top_k=4,
    )

    target = next(
        hit
        for hit in hits
        if hit.chunk_id.endswith(
            "b" * 24
        )
    )

    expected_score = (
        1 / (60 + 2)
        + 1 / (60 + 1)
    )

    assert target.dense_rank == 2
    assert target.bm25_rank == 1

    assert target.rrf_score == (
        pytest.approx(
            expected_score
        )
    )

    assert target.score == (
        pytest.approx(
            expected_score
        )
    )


def test_preserve_single_source_candidates(
) -> None:
    hits = reciprocal_rank_fusion(
        dense_hits=build_dense_hits(),
        bm25_hits=build_bm25_hits(),
        top_k=4,
    )

    dense_only = next(
        hit
        for hit in hits
        if hit.chunk_id.endswith(
            "c" * 24
        )
    )

    bm25_only = next(
        hit
        for hit in hits
        if hit.chunk_id.endswith(
            "d" * 24
        )
    )

    assert dense_only.dense_rank == 3
    assert dense_only.bm25_rank is None
    assert dense_only.source_retrievers == (
        "dense",
    )

    assert bm25_only.dense_rank is None
    assert bm25_only.bm25_rank == 2
    assert bm25_only.source_retrievers == (
        "bm25",
    )


def test_equal_scores_use_stable_chunk_id(
) -> None:
    dense_hits = (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="dense",
        ),
        build_hit(
            suffix="b",
            rank=2,
            retriever_type="dense",
        ),
    )

    bm25_hits = (
        build_hit(
            suffix="c",
            rank=1,
            retriever_type="bm25",
        ),
        build_hit(
            suffix="d",
            rank=2,
            retriever_type="bm25",
        ),
    )

    hits = reciprocal_rank_fusion(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        top_k=4,
    )

    assert [
        hit.chunk_id
        for hit in hits
    ] == [
        (
            f"chunk_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'c' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'b' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'d' * 24}"
        ),
    ]


def test_candidate_count_limits_are_applied(
) -> None:
    config = RRFConfig(
        dense_candidate_count=1,
        bm25_candidate_count=1,
    )

    hits = reciprocal_rank_fusion(
        dense_hits=build_dense_hits(),
        bm25_hits=build_bm25_hits(),
        config=config,
        top_k=10,
    )

    assert len(hits) == 2

    assert {
        hit.chunk_id
        for hit in hits
    } == {
        (
            f"chunk_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        (
            f"chunk_{REPORT_ID}_"
            f"{'b' * 24}"
        ),
    }


def test_top_k_larger_than_candidates_is_safe(
) -> None:
    hits = reciprocal_rank_fusion(
        dense_hits=build_dense_hits(),
        bm25_hits=build_bm25_hits(),
        top_k=100,
    )

    assert len(hits) == 4


def test_reject_invalid_top_k() -> None:
    with pytest.raises(
        InvalidRRFTopKError,
    ):
        reciprocal_rank_fusion(
            dense_hits=build_dense_hits(),
            bm25_hits=build_bm25_hits(),
            top_k=0,
        )


@pytest.mark.parametrize(
    ("dense_hits", "bm25_hits"),
    (
        ((), build_bm25_hits()),
        (build_dense_hits(), ()),
    ),
)
def test_reject_missing_retriever_results(
    dense_hits: tuple[
        RetrievalHit,
        ...,
    ],
    bm25_hits: tuple[
        RetrievalHit,
        ...,
    ],
) -> None:
    with pytest.raises(
        EmptyRRFInputError,
    ):
        reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
        )


def test_reject_non_continuous_dense_ranks(
) -> None:
    dense_hits = (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="dense",
        ),
        build_hit(
            suffix="b",
            rank=3,
            retriever_type="dense",
        ),
    )

    with pytest.raises(
        InvalidRetrieverHitsError,
        match="连续递增",
    ):
        reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=build_bm25_hits(),
        )


def test_reject_wrong_retriever_type(
) -> None:
    wrong_dense_hits = (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="bm25",
        ),
    )

    with pytest.raises(
        InvalidRetrieverHitsError,
        match="retriever_type",
    ):
        reciprocal_rank_fusion(
            dense_hits=wrong_dense_hits,
            bm25_hits=build_bm25_hits(),
        )


def test_reject_duplicate_chunk_in_one_source(
) -> None:
    dense_hits = (
        build_hit(
            suffix="a",
            rank=1,
            retriever_type="dense",
        ),
        build_hit(
            suffix="a",
            rank=2,
            retriever_type="dense",
        ),
    )

    with pytest.raises(
        DuplicateRRFChunkError,
        match="重复 chunk_id",
    ):
        reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=build_bm25_hits(),
        )


def test_reject_source_metadata_mismatch(
) -> None:
    dense_hits = (
        build_hit(
            suffix="b",
            rank=1,
            retriever_type="dense",
        ),
    )

    bm25_hits = (
        build_hit(
            suffix="b",
            rank=1,
            retriever_type="bm25",
            pdf_page=21,
        ),
    )

    with pytest.raises(
        RRFSourceMismatchError,
        match="来源元数据不一致",
    ):
        reciprocal_rank_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
        )