from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import (
    ChunkStrategy,
    PageMappingStatus,
    ReportType,
)
from app.schemas.hybrid_retrieval import (
    HybridRetrievalHit,
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


def build_source_hit(
    *,
    retriever_type: str = "dense",
) -> RetrievalHit:
    text = (
        "合并资产负债表中资产总计"
        "为若干元"
    )

    score_type = (
        "cosine_similarity"
        if retriever_type == "dense"
        else "bm25"
    )

    score = (
        0.8
        if retriever_type == "dense"
        else 18.5
    )

    return RetrievalHit(
        rank=1,
        retriever_type=retriever_type,
        score_type=score_type,
        score=score,
        chunk_id=(
            f"chunk_{REPORT_ID}_"
            f"{'c' * 24}"
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
            f"{DOCUMENT_ID}_page_0156"
        ),
        pdf_page=156,
        printed_page=155,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        chunk_index=0,
        strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        source_start_char=0,
        source_end_char=len(text),
        section_path=(),
        text=text,
    )


def build_hybrid_hit() -> HybridRetrievalHit:
    return HybridRetrievalHit.from_source_hit(
        rank=1,
        rrf_score=(
            1 / 61
            + 1 / 62
        ),
        source_hit=build_source_hit(),
        dense_rank=1,
        bm25_rank=2,
    )


def test_build_default_rrf_config() -> None:
    config = RRFConfig()

    assert config.fusion_method == (
        "reciprocal_rank_fusion"
    )

    assert config.fusion_version == (
        "rrf_v1"
    )

    assert config.rank_constant == 60
    assert config.dense_candidate_count == 50
    assert config.bm25_candidate_count == 50


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("rank_constant", 0),
        ("rank_constant", -1),
        ("dense_candidate_count", 0),
        ("bm25_candidate_count", 0),
    ),
)
def test_reject_invalid_rrf_config(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        RRFConfig(
            **{
                field_name: invalid_value,
            }
        )


def test_reject_unknown_rrf_config_field(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        RRFConfig(
            unknown_parameter=True,
        )


def test_rrf_config_is_frozen() -> None:
    config = RRFConfig()

    with pytest.raises(
        ValidationError,
    ):
        config.rank_constant = 30  # type: ignore[misc]


def test_build_hybrid_hit_from_source(
) -> None:
    hit = build_hybrid_hit()

    assert hit.rank == 1
    assert hit.retriever_type == (
        "hybrid_rrf"
    )
    assert hit.score_type == "rrf"

    assert hit.dense_rank == 1
    assert hit.bm25_rank == 2

    assert hit.source_retrievers == (
        "dense",
        "bm25",
    )

    assert hit.score == pytest.approx(
        hit.rrf_score
    )

    assert hit.chunk_id == (
        build_source_hit().chunk_id
    )

    assert hit.pdf_page == 156


def test_build_dense_only_hybrid_hit(
) -> None:
    hit = HybridRetrievalHit.from_source_hit(
        rank=1,
        rrf_score=1 / 61,
        source_hit=build_source_hit(),
        dense_rank=1,
        bm25_rank=None,
    )

    assert hit.source_retrievers == (
        "dense",
    )

    assert hit.dense_rank == 1
    assert hit.bm25_rank is None


def test_build_bm25_only_hybrid_hit(
) -> None:
    hit = HybridRetrievalHit.from_source_hit(
        rank=1,
        rrf_score=1 / 61,
        source_hit=build_source_hit(
            retriever_type="bm25"
        ),
        dense_rank=None,
        bm25_rank=1,
    )

    assert hit.source_retrievers == (
        "bm25",
    )

    assert hit.dense_rank is None
    assert hit.bm25_rank == 1


def test_reject_hybrid_hit_without_source_rank(
) -> None:
    with pytest.raises(
        ValidationError,
        match="至少需要一个",
    ):
        HybridRetrievalHit.from_source_hit(
            rank=1,
            rrf_score=0,
            source_hit=build_source_hit(),
            dense_rank=None,
            bm25_rank=None,
        )


def test_reject_incorrect_source_retrievers(
) -> None:
    values = build_hybrid_hit().model_dump()

    values["source_retrievers"] = (
        "bm25",
    )

    with pytest.raises(
        ValidationError,
        match="source_retrievers",
    ):
        HybridRetrievalHit.model_validate(
            values
        )


def test_reject_score_rrf_score_mismatch(
) -> None:
    values = build_hybrid_hit().model_dump()

    values["score"] = (
        values["rrf_score"]
        + 0.1
    )

    with pytest.raises(
        ValidationError,
        match="rrf_score",
    ):
        HybridRetrievalHit.model_validate(
            values
        )