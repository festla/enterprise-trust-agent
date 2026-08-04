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
)
from app.schemas.reranker import (
    RerankedRetrievalHit,
    RerankerRuntimeConfig,
    RerankerSpec,
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


def build_source_hit() -> RetrievalHit:
    text = (
        "合并资产负债表中资产总计"
        "为若干元"
    )

    return RetrievalHit(
        rank=2,
        retriever_type="dense",
        score_type=(
            "cosine_similarity"
        ),
        score=0.8,
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


def build_hybrid_hit(
) -> HybridRetrievalHit:
    return HybridRetrievalHit.from_source_hit(
        rank=4,
        rrf_score=(
            1 / 62
            + 1 / 61
        ),
        source_hit=build_source_hit(),
        dense_rank=2,
        bm25_rank=1,
    )


def build_reranked_hit(
) -> RerankedRetrievalHit:
    return (
        RerankedRetrievalHit
        .from_hybrid_hit(
            rank=1,
            reranker_score=3.25,
            source_hit=(
                build_hybrid_hit()
            ),
        )
    )


def test_build_reranker_spec() -> None:
    spec = RerankerSpec(
        model_revision=(
            "test_revision_001"
        )
    )

    assert spec.provider == (
        "sentence_transformers_cross_encoder"
    )

    assert spec.architecture == (
        "cross_encoder"
    )

    assert spec.model_name == (
        "BAAI/bge-reranker-base"
    )

    assert spec.max_length == 512

    assert spec.score_type == (
        "cross_encoder_logit"
    )

    assert (
        spec.input_template_version
        == "query_chunk_v1"
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("model_revision", ""),
        ("max_length", 0),
        ("max_length", 7),
    ),
)
def test_reject_invalid_reranker_spec(
    field_name: str,
    invalid_value: object,
) -> None:
    values: dict[str, object] = {
        "model_revision": (
            "test_revision_001"
        ),
    }

    values[field_name] = invalid_value

    with pytest.raises(
        ValidationError,
    ):
        RerankerSpec(
            **values,
        )


def test_reject_unknown_spec_field(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        RerankerSpec(
            model_revision=(
                "test_revision_001"
            ),
            unknown_parameter=True,
        )


def test_reranker_spec_is_frozen(
) -> None:
    spec = RerankerSpec(
        model_revision=(
            "test_revision_001"
        )
    )

    with pytest.raises(
        ValidationError,
    ):
        spec.max_length = 256  # type: ignore[misc]


def test_build_runtime_config() -> None:
    config = RerankerRuntimeConfig()

    assert config.batch_size == 8
    assert config.device == "cpu"

    assert (
        config.rerank_candidate_count
        == 50
    )

    assert config.return_count == 5
    assert not config.local_files_only


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("batch_size", 0),
        ("rerank_candidate_count", 0),
        ("return_count", 0),
        ("device", ""),
    ),
)
def test_reject_invalid_runtime_config(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        RerankerRuntimeConfig(
            **{
                field_name: invalid_value,
            }
        )


def test_reject_return_count_above_candidate_count(
) -> None:
    with pytest.raises(
        ValidationError,
        match="return_count",
    ):
        RerankerRuntimeConfig(
            rerank_candidate_count=10,
            return_count=11,
        )


def test_build_reranked_hit_from_hybrid(
) -> None:
    hit = build_reranked_hit()

    assert hit.rank == 1

    assert hit.retriever_type == (
        "hybrid_reranker"
    )

    assert hit.score_type == (
        "cross_encoder_logit"
    )

    assert hit.score == pytest.approx(
        3.25
    )

    assert hit.reranker_score == (
        pytest.approx(3.25)
    )

    assert hit.rrf_rank == 4

    assert hit.rrf_score == (
        pytest.approx(
            build_hybrid_hit()
            .rrf_score
        )
    )

    assert hit.dense_rank == 2
    assert hit.bm25_rank == 1

    assert hit.source_retrievers == (
        "dense",
        "bm25",
    )

    assert hit.pdf_page == 156

    assert hit.chunk_id == (
        build_hybrid_hit().chunk_id
    )


def test_allow_negative_reranker_score(
) -> None:
    hit = (
        RerankedRetrievalHit
        .from_hybrid_hit(
            rank=1,
            reranker_score=-2.5,
            source_hit=(
                build_hybrid_hit()
            ),
        )
    )

    assert hit.score == pytest.approx(
        -2.5
    )

    assert hit.reranker_score == (
        pytest.approx(-2.5)
    )


def test_reject_incorrect_source_retrievers(
) -> None:
    values = (
        build_reranked_hit()
        .model_dump()
    )

    values["source_retrievers"] = (
        "dense",
    )

    with pytest.raises(
        ValidationError,
        match="source_retrievers",
    ):
        RerankedRetrievalHit.model_validate(
            values
        )


def test_reject_score_mismatch(
) -> None:
    values = (
        build_reranked_hit()
        .model_dump()
    )

    values["score"] = (
        values["reranker_score"]
        + 0.1
    )

    with pytest.raises(
        ValidationError,
        match="reranker_score",
    ):
        RerankedRetrievalHit.model_validate(
            values
        )


def test_reject_non_finite_reranker_score(
) -> None:
    with pytest.raises(
        ValidationError,
    ):
        (
            RerankedRetrievalHit
            .from_hybrid_hit(
                rank=1,
                reranker_score=float("nan"),
                source_hit=(
                    build_hybrid_hit()
                ),
            )
        )