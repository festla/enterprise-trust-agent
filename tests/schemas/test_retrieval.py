from app.schemas.embedding import (
    EmbeddingSpec,
    calculate_embedding_spec_sha256,
)
from app.schemas.enums import ReportType
from app.schemas.retrieval import (
    RetrievalFilter,
)


def build_spec(
    *,
    model_version: str = "test_v1",
) -> EmbeddingSpec:
    return EmbeddingSpec(
        provider="test",
        model_name="fake_embedding",
        model_version=model_version,
        dimension=3,
        normalize_embeddings=True,
    )


def test_embedding_spec_hash_is_stable() -> None:
    first = build_spec()
    second = build_spec()

    assert (
        calculate_embedding_spec_sha256(
            first
        )
        == calculate_embedding_spec_sha256(
            second
        )
    )


def test_embedding_version_changes_hash() -> None:
    first = build_spec(
        model_version="test_v1"
    )

    second = build_spec(
        model_version="test_v2"
    )

    assert (
        calculate_embedding_spec_sha256(
            first
        )
        != calculate_embedding_spec_sha256(
            second
        )
    )


def test_retrieval_filter_normalizes_values(
) -> None:
    filters = RetrievalFilter(
        company_ids=(
            "midea_group",
            "hisense_home",
            "midea_group",
        ),
        fiscal_years=(
            2025,
            2024,
            2025,
        ),
        report_types=(
            ReportType.ANNUAL_REPORT,
            ReportType.ANNUAL_REPORT,
        ),
        pdf_pages=(3, 1, 3),
    )

    assert filters.company_ids == (
        "hisense_home",
        "midea_group",
    )

    assert filters.fiscal_years == (
        2024,
        2025,
    )

    assert filters.report_types == (
        ReportType.ANNUAL_REPORT,
    )

    assert filters.pdf_pages == (
        1,
        3,
    )