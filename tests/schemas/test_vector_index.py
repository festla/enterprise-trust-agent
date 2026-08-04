from datetime import (
    datetime,
    timezone,
)

import pytest
from pydantic import ValidationError

from app.schemas.embedding import (
    EmbeddingSpec,
    calculate_embedding_spec_sha256,
)
from app.schemas.enums import (
    ChunkStrategy,
    ReportType,
)
from app.schemas.vector_index import (
    VectorIndexManifest,
)


NOW = datetime(
    2026,
    7,
    27,
    tzinfo=timezone.utc,
)

REPORT_ID = "midea_group_2024"

SPEC = EmbeddingSpec(
    provider="test",
    model_name="fake_embedding",
    model_version="fake_v1",
    dimension=3,
)


def build_manifest(
    **overrides: object,
) -> VectorIndexManifest:
    values = {
        "index_id": (
            f"vector_index_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        "chunk_dataset_id": (
            f"chunk_dataset_{REPORT_ID}_"
            f"{'b' * 24}"
        ),
        "report_id": REPORT_ID,
        "company_id": "midea_group",
        "fiscal_year": 2024,
        "report_type": (
            ReportType.ANNUAL_REPORT
        ),
        "document_id": (
            f"doc_{REPORT_ID}_{'c' * 24}"
        ),
        "chunk_strategy": (
            ChunkStrategy.FIXED_LENGTH
        ),
        "chunk_dataset_manifest_sha256": (
            "d" * 64
        ),
        "source_chunks_jsonl_sha256": (
            "e" * 64
        ),
        "embedding_spec": SPEC,
        "embedding_spec_sha256": (
            calculate_embedding_spec_sha256(
                SPEC
            )
        ),
        "embedding_input_sha256": (
            "f" * 64
        ),
        "numpy_version": "2.3.5",
        "vector_count": 2,
        "vector_dimension": 3,
        "metadata_record_count": 2,
        "vectors_sha256": "1" * 64,
        "metadata_jsonl_sha256": "2" * 64,
        "quality_gate_passed": True,
        "quality_gate_errors": (),
        "created_at": NOW,
    }

    values.update(overrides)

    return VectorIndexManifest(
        **values
    )


def test_validate_vector_index_manifest(
) -> None:
    manifest = build_manifest()

    assert manifest.vector_count == 2


def test_reject_embedding_spec_hash_mismatch(
) -> None:
    with pytest.raises(
        ValidationError,
        match="embedding_spec_sha256",
    ):
        build_manifest(
            embedding_spec_sha256="0" * 64
        )


def test_reject_vector_metadata_count_mismatch(
) -> None:
    with pytest.raises(
        ValidationError,
        match="向量数量",
    ):
        build_manifest(
            metadata_record_count=1
        )