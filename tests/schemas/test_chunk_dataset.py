from datetime import (
    datetime,
    timezone,
)

import pytest
from pydantic import ValidationError

from app.schemas.chunk import (
    FixedLengthChunkingConfig,
)
from app.schemas.chunk_dataset import (
    ChunkDatasetManifest,
    calculate_chunking_config_sha256,
)
from app.schemas.enums import (
    ChunkStrategy,
    ReportType,
)


REPORT_ID = "midea_group_2024"

CONFIG = FixedLengthChunkingConfig(
    max_chars=800,
    overlap_chars=120,
)

NOW = datetime(
    2026,
    7,
    26,
    tzinfo=timezone.utc,
)


def build_manifest(
    **overrides: object,
) -> ChunkDatasetManifest:
    values = {
        "dataset_id": (
            f"chunk_dataset_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        "page_dataset_id": (
            f"page_dataset_{REPORT_ID}_"
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
        "page_dataset_manifest_sha256": (
            "d" * 64
        ),
        "source_pages_jsonl_sha256": (
            "e" * 64
        ),
        "report_snapshot_sha256": (
            "f" * 64
        ),
        "strategy": (
            ChunkStrategy.FIXED_LENGTH
        ),
        "chunker_name": "fixed_length",
        "chunker_version": (
            "fixed_length_chunker_v1"
        ),
        "chunking_config": CONFIG,
        "chunking_config_sha256": (
            calculate_chunking_config_sha256(
                CONFIG
            )
        ),
        "chunks_jsonl_sha256": "1" * 64,
        "input_page_count": 2,
        "eligible_page_count": 2,
        "chunked_page_count": 2,
        "skipped_page_count": 0,
        "skipped_page_ids": (),
        "chunk_record_count": 4,
        "chunk_char_count_total": 2000,
        "quality_gate_passed": True,
        "quality_gate_errors": (),
        "quality_warnings": (),
        "created_at": NOW,
    }

    values.update(overrides)

    return ChunkDatasetManifest(
        **values
    )


def test_validate_manifest() -> None:
    manifest = build_manifest()

    assert manifest.chunk_record_count == 4


def test_reject_config_hash_mismatch() -> None:
    with pytest.raises(
        ValidationError,
        match="chunking_config_sha256",
    ):
        build_manifest(
            chunking_config_sha256="0" * 64
        )


def test_reject_inconsistent_page_counts() -> None:
    with pytest.raises(
        ValidationError,
        match="eligible 与 skipped",
    ):
        build_manifest(
            skipped_page_count=1,
        )