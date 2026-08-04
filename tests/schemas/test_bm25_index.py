from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import pytest
from pydantic import ValidationError

from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)
from app.schemas.bm25_index import (
    BM25IndexData,
    BM25IndexDocumentRecord,
    BM25IndexManifest,
)
from app.schemas.enums import (
    ChunkStrategy,
    ReportType,
)


REPORT_ID = "midea_group_2024"


def build_index_data() -> BM25IndexData:
    return BM25IndexData(
        document_records=(
            BM25IndexDocumentRecord(
                chunk_id=(
                    f"chunk_{REPORT_ID}_"
                    f"{'1' * 24}"
                ),
                document_length=2,
                term_frequencies={
                    "营业": 1,
                    "收入": 1,
                },
            ),
            BM25IndexDocumentRecord(
                chunk_id=(
                    f"chunk_{REPORT_ID}_"
                    f"{'2' * 24}"
                ),
                document_length=2,
                term_frequencies={
                    "净利": 1,
                    "利润": 1,
                },
            ),
        ),
        document_frequencies={
            "营业": 1,
            "收入": 1,
            "净利": 1,
            "利润": 1,
        },
        document_count=2,
        vocabulary_size=4,
        total_token_count=4,
        average_document_length=2,
    )


def build_manifest() -> BM25IndexManifest:
    tokenizer_spec = BM25TokenizerSpec()
    config = BM25Config()

    return BM25IndexManifest(
        index_id=(
            f"bm25_index_{REPORT_ID}_"
            f"{'a' * 24}"
        ),
        chunk_dataset_id=(
            f"chunk_dataset_{REPORT_ID}_"
            f"{'b' * 24}"
        ),
        report_id=REPORT_ID,
        company_id="midea_group",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        document_id=(
            f"doc_{REPORT_ID}_"
            f"{'c' * 24}"
        ),
        chunk_strategy=(
            ChunkStrategy.FIXED_LENGTH
        ),
        chunk_dataset_manifest_sha256=(
            "d" * 64
        ),
        source_chunks_jsonl_sha256=(
            "e" * 64
        ),
        tokenizer_spec=tokenizer_spec,
        tokenizer_spec_sha256=(
            calculate_bm25_tokenizer_spec_sha256(
                tokenizer_spec
            )
        ),
        bm25_config=config,
        bm25_config_sha256=(
            calculate_bm25_config_sha256(
                config
            )
        ),
        tokenized_corpus_sha256=(
            "f" * 64
        ),
        document_count=2,
        metadata_record_count=2,
        vocabulary_size=4,
        total_token_count=4,
        average_document_length=2,
        index_json_sha256="1" * 64,
        metadata_jsonl_sha256="2" * 64,
        quality_gate_passed=True,
        quality_gate_errors=(),
        created_at=datetime.now(
            timezone.utc
        ),
    )


def test_build_valid_index_data() -> None:
    data = build_index_data()

    assert data.document_count == 2
    assert data.total_token_count == 4
    assert data.vocabulary_size == 4


def test_reject_incorrect_document_frequency(
) -> None:
    with pytest.raises(
        ValidationError,
        match="document_frequencies",
    ):
        BM25IndexData(
            document_records=(
                BM25IndexDocumentRecord(
                    chunk_id=(
                        f"chunk_{REPORT_ID}_"
                        f"{'1' * 24}"
                    ),
                    document_length=1,
                    term_frequencies={
                        "营业": 1,
                    },
                ),
                BM25IndexDocumentRecord(
                    chunk_id=(
                        f"chunk_{REPORT_ID}_"
                        f"{'2' * 24}"
                    ),
                    document_length=1,
                    term_frequencies={
                        "收入": 1,
                    },
                ),
            ),
            document_frequencies={
                # 不超过 document_count，
                # 但和两个文档的实际统计不一致。
                "营业": 2,
                "收入": 1,
            },
            document_count=2,
            vocabulary_size=2,
            total_token_count=2,
            average_document_length=1,
        )

def test_reject_frequency_larger_than_document_count(
) -> None:
    with pytest.raises(
        ValidationError,
        match="不能超过文档数量",
    ):
        BM25IndexData(
            document_records=(
                BM25IndexDocumentRecord(
                    chunk_id=(
                        f"chunk_{REPORT_ID}_"
                        f"{'1' * 24}"
                    ),
                    document_length=1,
                    term_frequencies={
                        "营业": 1,
                    },
                ),
            ),
            document_frequencies={
                "营业": 2,
            },
            document_count=1,
            vocabulary_size=1,
            total_token_count=1,
            average_document_length=1,
        )

        
def test_reject_incorrect_document_length(
) -> None:
    with pytest.raises(
        ValidationError,
        match="document_length",
    ):
        BM25IndexDocumentRecord(
            chunk_id=(
                f"chunk_{REPORT_ID}_"
                f"{'1' * 24}"
            ),
            document_length=3,
            term_frequencies={
                "营业": 1,
                "收入": 1,
            },
        )


def test_build_valid_manifest() -> None:
    manifest = build_manifest()

    assert manifest.index_type == (
        "exact_bm25"
    )

    assert manifest.document_count == 2


def test_reject_tokenizer_hash_mismatch(
) -> None:
    values = build_manifest().model_dump()

    values["tokenizer_spec_sha256"] = (
        "0" * 64
    )

    with pytest.raises(
        ValidationError,
        match="tokenizer_spec_sha256",
    ):
        BM25IndexManifest.model_validate(
            values
        )

def test_reject_naive_created_at() -> None:
    values = build_manifest().model_dump()

    values["created_at"] = datetime(
        2026,
        7,
        30,
        12,
        0,
    )

    with pytest.raises(
        ValidationError,
        match="时区",
    ):
        BM25IndexManifest.model_validate(
            values
        )