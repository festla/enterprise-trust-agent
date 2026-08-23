from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import pytest

from app.schemas.bm25 import (
    BM25Config,
    BM25TokenizerSpec,
    calculate_bm25_config_sha256,
    calculate_bm25_tokenizer_spec_sha256,
)
from app.schemas.competition_bm25 import (
    CompetitionBM25DocumentRecord,
    CompetitionBM25IndexData,
    CompetitionBM25IndexManifest,
    build_competition_bm25_index_id,
    calculate_competition_bm25_identity,
)


def _index_data() -> (
    CompetitionBM25IndexData
):
    records = (
        CompetitionBM25DocumentRecord(
            chunk_id="chunk:doc:test:00000",
            document_length=3,
            term_frequencies={
                "资本": 2,
                "监管": 1,
            },
        ),
        CompetitionBM25DocumentRecord(
            chunk_id="chunk:doc:test:00001",
            document_length=2,
            term_frequencies={
                "资本": 1,
                "风险": 1,
            },
        ),
    )

    return CompetitionBM25IndexData(
        document_records=records,
        document_frequencies={
            "资本": 2,
            "监管": 1,
            "风险": 1,
        },
        document_count=2,
        vocabulary_size=3,
        total_token_count=5,
        average_document_length=2.5,
    )


def _manifest() -> (
    CompetitionBM25IndexManifest
):
    tokenizer_spec = (
        BM25TokenizerSpec()
    )
    config = BM25Config()

    payload: dict[str, object] = {
        "schema_version": 1,
        "corpus_id": (
            "competition_corpus_"
            "0123456789abcdef"
        ),
        "corpus_identity_sha256": (
            "1" * 64
        ),
        "corpus_manifest_sha256": (
            "2" * 64
        ),
        "corpus_chunks_jsonl_sha256": (
            "3" * 64
        ),
        "corpus_source_count": 14,
        "corpus_doc_count": 14,
        "corpus_chunk_count": 2,
        "tokenizer_spec": (
            tokenizer_spec
        ),
        "tokenizer_spec_sha256": (
            calculate_bm25_tokenizer_spec_sha256(
                tokenizer_spec
            )
        ),
        "bm25_config": config,
        "bm25_config_sha256": (
            calculate_bm25_config_sha256(
                config
            )
        ),
        "tokenized_corpus_sha256": (
            "4" * 64
        ),
        "index_type": "exact_bm25",
        "index_version": (
            "competition_exact_bm25_v1"
        ),
        "document_count": 2,
        "metadata_record_count": 2,
        "vocabulary_size": 3,
        "total_token_count": 5,
        "average_document_length": 2.5,
        "index_json_sha256": (
            "5" * 64
        ),
        "metadata_jsonl_sha256": (
            "6" * 64
        ),
        "quality_gate_passed": True,
        "quality_gate_errors": (),
        "created_at": datetime.now(
            timezone.utc
        ),
    }

    identity = (
        calculate_competition_bm25_identity(
            payload
        )
    )

    return CompetitionBM25IndexManifest(
        **payload,
        index_identity_sha256=identity,
        index_id=(
            build_competition_bm25_index_id(
                identity
            )
        ),
    )


def test_document_record_accepts_competition_chunk_id(
) -> None:
    record = (
        CompetitionBM25DocumentRecord(
            chunk_id=(
                "chunk:doc_src_test:"
                "00001"
            ),
            document_length=3,
            term_frequencies={
                "风险": 1,
                "资本": 2,
            },
        )
    )

    assert record.chunk_id == (
        "chunk:doc_src_test:00001"
    )

    assert list(
        record.term_frequencies
    ) == [
        "资本",
        "风险",
    ]


def test_index_data_validates_token_statistics(
) -> None:
    data = _index_data()

    assert data.document_count == 2
    assert data.total_token_count == 5
    assert (
        data.average_document_length
        == 2.5
    )

    payload = data.model_dump(
        mode="python"
    )

    payload[
        "document_frequencies"
    ]["资本"] = 1

    with pytest.raises(
        ValueError,
        match="document_frequencies",
    ):
        CompetitionBM25IndexData(
            **payload
        )


def test_manifest_accepts_consistent_identity(
) -> None:
    manifest = _manifest()

    assert manifest.index_id.startswith(
        "competition_bm25_index_"
    )

    assert (
        manifest.document_count
        == manifest.corpus_chunk_count
    )


def test_manifest_rejects_identity_tampering(
) -> None:
    manifest = _manifest()

    payload = manifest.model_dump(
        mode="python"
    )

    payload["vocabulary_size"] = 4

    with pytest.raises(
        ValueError,
        match=(
            "index_identity_sha256"
        ),
    ):
        CompetitionBM25IndexManifest(
            **payload
        )


def test_index_id_rejects_invalid_sha256(
) -> None:
    with pytest.raises(
        ValueError,
        match="64位小写十六进制",
    ):
        build_competition_bm25_index_id(
            "invalid"
        )