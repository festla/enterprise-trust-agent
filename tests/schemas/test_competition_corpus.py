from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.schemas.competition_corpus import (
    CompetitionChunkCorpusManifest,
    build_competition_corpus_id,
    calculate_competition_corpus_identity,
)


def _manifest_payload(
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "corpus_scope": "qa_used_all",
        "chunker_version": (
            "competition_text_chunker_v1"
        ),
        "max_chars": 900,
        "chunks_jsonl_sha256": "a" * 64,
        "source_count": 2,
        "doc_count": 2,
        "chunk_count": 5,
        "chunk_char_count": 200,
        "text_chunk_count": 3,
        "table_chunk_count": 2,
        "source_type_counts": {
            "word": 1,
            "pdf": 1,
        },
        "chunk_type_counts": {
            "text": 3,
            "table": 2,
        },
        "source_records": (
            {
                "source_id": "src_0000000000000001",
                "doc_id": "doc_source_1",
                "source_type": "word",
                "relative_path": "word/test.docx",
                "source_sha256": "1" * 64,
                "chunk_count": 3,
                "chunk_char_count": 120,
                "text_chunk_count": 2,
                "table_chunk_count": 1,
            },
            {
                "source_id": "src_0000000000000002",
                "doc_id": "doc_source_2",
                "source_type": "pdf",
                "relative_path": "pdf/test.pdf",
                "source_sha256": "2" * 64,
                "chunk_count": 2,
                "chunk_char_count": 80,
                "text_chunk_count": 1,
                "table_chunk_count": 1,
            },
        ),
    }


def _build_manifest(
    payload: dict[
        str,
        object,
    ],
) -> CompetitionChunkCorpusManifest:
    identity = (
        calculate_competition_corpus_identity(
            payload
        )
    )

    return CompetitionChunkCorpusManifest(
        **payload,
        corpus_identity_sha256=identity,
        corpus_id=(
            build_competition_corpus_id(
                identity
            )
        ),
    )


def test_manifest_accepts_consistent_corpus(
) -> None:
    manifest = _build_manifest(
        _manifest_payload()
    )

    assert manifest.source_count == 2
    assert manifest.doc_count == 2
    assert manifest.chunk_count == 5
    assert manifest.text_chunk_count == 3
    assert manifest.table_chunk_count == 2

    assert manifest.corpus_id.startswith(
        "competition_corpus_"
    )


def test_manifest_rejects_aggregate_count_mismatch(
) -> None:
    payload = _manifest_payload()

    payload[
        "chunk_count"
    ] = 6

    with pytest.raises(
        ValidationError,
        match="chunk_count",
    ):
        _build_manifest(
            payload
        )


def test_manifest_rejects_duplicate_source_id(
) -> None:
    payload = _manifest_payload()

    records = list(
        copy.deepcopy(
            payload["source_records"]
        )
    )

    records[1][
        "source_id"
    ] = records[0][
        "source_id"
    ]

    payload[
        "source_records"
    ] = tuple(records)

    with pytest.raises(
        ValidationError,
        match="重复 source_id",
    ):
        _build_manifest(
            payload
        )


def test_manifest_rejects_identity_mismatch(
) -> None:
    payload = _manifest_payload()

    with pytest.raises(
        ValidationError,
        match=(
            "corpus_identity_sha256"
        ),
    ):
        CompetitionChunkCorpusManifest(
            **payload,
            corpus_identity_sha256=(
                "0" * 64
            ),
            corpus_id=(
                "competition_corpus_"
                "0000000000000000"
            ),
        )