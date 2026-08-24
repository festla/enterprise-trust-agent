from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.competition_chunk import (
    CompetitionChunkDocument,
)
from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_corpus import (
    CorruptCompetitionCorpusError,
    InvalidCompetitionCorpusError,
    build_competition_chunk_corpus,
    load_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _chunk_document(
    *,
    sha256: str | None = "1" * 64,
) -> CompetitionChunkDocument:
    source = CompetitionKnowledgeSource(
        source_id=(
            "src_pdf_corpus_test"
        ),
        doc_id=(
            "doc_src_pdf_corpus_test_"
            "0123456789abcdef01234567"
        ),
        title="测试 PDF Corpus",
        source_type="pdf",
        relative_path="pdf/test.pdf",
        sha256=sha256,
    )

    text_document = (
        CompetitionTextDocument(
            source=source,
            blocks=(
                CompetitionTextBlock(
                    block_id=(
                        f"block:"
                        f"{source.doc_id}:"
                        "00000"
                    ),
                    source_id=(
                        source.source_id
                    ),
                    doc_id=source.doc_id,
                    source_type="pdf",
                    block_index=0,
                    block_type="page_text",
                    text="表格前正文。",
                    page=1,
                ),
                CompetitionTextBlock(
                    block_id=(
                        f"block:"
                        f"{source.doc_id}:"
                        "00001"
                    ),
                    source_id=(
                        source.source_id
                    ),
                    doc_id=source.doc_id,
                    source_type="pdf",
                    block_index=1,
                    block_type="table",
                    text=(
                        "项目\t金额\n"
                        "资本保证金\t100"
                    ),
                    page=1,
                    pdf_bbox=(
                        100.0,
                        200.0,
                        500.0,
                        400.0,
                    ),
                    table_index=0,
                    table_rows=(
                        (
                            "项目",
                            "金额",
                        ),
                        (
                            "资本保证金",
                            "100",
                        ),
                    ),
                ),
            ),
        )
    )

    chunks = (
        build_competition_text_chunks(
            text_document
        )
    )

    return CompetitionChunkDocument(
        source=source,
        chunks=chunks,
    )


def test_build_and_load_corpus_round_trip(
    tmp_path: Path,
) -> None:
    result = (
        build_competition_chunk_corpus(
            documents=(
                _chunk_document(),
            ),
            output_root=tmp_path,
        )
    )

    assert result.chunks_path.is_file()
    assert result.manifest_path.is_file()

    assert result.manifest.source_count == 1
    assert result.manifest.doc_count == 1
    assert result.manifest.chunk_count == 2
    assert result.manifest.text_chunk_count == 1
    assert result.manifest.table_chunk_count == 1

    loaded = (
        load_competition_chunk_corpus(
            result.corpus_directory
        )
    )

    assert loaded.manifest == result.manifest
    assert loaded.chunks == result.chunks

    repeated = (
        build_competition_chunk_corpus(
            documents=(
                _chunk_document(),
            ),
            output_root=tmp_path,
        )
    )

    assert (
        repeated.corpus_directory
        == result.corpus_directory
    )

    assert (
        repeated.manifest
        == result.manifest
    )

def test_build_corpus_preserves_zero_chunk_type(
    tmp_path: Path,
) -> None:
    original_document = (
        _chunk_document()
    )

    text_only_document = (
        CompetitionChunkDocument(
            source=(
                original_document.source
            ),
            chunks=tuple(
                chunk
                for chunk
                in original_document.chunks
                if (
                    chunk.chunk_type
                    == "text"
                )
            ),
        )
    )

    result = (
        build_competition_chunk_corpus(
            documents=(
                text_only_document,
            ),
            output_root=tmp_path,
        )
    )

    assert (
        result.manifest.chunk_type_counts
        == {
            "text": 1,
            "table": 0,
        }
    )

    assert (
        result.manifest.text_chunk_count
        == 1
    )

    assert (
        result.manifest.table_chunk_count
        == 0
    )

    loaded = (
        load_competition_chunk_corpus(
            result.corpus_directory
        )
    )

    assert (
        loaded.manifest.chunk_type_counts
        == {
            "text": 1,
            "table": 0,
        }
    )

def test_build_corpus_requires_source_sha256(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        InvalidCompetitionCorpusError,
        match="sha256",
    ):
        build_competition_chunk_corpus(
            documents=(
                _chunk_document(
                    sha256=None
                ),
            ),
            output_root=tmp_path,
        )


def test_load_corpus_rejects_tampered_chunks(
    tmp_path: Path,
) -> None:
    result = (
        build_competition_chunk_corpus(
            documents=(
                _chunk_document(),
            ),
            output_root=tmp_path,
        )
    )

    original = (
        result.chunks_path
        .read_bytes()
    )

    result.chunks_path.write_bytes(
        original + b"\n"
    )

    with pytest.raises(
        CorruptCompetitionCorpusError,
        match="SHA256",
    ):
        load_competition_chunk_corpus(
            result.corpus_directory
        )