from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.tokenization import (
    DeterministicChineseBigramTokenizer,
)
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
from app.services.competition_bm25_index import (
    CompetitionBM25IndexIdentityConflictError,
    CorruptCompetitionBM25IndexError,
    build_competition_bm25_index,
    load_competition_bm25_index,
)
from app.services.competition_corpus import (
    build_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)


def _chunk_document(
    *,
    source_suffix: str = "main",
    source_sha256: str = "1" * 64,
) -> CompetitionChunkDocument:
    source_id = (
        f"src_pdf_bm25_{source_suffix}"
    )

    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title="BM25 测试文档",
        source_type="pdf",
        relative_path=(
            f"pdf/{source_suffix}.pdf"
        ),
        sha256=source_sha256,
    )

    document = CompetitionTextDocument(
        source=source,
        blocks=(
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:"
                    "00000"
                ),
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type="pdf",
                block_index=0,
                block_type="page_text",
                text=(
                    "商业银行应当持续满足"
                    "资本充足率监管要求。"
                ),
                page=1,
            ),
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:"
                    "00001"
                ),
                source_id=source.source_id,
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

    return CompetitionChunkDocument(
        source=source,
        chunks=(
            build_competition_text_chunks(
                document
            )
        ),
    )


def _build_corpus(
    *,
    root: Path,
    source_suffix: str = "main",
    source_sha256: str = "1" * 64,
):
    return build_competition_chunk_corpus(
        documents=(
            _chunk_document(
                source_suffix=(
                    source_suffix
                ),
                source_sha256=(
                    source_sha256
                ),
            ),
        ),
        output_root=root,
    )


def test_build_load_and_search_round_trip(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        root=tmp_path / "corpora"
    )

    tokenizer = (
        DeterministicChineseBigramTokenizer()
    )

    result = build_competition_bm25_index(
        corpus_directory=(
            corpus.corpus_directory
        ),
        output_root=(
            tmp_path / "indexes"
        ),
        tokenizer=tokenizer,
    )

    assert result.index_path.is_file()
    assert result.metadata_path.is_file()
    assert result.manifest_path.is_file()

    assert (
        result.manifest.document_count
        == len(corpus.chunks)
    )

    assert (
        result.manifest.corpus_id
        == corpus.manifest.corpus_id
    )

    loaded = load_competition_bm25_index(
        result.index_directory,
        corpus_directory=(
            corpus.corpus_directory
        ),
    )

    assert (
        loaded.manifest
        == result.manifest
    )

    assert (
        loaded.index.chunks
        == corpus.chunks
    )

    hits = loaded.index.search(
        query="资本保证金",
        tokenizer=tokenizer,
        top_k=5,
    )

    assert hits
    assert hits[0].chunk_type == "table"

    repeated = build_competition_bm25_index(
        corpus_directory=(
            corpus.corpus_directory
        ),
        output_root=(
            tmp_path / "indexes"
        ),
        tokenizer=tokenizer,
    )

    assert (
        repeated.index_directory
        == result.index_directory
    )

    assert (
        repeated.manifest
        == result.manifest
    )


def test_load_rejects_tampered_index_json(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        root=tmp_path / "corpora"
    )

    result = build_competition_bm25_index(
        corpus_directory=(
            corpus.corpus_directory
        ),
        output_root=(
            tmp_path / "indexes"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    original = result.index_path.read_bytes()

    result.index_path.write_bytes(
        original + b" "
    )

    with pytest.raises(
        CorruptCompetitionBM25IndexError,
        match="index.json SHA256",
    ):
        load_competition_bm25_index(
            result.index_directory
        )


def test_load_rejects_tampered_metadata(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        root=tmp_path / "corpora"
    )

    result = build_competition_bm25_index(
        corpus_directory=(
            corpus.corpus_directory
        ),
        output_root=(
            tmp_path / "indexes"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    original = (
        result.metadata_path.read_bytes()
    )

    result.metadata_path.write_bytes(
        original + b"\n"
    )

    with pytest.raises(
        CorruptCompetitionBM25IndexError,
        match="metadata.jsonl SHA256",
    ):
        load_competition_bm25_index(
            result.index_directory
        )


def test_load_rejects_wrong_corpus_binding(
    tmp_path: Path,
) -> None:
    first_corpus = _build_corpus(
        root=tmp_path / "first-corpora",
        source_suffix="first",
        source_sha256="1" * 64,
    )

    second_corpus = _build_corpus(
        root=tmp_path / "second-corpora",
        source_suffix="second",
        source_sha256="2" * 64,
    )

    result = build_competition_bm25_index(
        corpus_directory=(
            first_corpus.corpus_directory
        ),
        output_root=(
            tmp_path / "indexes"
        ),
        tokenizer=(
            DeterministicChineseBigramTokenizer()
        ),
    )

    with pytest.raises(
        CompetitionBM25IndexIdentityConflictError,
        match="身份不一致",
    ):
        load_competition_bm25_index(
            result.index_directory,
            corpus_directory=(
                second_corpus
                .corpus_directory
            ),
        )