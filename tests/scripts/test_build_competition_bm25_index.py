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
    build_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts import (
    build_competition_bm25_index
    as bm25_script,
)


def _build_test_corpus(
    tmp_path: Path,
):
    source = CompetitionKnowledgeSource(
        source_id="src_pdf_bm25_gate",
        doc_id=(
            "doc_src_pdf_bm25_gate_"
            "0123456789abcdef01234567"
        ),
        title="BM25 Gate 测试来源",
        source_type="pdf",
        relative_path="pdf/bm25-gate.pdf",
        sha256="a" * 64,
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
                    "商业银行资本充足率"
                    "应当满足监管要求。"
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
                    "项目\t数值\n"
                    "资本充足率\t12%"
                ),
                page=1,
                pdf_bbox=(
                    10.0,
                    20.0,
                    300.0,
                    400.0,
                ),
                table_index=0,
                table_rows=(
                    (
                        "项目",
                        "数值",
                    ),
                    (
                        "资本充足率",
                        "12%",
                    ),
                ),
            ),
        ),
    )

    chunk_document = (
        CompetitionChunkDocument(
            source=source,
            chunks=(
                build_competition_text_chunks(
                    document
                )
            ),
        )
    )

    return build_competition_chunk_corpus(
        documents=(chunk_document,),
        output_root=(
            tmp_path / "corpora"
        ),
    )


def test_resolve_corpus_directory(
    tmp_path: Path,
) -> None:
    corpus = _build_test_corpus(
        tmp_path
    )

    explicit = (
        bm25_script
        .resolve_corpus_directory(
            corpus_directory=(
                corpus.corpus_directory
            ),
            corpus_root=(
                tmp_path / "unused"
            ),
        )
    )

    assert explicit == (
        corpus.corpus_directory
    )

    discovered = (
        bm25_script
        .resolve_corpus_directory(
            corpus_directory=None,
            corpus_root=(
                tmp_path / "corpora"
            ),
        )
    )

    assert discovered == (
        corpus.corpus_directory
    )


def test_resolve_rejects_multiple_corpora(
    tmp_path: Path,
) -> None:
    corpus_root = (
        tmp_path / "corpora"
    )

    _build_test_corpus(tmp_path)

    extra = (
        corpus_root
        / (
            "competition_corpus_"
            "0123456789abcdef"
        )
    )

    extra.mkdir()

    (
        extra / "corpus_manifest.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        extra / "chunks.jsonl"
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="发现多个",
    ):
        bm25_script.resolve_corpus_directory(
            corpus_directory=None,
            corpus_root=corpus_root,
        )


def test_pipeline_builds_real_loadable_index(
    tmp_path: Path,
) -> None:
    corpus = _build_test_corpus(
        tmp_path
    )

    result = (
        bm25_script
        .build_competition_bm25_pipeline(
            corpus_directory=(
                corpus.corpus_directory
            ),
            output_root=(
                tmp_path / "indexes"
            ),
            smoke_query="资本充足率",
            top_k=5,
        )
    )

    assert (
        result.index_result
        .manifest.document_count
        == corpus.manifest.chunk_count
    )

    assert (
        result.index_result.index.chunks
        == corpus.chunks
    )

    assert result.smoke_hits

    assert (
        result.index_result
        .index_path
        .is_file()
    )

    assert (
        result.index_result
        .metadata_path
        .is_file()
    )

    assert (
        result.index_result
        .manifest_path
        .is_file()
    )


def test_main_prints_index_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = _build_test_corpus(
        tmp_path
    )

    exit_code = bm25_script.main(
        [
            "--corpus",
            str(
                corpus.corpus_directory
            ),
            "--output-root",
            str(
                tmp_path / "indexes"
            ),
            "--smoke-query",
            "资本充足率",
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out

    assert (
        "Competition QA-used BM25 Index"
        in output
    )

    assert "Smoke hits:" in output
    assert "Top hit:" in output
    assert "Saved:" in output