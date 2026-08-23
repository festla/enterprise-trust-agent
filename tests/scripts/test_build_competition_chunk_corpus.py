from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    load_competition_chunk_corpus,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts import (
    build_competition_chunk_corpus
    as corpus_script,
)


def _case(
    case_id: str,
    source_type: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        case_id=case_id,
        source_type=source_type,
    )


def _manifest_source(
    source_id: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        source_id=source_id,
        actual_filename=(
            f"{source_id}.dat"
        ),
    )


def _chunk_document(
    *,
    source_id: str,
    source_type: str,
    sha_char: str,
    include_table: bool = False,
) -> CompetitionChunkDocument:
    source = CompetitionKnowledgeSource(
        source_id=source_id,
        doc_id=(
            f"doc_{source_id}_"
            "0123456789abcdef01234567"
        ),
        title=f"测试来源 {source_id}",
        source_type=source_type,
        relative_path=(
            f"{source_type}/{source_id}"
        ),
        sha256=sha_char * 64,
    )

    text_block_type = (
        "page_text"
        if source_type == "pdf"
        else "paragraph"
    )

    blocks = [
        CompetitionTextBlock(
            block_id=(
                f"block:{source.doc_id}:00000"
            ),
            source_id=source.source_id,
            doc_id=source.doc_id,
            source_type=source_type,
            block_index=0,
            block_type=text_block_type,
            text="这是用于构建 Corpus 的正文。",
            page=(
                1
                if source_type == "pdf"
                else None
            ),
            paragraph_index=(
                0
                if source_type == "word"
                else None
            ),
        )
    ]

    if include_table:
        blocks.append(
            CompetitionTextBlock(
                block_id=(
                    f"block:{source.doc_id}:00001"
                ),
                source_id=source.source_id,
                doc_id=source.doc_id,
                source_type=source_type,
                block_index=1,
                block_type="table",
                text="项目\t金额\n资本保证金\t100",
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
                        "金额",
                    ),
                    (
                        "资本保证金",
                        "100",
                    ),
                ),
            )
        )

    document = CompetitionTextDocument(
        source=source,
        blocks=tuple(blocks),
    )

    return CompetitionChunkDocument(
        source=source,
        chunks=(
            build_competition_text_chunks(
                document,
                max_chars=900,
            )
        ),
    )


def test_resolve_qa_used_text_sources_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_source = _manifest_source(
        "src_pdf_test"
    )
    word_source = _manifest_source(
        "src_word_test"
    )

    manifest = (
        pdf_source,
        word_source,
    )

    cases = (
        _case(
            "qa_pdf_b",
            "pdf",
        ),
        _case(
            "qa_excel",
            "excel",
        ),
        _case(
            "qa_word",
            "word",
        ),
        _case(
            "qa_pdf_a",
            "pdf",
        ),
    )

    source_ids = {
        "qa_pdf_a": "src_pdf_test",
        "qa_pdf_b": "src_pdf_test",
        "qa_word": "src_word_test",
    }

    class FakeResolver:
        def __init__(
            self,
            provided_manifest: object,
        ) -> None:
            assert tuple(
                provided_manifest
            ) == manifest

        def resolve(
            self,
            case: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                source_id=(
                    source_ids[case.case_id]
                )
            )

    monkeypatch.setattr(
        corpus_script,
        "CompetitionSourceResolver",
        FakeResolver,
    )

    (
        text_case_count,
        resolved,
    ) = (
        corpus_script
        .resolve_qa_used_text_sources(
            cases=cases,
            manifest=manifest,
        )
    )

    assert text_case_count == 3

    assert [
        source.source_id
        for _, source in resolved
    ] == [
        "src_pdf_test",
        "src_word_test",
    ]

    # 同一 PDF 的代表问题应为最小 case_id。
    assert resolved[0][0].case_id == (
        "qa_pdf_a"
    )


def test_build_chunk_documents_parses_each_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_case = _case(
        "qa_pdf",
        "pdf",
    )
    word_case = _case(
        "qa_word",
        "word",
    )

    pdf_manifest_source = (
        _manifest_source(
            "src_pdf_test"
        )
    )
    word_manifest_source = (
        _manifest_source(
            "src_word_test"
        )
    )

    parsed_documents = {
        "src_pdf_test": (
            _chunk_document(
                source_id="src_pdf_test",
                source_type="pdf",
                sha_char="a",
                include_table=True,
            )
        ),
        "src_word_test": (
            _chunk_document(
                source_id="src_word_test",
                source_type="word",
                sha_char="b",
            )
        ),
    }

    monkeypatch.setattr(
        corpus_script,
        "build_competition_question",
        lambda case: case.case_id,
    )

    def fake_parse(
        *,
        question: str,
        source: object,
        attachments_root: Path,
    ) -> CompetitionTextDocument:
        assert question.startswith(
            "qa_"
        )
        assert attachments_root == tmp_path

        chunk_document = (
            parsed_documents[
                source.source_id
            ]
        )

        blocks = tuple(
            CompetitionTextBlock(
                block_id=(
                    f"block:"
                    f"{chunk_document.source.doc_id}:"
                    f"{index:05d}"
                ),
                source_id=(
                    chunk_document.source.source_id
                ),
                doc_id=(
                    chunk_document.source.doc_id
                ),
                source_type=(
                    chunk.source_type
                ),
                block_index=index,
                block_type=(
                    "page_text"
                    if chunk.source_type == "pdf"
                    else "paragraph"
                ),
                text=chunk.text,
                page=(
                    chunk.page_start
                    if chunk.source_type == "pdf"
                    else None
                ),
                paragraph_index=(
                    index
                    if chunk.source_type == "word"
                    else None
                ),
            )
            for index, chunk in enumerate(
                chunk_document.chunks
            )
            if chunk.chunk_type == "text"
        )

        return CompetitionTextDocument(
            source=chunk_document.source,
            blocks=blocks,
        )

    monkeypatch.setattr(
        corpus_script,
        "parse_competition_text_document",
        fake_parse,
    )

    documents = (
        corpus_script.build_chunk_documents(
            resolved_sources=(
                (
                    pdf_case,
                    pdf_manifest_source,
                ),
                (
                    word_case,
                    word_manifest_source,
                ),
            ),
            attachments_root=tmp_path,
            max_chars=900,
        )
    )

    assert len(documents) == 2
    assert {
        document.source.source_id
        for document in documents
    } == {
        "src_pdf_test",
        "src_word_test",
    }

    assert all(
        document.chunks
        for document in documents
    )


def test_pipeline_builds_and_loads_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = (
        _case(
            "qa_pdf",
            "pdf",
        ),
        _case(
            "qa_word",
            "word",
        ),
        _case(
            "qa_excel",
            "excel",
        ),
    )

    manifest = (
        _manifest_source(
            "src_pdf_test"
        ),
        _manifest_source(
            "src_word_test"
        ),
    )

    documents = (
        _chunk_document(
            source_id="src_pdf_test",
            source_type="pdf",
            sha_char="a",
            include_table=True,
        ),
        _chunk_document(
            source_id="src_word_test",
            source_type="word",
            sha_char="b",
        ),
    )

    monkeypatch.setattr(
        corpus_script,
        "load_competition_qa_excel",
        lambda path: cases,
    )

    monkeypatch.setattr(
        corpus_script,
        "build_competition_source_manifest",
        lambda path: manifest,
    )

    monkeypatch.setattr(
        corpus_script,
        "resolve_qa_used_text_sources",
        lambda **kwargs: (
            2,
            (
                (
                    cases[0],
                    manifest[0],
                ),
                (
                    cases[1],
                    manifest[1],
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        corpus_script,
        "build_chunk_documents",
        lambda **kwargs: documents,
    )

    result = (
        corpus_script
        .build_qa_used_text_corpus(
            qa_file=tmp_path / "QA.xlsx",
            attachments_root=(
                tmp_path / "attachments"
            ),
            output_root=(
                tmp_path / "corpora"
            ),
            max_chars=900,
        )
    )

    assert result.qa_case_count == 3
    assert result.text_qa_case_count == 2
    assert result.resolved_source_count == 2

    assert result.corpus.manifest.source_count == 2
    assert result.corpus.manifest.doc_count == 2
    assert result.corpus.manifest.table_chunk_count == 1

    loaded = load_competition_chunk_corpus(
        result.corpus.corpus_directory
    )

    assert loaded.manifest == (
        result.corpus.manifest
    )
    assert loaded.chunks == (
        result.corpus.chunks
    )