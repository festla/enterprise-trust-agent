from __future__ import annotations

from app.schemas.competition_evidence import (
    CompetitionKnowledgeSource,
)
from app.schemas.competition_text import (
    CompetitionTextBlock,
    CompetitionTextDocument,
)
from app.services.competition_text_chunker import (
    build_competition_text_chunks,
)
from scripts.smoke_competition_pdf_table_chunks import (
    PdfTableChunkSmokeSummary,
    collect_gate_errors,
    validate_pdf_table_chunks,
)


def _pdf_document(
) -> CompetitionTextDocument:
    source = CompetitionKnowledgeSource(
        source_id="src_pdf_table_test",
        doc_id=(
            "doc_src_pdf_table_test_"
            "0123456789abcdef01234567"
        ),
        title="测试 PDF",
        source_type="pdf",
        relative_path="test.pdf",
        sha256="0" * 64,
    )

    rows = (
        (
            "项目",
            "金额",
        ),
        (
            "资本保证金",
            "100",
        ),
    )

    return CompetitionTextDocument(
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
                text="表格前正文。",
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
                table_rows=rows,
            ),
        ),
    )


def test_validate_pdf_table_chunks_accepts_generated_chunks(
) -> None:
    document = _pdf_document()

    chunks = (
        build_competition_text_chunks(
            document
        )
    )

    errors = validate_pdf_table_chunks(
        document=document,
        chunks=chunks,
    )

    assert errors == []


def test_validate_pdf_table_chunks_rejects_missing_chunks(
) -> None:
    document = _pdf_document()

    chunks = tuple(
        chunk
        for chunk in (
            build_competition_text_chunks(
                document
            )
        )
        if chunk.chunk_type != "table"
    )

    errors = validate_pdf_table_chunks(
        document=document,
        chunks=chunks,
    )

    assert any(
        "missing table chunks" in error
        for error in errors
    )


def test_collect_gate_errors_rejects_incomplete_corpus(
) -> None:
    summary = PdfTableChunkSmokeSummary(
        source_count=5,
        success_count=4,
        failure_count=1,
        table_source_count=1,
        table_block_count=10,
        table_chunk_count=9,
        failures=(
            (
                "src_failed",
                "RuntimeError",
            ),
        ),
    )

    errors = collect_gate_errors(
        summary=summary,
        expected_source_count=6,
        minimum_table_sources=2,
        minimum_table_blocks=37,
    )

    assert len(errors) == 6