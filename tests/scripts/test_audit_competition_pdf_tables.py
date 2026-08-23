from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.audit_competition_pdf_tables as audit_module
from app.schemas.competition import (
    CompetitionQaCase,
)
from app.services.competition_source_resolver import (
    build_competition_source_manifest,
)
from scripts.audit_competition_pdf_tables import (
    PdfTableCorpusAudit,
    PdfTableSourceAudit,
    audit_pdf_table_source,
    collect_gate_errors,
    collect_qa_used_pdf_sources,
)


def _case(
    *,
    case_id: str,
    source_type: str,
    file_label: str,
) -> CompetitionQaCase:
    qa_type = (
        "单事实检索"
        if source_type in {"pdf", "word"}
        else "表格取数"
    )

    return CompetitionQaCase(
        case_id=case_id,
        source_type=source_type,
        difficulty="easy",
        difficulty_cn="简单",
        qa_type=qa_type,
        question="测试问题是什么？",
        option_a="选项甲",
        option_b="选项乙",
        option_c="选项丙",
        option_d="选项丁",
        answer="A",
        answer_text="选项甲",
        evidence="测试证据",
        source_title="测试文档",
        file_label=file_label,
    )


def test_collect_pdf_sources_deduplicates_cases(
    tmp_path: Path,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    (
        attachments_root
        / "first.pdf"
    ).write_bytes(
        b"%PDF-first"
    )

    (
        attachments_root
        / "second.pdf"
    ).write_bytes(
        b"%PDF-second"
    )

    (
        attachments_root
        / "word.docx"
    ).write_bytes(
        b"PK\x03\x04word"
    )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    cases = (
        _case(
            case_id="Q001",
            source_type="pdf",
            file_label="first.pdf",
        ),
        _case(
            case_id="Q002",
            source_type="pdf",
            file_label="first.pdf",
        ),
        _case(
            case_id="Q003",
            source_type="pdf",
            file_label="second.pdf",
        ),
        _case(
            case_id="Q004",
            source_type="word",
            file_label="word.docx",
        ),
    )

    sources = collect_qa_used_pdf_sources(
        cases=cases,
        source_manifest=source_manifest,
    )

    assert len(sources) == 2

    assert all(
        source.extension == ".pdf"
        for source in sources.values()
    )


def test_audit_pdf_table_source_counts_structure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = (
        tmp_path
        / "source.pdf"
    )

    pdf_path.write_bytes(
        b"%PDF-test"
    )

    table = SimpleNamespace(
        row_count=2,
        col_count=2,
        bbox=(
            10.0,
            20.0,
            200.0,
            100.0,
        ),
        extract=lambda: [
            [
                "项目",
                "数值",
            ],
            [
                "资本",
                "100",
            ],
        ],
    )

    pages = [
        SimpleNamespace(
            find_tables=lambda: (
                SimpleNamespace(
                    tables=[
                        table
                    ]
                )
            )
        ),
        SimpleNamespace(
            find_tables=lambda: (
                SimpleNamespace(
                    tables=[]
                )
            )
        ),
    ]

    class FakeDocument:
        is_pdf = True
        needs_pass = False
        page_count = 2

        def load_page(
            self,
            page_index: int,
        ):
            return pages[
                page_index
            ]

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        audit_module.pymupdf,
        "open",
        lambda path: FakeDocument(),
    )

    audit = audit_pdf_table_source(
        source_id=(
            "src_0000000000000000"
        ),
        path=pdf_path,
    )

    assert audit.page_count == 2
    assert audit.table_pages == (1,)
    assert audit.table_count == 1
    assert audit.extracted_table_count == 1
    assert audit.total_rows == 2
    assert audit.total_cells == 4
    assert audit.non_empty_cells == 4
    assert audit.invalid_table_count == 0


def test_gate_rejects_missing_tables_and_failures(
) -> None:
    source_audit = PdfTableSourceAudit(
        source_id=(
            "src_0000000000000000"
        ),
        page_count=3,
        table_pages=(),
        table_count=0,
        extracted_table_count=0,
        detection_failure_count=1,
        extraction_failure_count=0,
        invalid_table_count=0,
        total_rows=0,
        total_cells=0,
        non_empty_cells=0,
    )

    corpus_audit = PdfTableCorpusAudit(
        source_count=2,
        source_audits=(
            source_audit,
        ),
        failures=(
            (
                "src_1111111111111111",
                "RuntimeError",
            ),
        ),
    )

    errors = collect_gate_errors(
        audit=corpus_audit,
        expected_source_count=2,
        minimum_table_sources=1,
        minimum_tables=1,
    )

    assert any(
        "source audit failures"
        in error
        for error in errors
    )

    assert any(
        "detection failures"
        in error
        for error in errors
    )

    assert any(
        "Too few PDF table sources"
        in error
        for error in errors
    )

    assert any(
        "Too few PDF tables"
        in error
        for error in errors
    )