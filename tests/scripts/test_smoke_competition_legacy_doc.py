from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import scripts.smoke_competition_legacy_doc as smoke_module
from app.schemas.competition import (
    CompetitionQaCase,
)
from app.services.competition_source_resolver import (
    build_competition_source_manifest,
)
from app.services.document_ingestion import (
    calculate_file_sha256,
)
from scripts.smoke_competition_legacy_doc import (
    collect_gate_errors,
    collect_qa_used_legacy_doc_sources,
    run_legacy_doc_smoke,
)


def _case(
    *,
    case_id: str,
    file_label: str,
) -> CompetitionQaCase:
    return CompetitionQaCase(
        case_id=case_id,
        source_type="word",
        difficulty="easy",
        difficulty_cn="简单",
        qa_type="单事实检索",
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


def test_collect_legacy_sources_deduplicates_qa_cases(
    tmp_path: Path,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    (
        attachments_root
        / "legacy.doc"
    ).write_bytes(
        b"\xd0\xcf\x11\xe0"
        b"\xa1\xb1\x1a\xe1"
        b"legacy"
    )

    (
        attachments_root
        / "modern.docx"
    ).write_bytes(
        b"PK\x03\x04modern"
    )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    cases = (
        _case(
            case_id="Q001",
            file_label="legacy.doc",
        ),
        _case(
            case_id="Q002",
            file_label="legacy.doc",
        ),
        _case(
            case_id="Q003",
            file_label="modern.docx",
        ),
    )

    legacy_sources = (
        collect_qa_used_legacy_doc_sources(
            cases=cases,
            source_manifest=source_manifest,
        )
    )

    assert len(
        legacy_sources
    ) == 1

    (
        selected_case,
        selected_source,
    ) = next(
        iter(
            legacy_sources.values()
        )
    )

    assert (
        selected_case.case_id
        == "Q001"
    )

    assert (
        selected_source.extension
        == ".doc"
    )


def test_run_legacy_doc_smoke_preserves_original_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    legacy_path = (
        attachments_root
        / "legacy.doc"
    )

    legacy_path.write_bytes(
        b"\xd0\xcf\x11\xe0"
        b"\xa1\xb1\x1a\xe1"
        b"legacy"
    )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    cases = (
        _case(
            case_id="Q001",
            file_label="legacy.doc",
        ),
    )


    expected_sha256 = (
        calculate_file_sha256(
            legacy_path
        )
    )

    def fake_parse_competition_text_document(
        *,
        question,
        source,
        attachments_root,
    ):
        # Parser 不应收到 Gold 字段。
        assert not hasattr(
            question,
            "answer",
        )

        assert not hasattr(
            question,
            "evidence",
        )

        return SimpleNamespace(
            source=SimpleNamespace(
                source_id=source.source_id,
                relative_path=(
                    source.relative_path
                ),
                source_type="word",
                sha256=expected_sha256,
            ),
            blocks=(
                SimpleNamespace(
                    block_type="paragraph",
                    text="有效正文",
                ),
                SimpleNamespace(
                    block_type="table",
                    text="项目\t金额",
                ),
            ),
        )

    monkeypatch.setattr(
        smoke_module,
        "parse_competition_text_document",
        fake_parse_competition_text_document,
    )

    summary = run_legacy_doc_smoke(
        cases=cases,
        source_manifest=source_manifest,
        attachments_root=attachments_root,
    )

    assert summary.source_count == 1
    assert summary.success_count == 1
    assert summary.failure_count == 0
    assert summary.total_blocks == 2
    assert summary.total_chars > 0

    assert collect_gate_errors(
        summary=summary,
        expected_source_count=1,
    ) == []


def test_gate_rejects_source_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    attachments_root = (
        tmp_path
        / "attachments"
    )

    attachments_root.mkdir()

    (
        attachments_root
        / "legacy.doc"
    ).write_bytes(
        b"\xd0\xcf\x11\xe0"
        b"\xa1\xb1\x1a\xe1"
        b"legacy"
    )

    source_manifest = (
        build_competition_source_manifest(
            attachments_root
        )
    )

    cases = (
        _case(
            case_id="Q001",
            file_label="legacy.doc",
        ),
    )

    def fake_parse_competition_text_document(
        *,
        question,
        source,
        attachments_root,
    ):
        return SimpleNamespace(
            source=SimpleNamespace(
                source_id=source.source_id,
                relative_path=(
                    "temporary-converted.docx"
                ),
                source_type="word",
                sha256="wrong-sha256",
            ),
            blocks=(
                SimpleNamespace(
                    block_type="paragraph",
                    text="有效正文",
                ),
            ),
        )

    monkeypatch.setattr(
        smoke_module,
        "parse_competition_text_document",
        fake_parse_competition_text_document,
    )

    summary = run_legacy_doc_smoke(
        cases=cases,
        source_manifest=source_manifest,
        attachments_root=attachments_root,
    )

    assert summary.success_count == 0
    assert summary.failure_count == 1

    errors = collect_gate_errors(
        summary=summary,
        expected_source_count=1,
    )

    assert any(
        "parse failures" in error
        for error in errors
    )