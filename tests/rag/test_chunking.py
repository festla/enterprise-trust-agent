from datetime import (
    date,
    datetime,
    timezone,
)

import pytest
from pydantic import ValidationError

from app.rag.chunking import (
    build_chunk_dataset_id,
    build_fixed_length_chunks,
    build_fixed_length_chunks_for_pages,
    build_paragraph_chunks,
    build_paragraph_chunks_for_pages,
    detect_paragraph_spans,
    build_section_paragraph_chunks_for_pages,
    detect_section_headings,
)
from app.schemas.chunk import (
    FixedLengthChunkingConfig,
    ParagraphChunkingConfig,
    SectionParagraphChunkingConfig,
)
from app.schemas.enums import (
    DocumentQualityGrade,
    PageContentType,
    PageMappingStatus,
    PageParseStatus,
    RecordStatus,
    ReportType,
    Severity,
)
from app.schemas.page import ParsedPage
from app.schemas.page_dataset import (
    PageDatasetManifest,
)
from app.schemas.report import Report


REPORT_ID = "midea_group_2024"

DOCUMENT_ID = (
    f"doc_{REPORT_ID}_{'a' * 24}"
)

PAGE_DATASET_ID = (
    f"page_dataset_{REPORT_ID}_"
    f"{'b' * 24}"
)

NOW = datetime(
    2026,
    7,
    26,
    tzinfo=timezone.utc,
)


def build_report() -> Report:
    return Report(
        report_id=REPORT_ID,
        company_id="midea_group",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        title="美的集团2024年年度报告",
        publication_date=date(
            2025,
            3,
            29,
        ),
        source_name="test",
        quality_grade=(
            DocumentQualityGrade.A
        ),
        citation_risk=Severity.LOW,
        expected_pdf_page_count=2,
        active_document_id=DOCUMENT_ID,
        status=RecordStatus.ACTIVE,
        created_at=NOW,
        updated_at=NOW,
    )


def build_manifest() -> PageDatasetManifest:
    return PageDatasetManifest(
        dataset_id=PAGE_DATASET_ID,
        document_id=DOCUMENT_ID,
        report_id=REPORT_ID,
        source_sha256="c" * 64,
        mapping_sha256="d" * 64,
        pages_jsonl_sha256="e" * 64,
        parser_name="pymupdf",
        parser_version="1.28.0",
        normalizer_version=(
            "page_text_normalizer_v1"
        ),
        classifier_version=(
            "page_content_classifier_v1"
        ),
        total_pdf_pages=2,
        page_record_count=2,
        mapped_page_count=2,
        unmapped_page_count=0,
        raw_char_count_total=20,
        normalized_char_count_total=20,
        content_type_counts={
            PageContentType.TEXT: 2,
            PageContentType.EMPTY: 0,
            PageContentType.SCANNED: 0,
            PageContentType.MIXED: 0,
            PageContentType.UNKNOWN: 0,
        },
        parse_status_counts={
            PageParseStatus.SUCCESS: 2,
            PageParseStatus.PARSE_ERROR: 0,
        },
        quality_gate_passed=True,
        created_at=NOW,
    )


def build_page(
    *,
    pdf_page: int = 1,
    printed_page: int = 1,
    text: str = "abcdefghij",
) -> ParsedPage:
    return ParsedPage(
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        document_id=DOCUMENT_ID,
        report_id=REPORT_ID,
        pdf_page=pdf_page,
        printed_page=printed_page,
        mapping_status=(
            PageMappingStatus.MAPPED
        ),
        mapping_id=f"mapping_{pdf_page}",
        raw_text=text,
        normalized_text=text,
        raw_char_count=len(text),
        normalized_char_count=len(text),
        text_block_count=1,
        image_block_count=0,
        embedded_image_count=0,
        max_image_area_ratio=0,
        content_type=PageContentType.TEXT,
        parse_status=PageParseStatus.SUCCESS,
        parser_name="pymupdf",
        parser_version="1.28.0",
        parsed_at=NOW,
    )


def test_reject_invalid_overlap() -> None:
    with pytest.raises(ValidationError):
        FixedLengthChunkingConfig(
            max_chars=4,
            overlap_chars=4,
        )


def test_build_chunks_with_overlap() -> None:
    report = build_report()
    manifest = build_manifest()
    page = build_page()

    config = FixedLengthChunkingConfig(
        max_chars=4,
        overlap_chars=1,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    chunks = build_fixed_length_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    assert [
        chunk.text
        for chunk in chunks
    ] == [
        "abcd",
        "defg",
        "ghij",
    ]

    assert [
        (
            chunk.source_start_char,
            chunk.source_end_char,
        )
        for chunk in chunks
    ] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]

    for chunk in chunks:
        assert chunk.text == (
            page.normalized_text[
                chunk.source_start_char:
                chunk.source_end_char
            ]
        )


def test_same_input_produces_same_ids() -> None:
    report = build_report()
    manifest = build_manifest()
    page = build_page()

    config = FixedLengthChunkingConfig(
        max_chars=4,
        overlap_chars=1,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    first = build_fixed_length_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    second = build_fixed_length_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    assert first == second


def test_changed_config_changes_dataset_id() -> None:
    report = build_report()
    manifest = build_manifest()

    first_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=FixedLengthChunkingConfig(
            max_chars=4,
            overlap_chars=1,
        ),
    )

    second_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=FixedLengthChunkingConfig(
            max_chars=5,
            overlap_chars=1,
        ),
    )

    assert first_id != second_id


def test_duplicate_printed_page_not_merged() -> None:
    report = build_report()
    manifest = build_manifest()

    pages = (
        build_page(
            pdf_page=1,
            printed_page=33,
        ),
        build_page(
            pdf_page=2,
            printed_page=33,
        ),
    )

    config = FixedLengthChunkingConfig(
        max_chars=20,
        overlap_chars=0,
    )

    chunks = (
        build_fixed_length_chunks_for_pages(
            pages=pages,
            report=report,
            page_dataset_manifest=manifest,
            config=config,
        )
    )

    assert len(chunks) == 2

    assert (
        chunks[0].printed_page
        == chunks[1].printed_page
        == 33
    )

    assert (
        chunks[0].page_id
        != chunks[1].page_id
    )

    assert (
        chunks[0].chunk_id
        != chunks[1].chunk_id
    )


def test_detect_paragraph_spans() -> None:
    text = "第一段\n内部换行\n\n第二段\n\n第三段"

    spans = detect_paragraph_spans(text)

    assert len(spans) == 3

    assert [
        text[span.start_char:span.end_char]
        for span in spans
    ] == [
        "第一段\n内部换行",
        "第二段",
        "第三段",
    ]


def test_pack_complete_paragraphs() -> None:
    report = build_report()
    manifest = build_manifest()

    text = "甲甲\n\n乙乙\n\n丙丙"

    page = build_page(text=text)

    config = ParagraphChunkingConfig(
        max_chars=6,
        overlap_paragraphs=0,
        long_paragraph_overlap_chars=1,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    chunks = build_paragraph_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    assert [
        chunk.text
        for chunk in chunks
    ] == [
        "甲甲\n\n乙乙",
        "丙丙",
    ]

    assert (
        chunks[0].paragraph_start_index
        == 0
    )

    assert (
        chunks[0].paragraph_end_index
        == 1
    )


def test_paragraph_overlap_uses_whole_paragraphs(
) -> None:
    report = build_report()
    manifest = build_manifest()

    page = build_page(
        text="甲甲\n\n乙乙\n\n丙丙"
    )

    config = ParagraphChunkingConfig(
        max_chars=6,
        overlap_paragraphs=1,
        long_paragraph_overlap_chars=1,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    chunks = build_paragraph_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    assert [
        chunk.text
        for chunk in chunks
    ] == [
        "甲甲\n\n乙乙",
        "乙乙\n\n丙丙",
    ]


def test_long_paragraph_uses_fallback() -> None:
    report = build_report()
    manifest = build_manifest()

    page = build_page(
        text="abcdefghij"
    )

    config = ParagraphChunkingConfig(
        max_chars=4,
        overlap_paragraphs=0,
        long_paragraph_overlap_chars=1,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    chunks = build_paragraph_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    assert [
        chunk.text
        for chunk in chunks
    ] == [
        "abcd",
        "defg",
        "ghij",
    ]

    assert all(
        chunk.paragraph_start_index == 0
        and chunk.paragraph_end_index == 0
        for chunk in chunks
    )


def test_paragraph_chunks_restore_source_text(
) -> None:
    report = build_report()
    manifest = build_manifest()

    page = build_page(
        text="第一段\n\n第二段\n\n第三段"
    )

    config = ParagraphChunkingConfig(
        max_chars=10,
        overlap_paragraphs=1,
        long_paragraph_overlap_chars=2,
    )

    dataset_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=config,
    )

    chunks = build_paragraph_chunks(
        page=page,
        report=report,
        page_dataset_manifest=manifest,
        chunk_dataset_id=dataset_id,
        config=config,
    )

    for chunk in chunks:
        assert chunk.text == (
            page.normalized_text[
                chunk.source_start_char:
                chunk.source_end_char
            ]
        )


def test_paragraph_chunks_never_cross_pages(
) -> None:
    report = build_report()
    manifest = build_manifest()

    pages = (
        build_page(
            pdf_page=1,
            printed_page=33,
            text="第一页第一段\n\n第一页第二段",
        ),
        build_page(
            pdf_page=2,
            printed_page=33,
            text="第二页第一段\n\n第二页第二段",
        ),
    )

    chunks = build_paragraph_chunks_for_pages(
        pages=pages,
        report=report,
        page_dataset_manifest=manifest,
        config=ParagraphChunkingConfig(
            max_chars=100,
            overlap_paragraphs=1,
            long_paragraph_overlap_chars=10,
        ),
    )

    assert len(chunks) == 2
    assert chunks[0].pdf_page == 1
    assert chunks[1].pdf_page == 2

    assert (
        chunks[0].page_id
        != chunks[1].page_id
    )


def test_strategy_changes_dataset_id() -> None:
    report = build_report()
    manifest = build_manifest()

    fixed_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=FixedLengthChunkingConfig(
            max_chars=800,
            overlap_chars=120,
        ),
    )

    paragraph_id = build_chunk_dataset_id(
        report=report,
        page_dataset_manifest=manifest,
        config=ParagraphChunkingConfig(
            max_chars=800,
            overlap_paragraphs=1,
            long_paragraph_overlap_chars=120,
        ),
    )

    assert fixed_id != paragraph_id


def test_detect_section_headings() -> None:
    text = (
        "第四节 经营情况讨论与分析\n"
        "普通正文\n"
        "一、概述\n"
        "正文\n"
        "（一）总体经营情况\n"
    )

    headings = detect_section_headings(
        text,
        max_heading_chars=50,
    )

    assert [
        (heading.level, heading.title)
        for heading in headings
    ] == [
        (1, "第四节 经营情况讨论与分析"),
        (2, "一、概述"),
        (3, "（一）总体经营情况"),
    ]


def test_section_metadata_from_current_page(
) -> None:
    report = build_report()
    manifest = build_manifest()

    page = build_page(
        text=(
            "第四节 经营情况讨论与分析\n\n"
            "一、概述\n\n"
            "公司持续推动业务增长。"
        )
    )

    chunks = (
        build_section_paragraph_chunks_for_pages(
            pages=(page,),
            report=report,
            page_dataset_manifest=manifest,
            config=(
                SectionParagraphChunkingConfig(
                    max_chars=100,
                    overlap_paragraphs=0,
                    long_paragraph_overlap_chars=10,
                )
            ),
        )
    )

    final_chunk = chunks[-1]

    assert final_chunk.section_path == (
        "第四节 经营情况讨论与分析",
        "一、概述",
    )

    assert (
        final_chunk.section_source_page_id
        == page.page_id
    )

    assert final_chunk.section_inherited is False


def test_section_context_inherits_across_pages(
) -> None:
    report = build_report()
    manifest = build_manifest()

    pages = (
        build_page(
            pdf_page=1,
            printed_page=1,
            text=(
                "第四节 经营情况讨论与分析\n\n"
                "一、概述"
            ),
        ),
        build_page(
            pdf_page=2,
            printed_page=2,
            text="公司第二页的经营情况正文。",
        ),
    )

    chunks = (
        build_section_paragraph_chunks_for_pages(
            pages=pages,
            report=report,
            page_dataset_manifest=manifest,
            config=(
                SectionParagraphChunkingConfig(
                    max_chars=100,
                    overlap_paragraphs=0,
                    long_paragraph_overlap_chars=10,
                )
            ),
        )
    )

    second_page_chunk = next(
        chunk
        for chunk in chunks
        if chunk.pdf_page == 2
    )

    assert second_page_chunk.section_path == (
        "第四节 经营情况讨论与分析",
        "一、概述",
    )

    assert (
        second_page_chunk.section_source_page_id
        == pages[0].page_id
    )

    assert (
        second_page_chunk.section_inherited
        is True
    )


def test_new_top_level_heading_resets_old_path(
) -> None:
    report = build_report()
    manifest = build_manifest()

    pages = (
        build_page(
            pdf_page=1,
            printed_page=1,
            text=(
                "第四节 经营情况讨论与分析\n\n"
                "一、概述"
            ),
        ),
        build_page(
            pdf_page=2,
            printed_page=2,
            text=(
                "第五节 环境和社会责任\n\n"
                "本节正文。"
            ),
        ),
    )

    chunks = (
        build_section_paragraph_chunks_for_pages(
            pages=pages,
            report=report,
            page_dataset_manifest=manifest,
            config=(
                SectionParagraphChunkingConfig(
                    max_chars=100,
                    overlap_paragraphs=0,
                    long_paragraph_overlap_chars=10,
                )
            ),
        )
    )

    final_chunk = chunks[-1]

    assert final_chunk.section_path == (
        "第五节 环境和社会责任",
    )

    assert (
        "一、概述"
        not in final_chunk.section_path
    )


def test_section_chunk_text_is_restorable(
) -> None:
    report = build_report()
    manifest = build_manifest()

    page = build_page(
        text=(
            "第四节 经营情况讨论与分析\n\n"
            "公司经营情况正文。"
        )
    )

    chunks = (
        build_section_paragraph_chunks_for_pages(
            pages=(page,),
            report=report,
            page_dataset_manifest=manifest,
            config=(
                SectionParagraphChunkingConfig(
                    max_chars=100,
                    overlap_paragraphs=0,
                    long_paragraph_overlap_chars=10,
                )
            ),
        )
    )

    for chunk in chunks:
        assert chunk.text == (
            page.normalized_text[
                chunk.source_start_char:
                chunk.source_end_char
            ]
        )