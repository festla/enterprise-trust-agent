from __future__ import annotations

import hashlib
import json
from datetime import (
    date,
    datetime,
    timezone,
)
from pathlib import Path

import pytest

from app.schemas.chunk import (
    Chunk,
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
from app.services.chunk_dataset import (
    ExistingChunkDatasetError,
    build_chunk_dataset,
)


REPORT_ID = "hisense_home_2024"

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
        company_id="hisense_home",
        fiscal_year=2024,
        report_type=(
            ReportType.ANNUAL_REPORT
        ),
        title="海信家电2024年年度报告",
        publication_date=date(
            2025,
            3,
            31,
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


def build_page(
    *,
    pdf_page: int,
    text: str,
) -> ParsedPage:
    return ParsedPage(
        page_id=(
            f"{DOCUMENT_ID}_page_"
            f"{pdf_page:04d}"
        ),
        document_id=DOCUMENT_ID,
        report_id=REPORT_ID,
        pdf_page=pdf_page,
        printed_page=33,
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


def write_page_dataset(
    root: Path,
) -> Path:
    pages = (
        build_page(
            pdf_page=1,
            text="abcdefghij",
        ),
        build_page(
            pdf_page=2,
            text="klmnopqrst",
        ),
    )

    page_lines = [
        json.dumps(
            page.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for page in pages
    ]

    pages_bytes = (
        "\n".join(page_lines) + "\n"
    ).encode("utf-8")

    manifest = PageDatasetManifest(
        dataset_id=PAGE_DATASET_ID,
        document_id=DOCUMENT_ID,
        report_id=REPORT_ID,
        source_sha256="c" * 64,
        mapping_sha256="d" * 64,
        pages_jsonl_sha256=(
            hashlib.sha256(
                pages_bytes
            ).hexdigest()
        ),
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

    page_dataset_directory = (
        root / PAGE_DATASET_ID
    )

    page_dataset_directory.mkdir(
        parents=True
    )

    (
        page_dataset_directory
        / "pages.jsonl"
    ).write_bytes(pages_bytes)

    (
        page_dataset_directory
        / "dataset_manifest.json"
    ).write_text(
        manifest.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    return page_dataset_directory


def test_build_chunk_dataset(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    result = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=FixedLengthChunkingConfig(
            max_chars=4,
            overlap_chars=1,
        ),
        created_at=NOW,
    )

    assert result.created is True
    assert result.chunks_path.is_file()
    assert result.manifest_path.is_file()
    assert (
        result.manifest.chunk_record_count
        == 6
    )


def test_repeated_build_is_idempotent(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    config = FixedLengthChunkingConfig(
        max_chars=4,
        overlap_chars=1,
    )

    first = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
        created_at=NOW,
    )

    second = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.dataset_id
        == second.manifest.dataset_id
    )


def test_changed_config_creates_new_dataset(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    first = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=FixedLengthChunkingConfig(
            max_chars=4,
            overlap_chars=1,
        ),
        created_at=NOW,
    )

    second = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=FixedLengthChunkingConfig(
            max_chars=5,
            overlap_chars=1,
        ),
        created_at=NOW,
    )

    assert (
        first.manifest.dataset_id
        != second.manifest.dataset_id
    )

    assert first.created is True
    assert second.created is True


def test_detect_tampered_chunks_file(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    config = FixedLengthChunkingConfig(
        max_chars=4,
        overlap_chars=1,
    )

    result = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
        created_at=NOW,
    )

    with result.chunks_path.open(
        "ab"
    ) as file:
        file.write(b"tampered")

    with pytest.raises(
        ExistingChunkDatasetError,
        match="哈希校验失败",
    ):
        build_chunk_dataset(
            report=build_report(),
            page_dataset_directory=(
                page_dataset_directory
            ),
            output_root=tmp_path / "chunks",
            config=config,
        )


def test_duplicate_printed_pages_remain_separate(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    result = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=FixedLengthChunkingConfig(
            max_chars=20,
            overlap_chars=0,
        ),
        created_at=NOW,
    )

    chunks = tuple(
        Chunk.model_validate_json(line)
        for line
        in result.chunks_path.read_text(
            encoding="utf-8"
        ).splitlines()
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

    assert (
        chunks[0].chunk_index
        == chunks[1].chunk_index
        == 0
    )


def test_build_paragraph_chunk_dataset(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    result = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=ParagraphChunkingConfig(
            max_chars=4,
            overlap_paragraphs=1,
            long_paragraph_overlap_chars=1,
        ),
        created_at=NOW,
    )

    assert result.created is True

    assert (
        result.manifest.strategy.value
        == "paragraph"
    )

    assert result.chunks_path.is_file()


def test_repeated_paragraph_build_is_idempotent(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    config = ParagraphChunkingConfig(
        max_chars=4,
        overlap_paragraphs=1,
        long_paragraph_overlap_chars=1,
    )

    first = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
        created_at=NOW,
    )

    second = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.dataset_id
        == second.manifest.dataset_id
    )


def test_build_section_paragraph_dataset(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    result = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=(
            SectionParagraphChunkingConfig(
                max_chars=4,
                overlap_paragraphs=1,
                long_paragraph_overlap_chars=1,
            )
        ),
        created_at=NOW,
    )

    assert result.created is True

    assert (
        result.manifest.strategy.value
        == "section_paragraph"
    )

    assert result.chunks_path.is_file()


def test_repeated_section_build_is_idempotent(
    tmp_path: Path,
) -> None:
    page_dataset_directory = (
        write_page_dataset(tmp_path)
    )

    config = (
        SectionParagraphChunkingConfig(
            max_chars=4,
            overlap_paragraphs=1,
            long_paragraph_overlap_chars=1,
        )
    )

    first = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
        created_at=NOW,
    )

    second = build_chunk_dataset(
        report=build_report(),
        page_dataset_directory=(
            page_dataset_directory
        ),
        output_root=tmp_path / "chunks",
        config=config,
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.dataset_id
        == second.manifest.dataset_id
    )