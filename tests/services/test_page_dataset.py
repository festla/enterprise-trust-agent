from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pytest

from app.schemas.enums import PageContentType
from app.schemas.document import DocumentManifest
from app.schemas.report import PageMappingSegment
from app.services.page_dataset import (
    ExistingPageDatasetError,
    build_page_dataset,
)
from app.services.document_ingestion import (
    calculate_file_sha256,
)


def create_test_pdf(path: Path) -> None:
    """创建一页文本、一页空白的 PDF。"""

    document = pymupdf.open()

    try:
        text_page = document.new_page()

        text_page.insert_text(
            (72, 72),
            "Annual report page",
        )

        document.new_page()

        document.save(path)

    finally:
        document.close()


def build_manifest(
    *,
    project_root: Path,
    pdf_path: Path,
) -> DocumentManifest:
    """创建测试用文档 Manifest。"""

    sha256 = calculate_file_sha256(
        pdf_path
    )

    return DocumentManifest(
        document_id=(
            "doc_midea_group_2024_"
            f"{sha256[:24]}"
        ),
        report_id="midea_group_2024",
        source_path=(
            pdf_path.relative_to(
                project_root
            ).as_posix()
        ),
        source_filename=pdf_path.name,
        sha256=sha256,
        file_size_bytes=(
            pdf_path.stat().st_size
        ),
        pdf_page_count=2,
        expected_pdf_page_count=2,
        page_count_status="matched",
        parser_name="pymupdf",
        parser_version="1.28.0",
        validation_status="valid",
        created_at=datetime.now(timezone.utc),
    )


def build_mapping(
    mapping_id: str = "mapping_main",
) -> PageMappingSegment:
    """创建第二页对应印刷第 1 页的规则。"""

    return PageMappingSegment(
        mapping_id=mapping_id,
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=1,
        pdf_page_start=2,
        pdf_page_end=2,
        offset=1,
        rule_type="offset",
        notes="第一页作为封面",
        validation_status="verified",
    )


def test_build_complete_page_dataset(
    tmp_path: Path,
) -> None:
    """应生成页面 JSONL 和数据集 Manifest。"""

    pdf_path = tmp_path / "report.pdf"
    create_test_pdf(pdf_path)

    result = build_page_dataset(
        source_manifest=build_manifest(
            project_root=tmp_path,
            pdf_path=pdf_path,
        ),
        page_mappings=[build_mapping()],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    assert result.created is True
    assert result.pages_path.exists()
    assert result.manifest_path.exists()

    assert result.manifest.page_record_count == 2
    assert result.manifest.mapped_page_count == 1
    assert result.manifest.unmapped_page_count == 1

    result.manifest.content_type_counts[
        PageContentType.TEXT
    ] == 1

    result.manifest.content_type_counts[
        PageContentType.EMPTY
    ] == 1

    assert result.manifest.quality_gate_passed is True


def test_repeated_build_is_idempotent(
    tmp_path: Path,
) -> None:
    """相同输入重复构建不得覆盖数据集。"""

    pdf_path = tmp_path / "report.pdf"
    create_test_pdf(pdf_path)

    manifest = build_manifest(
        project_root=tmp_path,
        pdf_path=pdf_path,
    )

    first = build_page_dataset(
        source_manifest=manifest,
        page_mappings=[build_mapping()],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    first_pages_content = (
        first.pages_path.read_bytes()
    )

    second = build_page_dataset(
        source_manifest=manifest,
        page_mappings=[build_mapping()],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    assert first.created is True
    assert second.created is False

    assert (
        first.manifest.dataset_id
        == second.manifest.dataset_id
    )

    assert (
        second.pages_path.read_bytes()
        == first_pages_content
    )


def test_mapping_change_creates_new_dataset(
    tmp_path: Path,
) -> None:
    """映射配置变化后应创建新的页面数据集。"""

    pdf_path = tmp_path / "report.pdf"
    create_test_pdf(pdf_path)

    manifest = build_manifest(
        project_root=tmp_path,
        pdf_path=pdf_path,
    )

    first = build_page_dataset(
        source_manifest=manifest,
        page_mappings=[
            build_mapping("mapping_v1")
        ],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    second = build_page_dataset(
        source_manifest=manifest,
        page_mappings=[
            build_mapping("mapping_v2")
        ],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    assert (
        first.manifest.dataset_id
        != second.manifest.dataset_id
    )

    assert first.dataset_directory.exists()
    assert second.dataset_directory.exists()


def test_reject_tampered_existing_pages(
    tmp_path: Path,
) -> None:
    """已有 JSONL 被修改后必须显式失败。"""

    pdf_path = tmp_path / "report.pdf"
    create_test_pdf(pdf_path)

    manifest = build_manifest(
        project_root=tmp_path,
        pdf_path=pdf_path,
    )

    first = build_page_dataset(
        source_manifest=manifest,
        page_mappings=[build_mapping()],
        project_root=tmp_path,
        output_root=tmp_path / "pages",
    )

    with first.pages_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(
        ExistingPageDatasetError
    ):
        build_page_dataset(
            source_manifest=manifest,
            page_mappings=[build_mapping()],
            project_root=tmp_path,
            output_root=tmp_path / "pages",
        )