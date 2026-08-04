import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pytest

from app.schemas.document import DocumentManifest
from app.schemas.report import PageMappingSegment
from app.services import page_parser
from app.services.page_parser import (
    PageSourceHashMismatchError,
    PageSourcePageCountMismatchError,
    normalize_page_text,
    parse_pdf_pages,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    "CAIAAACQd1PeAAAADElEQVR4nGP4//8/"
    "AAX+Av4N70a4AAAAAElFTkSuQmCC"
)

def create_mixed_test_pdf(
    path: Path,
) -> None:
    """创建文本、空白、扫描和混合四类页面。"""

    document = pymupdf.open()

    try:
        text_page = document.new_page()

        text_page.insert_text(
            (72, 72),
            "Text only page",
        )

        document.new_page()

        scanned_page = document.new_page()

        scanned_page.insert_image(
            scanned_page.rect,
            stream=_ONE_PIXEL_PNG,
        )

        mixed_page = document.new_page()

        mixed_page.insert_image(
            mixed_page.rect,
            stream=_ONE_PIXEL_PNG,
        )

        mixed_page.insert_text(
            (72, 72),
            "Text over image",
        )

        document.save(path)

    finally:
        document.close()


def calculate_test_sha256(
    path: Path,
) -> str:
    """计算小型测试文件的 SHA-256。"""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def build_manifest(
    *,
    project_root: Path,
    pdf_path: Path,
    page_count: int = 4,
) -> DocumentManifest:
    """创建与测试 PDF 匹配的 Manifest。"""

    sha256 = calculate_test_sha256(
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
        pdf_page_count=page_count,
        expected_pdf_page_count=page_count,
        page_count_status="matched",
        parser_name="pymupdf",
        parser_version="1.28.0",
        validation_status="valid",
        created_at=datetime.now(timezone.utc),
    )


def build_mapping() -> PageMappingSegment:
    """创建测试用固定偏移映射。"""

    return PageMappingSegment(
        mapping_id="midea_group_2024_main",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=3,
        pdf_page_start=2,
        pdf_page_end=4,
        offset=1,
        rule_type="offset",
        notes="PDF 第 1 页作为封面",
        validation_status="verified",
    )


def test_normalize_page_text() -> None:
    """文本规范化不得改变正文语义。"""

    raw_text = (
        "\x00第一行   \r\n"
        "\r\n"
        "\r\n"
        "\r\n"
        "第二行\t\r"
    )

    assert normalize_page_text(
        raw_text
    ) == "第一行\n\n第二行"


def test_parse_four_page_content_types(
    tmp_path: Path,
) -> None:
    """应识别文本、空白、扫描和混合页面。"""

    pdf_path = tmp_path / "test.pdf"

    create_mixed_test_pdf(
        pdf_path
    )

    pages = parse_pdf_pages(
        manifest=build_manifest(
            project_root=tmp_path,
            pdf_path=pdf_path,
        ),
        page_mappings=[build_mapping()],
        project_root=tmp_path,
    )

    assert len(pages) == 4

    assert pages[0].content_type.value == "text"
    assert pages[1].content_type.value == "empty"
    assert pages[2].content_type.value == "scanned"
    assert pages[3].content_type.value == "mixed"

    assert pages[0].mapping_status.value == (
        "unmapped"
    )

    assert pages[1].printed_page == 1
    assert pages[2].printed_page == 2
    assert pages[3].printed_page == 3


def test_page_text_is_not_merged_across_pages(
    tmp_path: Path,
) -> None:
    """每页文本必须保持独立。"""

    pdf_path = tmp_path / "test.pdf"

    create_mixed_test_pdf(pdf_path)

    pages = parse_pdf_pages(
        manifest=build_manifest(
            project_root=tmp_path,
            pdf_path=pdf_path,
        ),
        page_mappings=[build_mapping()],
        project_root=tmp_path,
        pdf_pages=[1, 4],
    )

    assert "Text only page" in (
        pages[0].normalized_text
    )

    assert "Text over image" not in (
        pages[0].normalized_text
    )

    assert "Text over image" in (
        pages[1].normalized_text
    )

    assert "Text only page" not in (
        pages[1].normalized_text
    )


def test_single_page_failure_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一个页面失败不应终止其他页面解析。"""

    pdf_path = tmp_path / "test.pdf"

    create_mixed_test_pdf(pdf_path)

    original_extract = (
        page_parser._extract_page_signals
    )

    def extract_with_failure(
        page: pymupdf.Page,
    ):
        if page.number == 1:
            raise RuntimeError(
                "simulated page failure"
            )

        return original_extract(page)

    monkeypatch.setattr(
        page_parser,
        "_extract_page_signals",
        extract_with_failure,
    )

    pages = parse_pdf_pages(
        manifest=build_manifest(
            project_root=tmp_path,
            pdf_path=pdf_path,
        ),
        page_mappings=[build_mapping()],
        project_root=tmp_path,
    )

    assert pages[0].parse_status.value == (
        "success"
    )

    assert pages[1].parse_status.value == (
        "parse_error"
    )

    assert "simulated page failure" in (
        pages[1].parse_error
    )

    assert pages[2].parse_status.value == (
        "success"
    )

    assert pages[3].parse_status.value == (
        "success"
    )


def test_reject_pdf_changed_after_registration(
    tmp_path: Path,
) -> None:
    """文件内容变化后不得继续使用旧 Manifest。"""

    pdf_path = tmp_path / "test.pdf"

    create_mixed_test_pdf(pdf_path)

    manifest = build_manifest(
        project_root=tmp_path,
        pdf_path=pdf_path,
    )

    with pdf_path.open("ab") as file:
        file.write(b"changed")

    with pytest.raises(
        PageSourceHashMismatchError
    ):
        parse_pdf_pages(
            manifest=manifest,
            page_mappings=[build_mapping()],
            project_root=tmp_path,
        )


def test_reject_manifest_page_count_mismatch(
    tmp_path: Path,
) -> None:
    """Manifest 页数与实际 PDF 不一致时应失败。"""

    pdf_path = tmp_path / "test.pdf"

    create_mixed_test_pdf(pdf_path)

    manifest = build_manifest(
        project_root=tmp_path,
        pdf_path=pdf_path,
        page_count=5,
    )

    mapping = PageMappingSegment(
        mapping_id="midea_group_2024_main",
        report_id="midea_group_2024",
        printed_page_start=1,
        printed_page_end=4,
        pdf_page_start=2,
        pdf_page_end=5,
        offset=1,
        rule_type="offset",
        notes="测试错误页数",
        validation_status="verified",
    )

    with pytest.raises(
        PageSourcePageCountMismatchError
    ):
        parse_pdf_pages(
            manifest=manifest,
            page_mappings=[mapping],
            project_root=tmp_path,
        )