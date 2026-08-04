from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.document import DocumentManifest


def build_valid_manifest_data() -> dict:
    """生成合法文档 Manifest 数据。"""

    sha256 = "a" * 64

    return {
        "document_id": (
            "doc_midea_group_2024_"
            + sha256[:24]
        ),
        "report_id": "midea_group_2024",
        "source_path": (
            "data/raw_reports/midea/2024/"
            "midea_2024_annual_report.pdf"
        ),
        "source_filename": (
            "midea_2024_annual_report.pdf"
        ),
        "sha256": sha256,
        "file_size_bytes": 6737887,
        "pdf_page_count": 295,
        "expected_pdf_page_count": 295,
        "page_count_status": "matched",
        "parser_name": "pymupdf",
        "parser_version": "1.0.0",
        "validation_status": "valid",
        "created_at": datetime.now(timezone.utc),
    }


def test_create_valid_document_manifest() -> None:
    """合法 Manifest 应创建成功。"""

    manifest = DocumentManifest(
        **build_valid_manifest_data()
    )

    assert manifest.report_id == (
        "midea_group_2024"
    )
    assert manifest.pdf_page_count == 295
    assert manifest.validation_status.value == "valid"


def test_reject_wrong_document_id() -> None:
    """document_id 必须由报告和哈希生成。"""

    data = build_valid_manifest_data()
    data["document_id"] = (
        "doc_midea_group_2024_wrong"
    )

    with pytest.raises(ValidationError):
        DocumentManifest(**data)


def test_reject_absolute_source_path() -> None:
    """Manifest 不能保存本机绝对路径。"""

    data = build_valid_manifest_data()
    data["source_path"] = (
        "D:/private/report.pdf"
    )
    data["source_filename"] = "report.pdf"

    with pytest.raises(ValidationError):
        DocumentManifest(**data)


def test_reject_page_count_status_conflict() -> None:
    """页数相同不能标记为 mismatched。"""

    data = build_valid_manifest_data()
    data["page_count_status"] = "mismatched"
    data["validation_status"] = "blocked"

    with pytest.raises(ValidationError):
        DocumentManifest(**data)


def test_reject_valid_document_with_mismatched_pages() -> None:
    """页数不一致的文档不能标记为 valid。"""

    data = build_valid_manifest_data()
    data["pdf_page_count"] = 294
    data["page_count_status"] = "mismatched"

    with pytest.raises(ValidationError):
        DocumentManifest(**data)


def test_reject_datetime_without_timezone() -> None:
    """Manifest 创建时间必须包含时区。"""

    data = build_valid_manifest_data()
    data["created_at"] = datetime.now()

    with pytest.raises(ValidationError):
        DocumentManifest(**data)