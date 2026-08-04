from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.page_dataset import (
    PageDatasetManifest,
)


def build_valid_data() -> dict:
    """生成合法页面数据集 Manifest。"""

    return {
        "dataset_id": (
            "page_dataset_midea_group_2024_"
            + "a" * 24
        ),
        "document_id": (
            "doc_midea_group_2024_"
            + "b" * 24
        ),
        "report_id": "midea_group_2024",
        "source_sha256": "b" * 64,
        "mapping_sha256": "c" * 64,
        "pages_jsonl_sha256": "d" * 64,
        "parser_name": "pymupdf",
        "parser_version": "1.28.0",
        "normalizer_version": (
            "page_text_normalizer_v1"
        ),
        "classifier_version": (
            "page_content_classifier_v1"
        ),
        "total_pdf_pages": 2,
        "page_record_count": 2,
        "mapped_page_count": 1,
        "unmapped_page_count": 1,
        "raw_char_count_total": 20,
        "normalized_char_count_total": 18,
        "content_type_counts": {
            "text": 1,
            "empty": 1,
            "scanned": 0,
            "mixed": 0,
            "unknown": 0,
        },
        "parse_status_counts": {
            "success": 2,
            "parse_error": 0,
        },
        "quality_gate_passed": True,
        "quality_gate_errors": (),
        "quality_warnings": (
            "存在 1 个空白页面",
        ),
        "created_at": datetime.now(
            timezone.utc
        ),
    }


def test_create_valid_page_dataset_manifest() -> None:
    """合法数据集 Manifest 应创建成功。"""

    manifest = PageDatasetManifest(
        **build_valid_data()
    )

    assert manifest.page_record_count == 2
    assert manifest.quality_gate_passed is True


def test_reject_page_count_mismatch() -> None:
    """页面记录数必须等于 PDF 总页数。"""

    data = build_valid_data()
    data["page_record_count"] = 1

    with pytest.raises(ValidationError):
        PageDatasetManifest(**data)


def test_reject_inconsistent_quality_status() -> None:
    """质量状态必须与错误列表一致。"""

    data = build_valid_data()
    data["quality_gate_errors"] = (
        "存在页面解析失败",
    )

    with pytest.raises(ValidationError):
        PageDatasetManifest(**data)


def test_reject_unknown_and_parse_error_mismatch() -> None:
    """unknown 页数必须对应解析错误页数。"""

    data = build_valid_data()

    data["content_type_counts"] = {
        "text": 0,
        "empty": 1,
        "scanned": 0,
        "mixed": 0,
        "unknown": 1,
    }

    with pytest.raises(ValidationError):
        PageDatasetManifest(**data)