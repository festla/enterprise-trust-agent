from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.page import ParsedPage


DOCUMENT_ID = (
    "doc_midea_group_2024_"
    + "a" * 24
)


def build_valid_text_page_data() -> dict:
    """生成合法文本页面数据。"""

    raw_text = "营业收入 407,149,600\n"
    normalized_text = (
        "营业收入 407,149,600"
    )

    return {
        "page_id": (
            f"{DOCUMENT_ID}_page_0158"
        ),
        "document_id": DOCUMENT_ID,
        "report_id": "midea_group_2024",
        "pdf_page": 158,
        "printed_page": 157,
        "mapping_status": "mapped",
        "mapping_id": (
            "midea_group_2024_main"
        ),
        "raw_text": raw_text,
        "normalized_text": (
            normalized_text
        ),
        "raw_char_count": len(raw_text),
        "normalized_char_count": len(
            normalized_text
        ),
        "text_block_count": 2,
        "image_block_count": 0,
        "embedded_image_count": 0,
        "max_image_area_ratio": 0,
        "content_type": "text",
        "parse_status": "success",
        "parse_error": None,
        "parser_name": "pymupdf",
        "parser_version": "1.28.0",
        "parsed_at": datetime.now(
            timezone.utc
        ),
    }


def test_create_valid_text_page() -> None:
    """合法文本页面应创建成功。"""

    page = ParsedPage(
        **build_valid_text_page_data()
    )

    assert page.content_type.value == "text"
    assert page.parse_status.value == "success"


def test_reject_wrong_raw_character_count() -> None:
    """原始字符数必须与文本长度一致。"""

    data = build_valid_text_page_data()
    data["raw_char_count"] += 1

    with pytest.raises(ValidationError):
        ParsedPage(**data)


def test_reject_success_page_with_error() -> None:
    """成功页面不能保留 parse_error。"""

    data = build_valid_text_page_data()
    data["parse_error"] = "unexpected error"

    with pytest.raises(ValidationError):
        ParsedPage(**data)


def test_create_valid_parse_error_page() -> None:
    """单页失败应保存结构化错误记录。"""

    data = build_valid_text_page_data()

    data.update(
        {
            "raw_text": "",
            "normalized_text": "",
            "raw_char_count": 0,
            "normalized_char_count": 0,
            "text_block_count": 0,
            "image_block_count": 0,
            "embedded_image_count": 0,
            "max_image_area_ratio": 0,
            "content_type": "unknown",
            "parse_status": "parse_error",
            "parse_error": (
                "RuntimeError: page failed"
            ),
        }
    )

    page = ParsedPage(**data)

    assert page.parse_status.value == (
        "parse_error"
    )


def test_reject_scanned_page_with_text() -> None:
    """扫描页面不能同时保存规范化文本。"""

    data = build_valid_text_page_data()

    data.update(
        {
            "content_type": "scanned",
            "image_block_count": 1,
            "embedded_image_count": 1,
            "max_image_area_ratio": 1,
        }
    )

    with pytest.raises(ValidationError):
        ParsedPage(**data)


def test_reject_parsed_at_without_timezone() -> None:
    """页面解析时间必须包含时区。"""

    data = build_valid_text_page_data()
    data["parsed_at"] = datetime.now()

    with pytest.raises(ValidationError):
        ParsedPage(**data)