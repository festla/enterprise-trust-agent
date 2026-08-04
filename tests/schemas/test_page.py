import pytest
from pydantic import ValidationError

from app.schemas.page import (
    PageMappingAudit,
    PageMappingResult,
)


DOCUMENT_ID = (
    "doc_midea_group_2024_"
    + "a" * 24
)


def test_create_mapped_page_result() -> None:
    """有映射的页面应保留双页码。"""

    result = PageMappingResult(
        page_id=(
            f"{DOCUMENT_ID}_page_0158"
        ),
        document_id=DOCUMENT_ID,
        report_id="midea_group_2024",
        pdf_page=158,
        printed_page=157,
        mapping_status="mapped",
        mapping_id="midea_group_2024_main",
    )

    assert result.pdf_page == 158
    assert result.printed_page == 157


def test_create_unmapped_page_result() -> None:
    """封面可以没有印刷页码。"""

    result = PageMappingResult(
        page_id=(
            f"{DOCUMENT_ID}_page_0001"
        ),
        document_id=DOCUMENT_ID,
        report_id="midea_group_2024",
        pdf_page=1,
        printed_page=None,
        mapping_status="unmapped",
        mapping_id=None,
    )

    assert result.printed_page is None


def test_reject_wrong_page_id() -> None:
    """page_id 必须与文档和 PDF 页码一致。"""

    with pytest.raises(ValidationError):
        PageMappingResult(
            page_id=(
                f"{DOCUMENT_ID}_page_0157"
            ),
            document_id=DOCUMENT_ID,
            report_id="midea_group_2024",
            pdf_page=158,
            printed_page=157,
            mapping_status="mapped",
            mapping_id=(
                "midea_group_2024_main"
            ),
        )


def test_reject_mapped_page_without_printed_page() -> None:
    """mapped 页面必须有印刷页码。"""

    with pytest.raises(ValidationError):
        PageMappingResult(
            page_id=(
                f"{DOCUMENT_ID}_page_0158"
            ),
            document_id=DOCUMENT_ID,
            report_id="midea_group_2024",
            pdf_page=158,
            printed_page=None,
            mapping_status="mapped",
            mapping_id=(
                "midea_group_2024_main"
            ),
        )


def test_reject_unmapped_page_with_mapping_id() -> None:
    """unmapped 页面不能声称命中了映射规则。"""

    with pytest.raises(ValidationError):
        PageMappingResult(
            page_id=(
                f"{DOCUMENT_ID}_page_0001"
            ),
            document_id=DOCUMENT_ID,
            report_id="midea_group_2024",
            pdf_page=1,
            printed_page=None,
            mapping_status="unmapped",
            mapping_id=(
                "midea_group_2024_main"
            ),
        )


def test_create_valid_page_mapping_audit() -> None:
    """合法的整份映射审计应创建成功。"""

    audit = PageMappingAudit(
        document_id=DOCUMENT_ID,
        report_id="midea_group_2024",
        total_pdf_pages=295,
        mapped_page_count=294,
        unmapped_pdf_pages=(1,),
        duplicate_printed_pages={},
    )

    assert audit.mapped_page_count == 294


def test_reject_inconsistent_audit_count() -> None:
    """映射统计总数不一致时应失败。"""

    with pytest.raises(ValidationError):
        PageMappingAudit(
            document_id=DOCUMENT_ID,
            report_id="midea_group_2024",
            total_pdf_pages=295,
            mapped_page_count=293,
            unmapped_pdf_pages=(1,),
            duplicate_printed_pages={},
        )