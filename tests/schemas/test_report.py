from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.enums import (
    DocumentQualityGrade,
    PageMappingRuleType,
    ReportType,
    Severity,
    ValidationStatus,
)
from app.schemas.report import PageMappingSegment, Report


def build_valid_report_data() -> dict:
    """生成合法的报告测试数据。"""

    now = datetime.now(timezone.utc)

    return {
        "report_id": "midea_2025",
        "company_id": "midea",
        "fiscal_year": 2025,
        "report_type": "annual_report",
        "title": "美的集团：2025年年度报告",
        "publication_date": date(2026, 3, 31),
        "source_name": "公司官网",
        "source_uri": None,
        "quality_grade": "A",
        "citation_risk": "low",
        "active_document_id": "midea_2025_pdf_v1",
        "status": "active",
        "notes": "PDF 页码通常为印刷页码加 1",
        "created_at": now,
        "updated_at": now,
    }


def build_valid_offset_mapping_data() -> dict:
    """生成合法的固定偏移页码映射。"""

    return {
        "mapping_id": "map_midea_2025_all",
        "report_id": "midea_2025",
        "printed_page_start": 1,
        "printed_page_end": 275,
        "pdf_page_start": 2,
        "pdf_page_end": 276,
        "offset": 1,
        "rule_type": "offset",
        "notes": "第一个 PDF 页面为封面",
        "validation_status": "verified",
    }


def test_create_valid_report() -> None:
    """合法数据应成功创建 Report。"""

    report = Report(**build_valid_report_data())

    assert report.report_id == "midea_2025"
    assert report.company_id == "midea"
    assert report.report_type is ReportType.ANNUAL_REPORT
    assert report.quality_grade is DocumentQualityGrade.A
    assert report.citation_risk is Severity.LOW


def test_reject_report_id_mismatch() -> None:
    """report_id 必须与 company_id 和 fiscal_year 一致。"""

    data = build_valid_report_data()
    data["report_id"] = "midea_2024"

    with pytest.raises(ValidationError):
        Report(**data)


def test_reject_report_updated_before_created() -> None:
    """报告更新时间不能早于创建时间。"""

    now = datetime.now(timezone.utc)

    data = build_valid_report_data()
    data["created_at"] = now
    data["updated_at"] = now - timedelta(minutes=1)

    with pytest.raises(ValidationError):
        Report(**data)


def test_create_valid_offset_mapping() -> None:
    """合法固定偏移映射应通过校验。"""

    mapping = PageMappingSegment(
        **build_valid_offset_mapping_data()
    )

    assert mapping.rule_type is PageMappingRuleType.OFFSET
    assert mapping.offset == 1
    assert mapping.validation_status is ValidationStatus.VERIFIED


def test_create_valid_identity_mapping() -> None:
    """PDF 页码与印刷页码相同时应支持 identity 映射。"""

    mapping = PageMappingSegment(
        mapping_id="map_haier_2025_all",
        report_id="haier_smart_home_2025",
        printed_page_start=1,
        printed_page_end=245,
        pdf_page_start=1,
        pdf_page_end=245,
        offset=0,
        rule_type="identity",
        notes=None,
        validation_status="verified",
    )

    assert mapping.rule_type is PageMappingRuleType.IDENTITY
    assert mapping.offset == 0


def test_reject_partial_printed_page_range() -> None:
    """印刷页码开始和结束必须同时存在。"""

    data = build_valid_offset_mapping_data()
    data["printed_page_end"] = None

    with pytest.raises(ValidationError):
        PageMappingSegment(**data)


def test_reject_reversed_pdf_page_range() -> None:
    """PDF 结束页不能早于开始页。"""

    data = build_valid_offset_mapping_data()
    data["pdf_page_start"] = 276
    data["pdf_page_end"] = 2

    with pytest.raises(ValidationError):
        PageMappingSegment(**data)


def test_reject_offset_mapping_mismatch() -> None:
    """PDF 页码必须与印刷页码和 offset 一致。"""

    data = build_valid_offset_mapping_data()
    data["pdf_page_start"] = 3

    with pytest.raises(ValidationError):
        PageMappingSegment(**data)


def test_reject_verified_mapping_without_printed_pages() -> None:
    """已核验映射不能缺少印刷页码区间。"""

    data = build_valid_offset_mapping_data()
    data["printed_page_start"] = None
    data["printed_page_end"] = None

    with pytest.raises(ValidationError):
        PageMappingSegment(**data)