from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.enums import (
    AttributionType,
    EvidenceType,
    ValidationStatus,
)
from app.schemas.evidence import SourceEvidence


def build_valid_financial_evidence_data() -> dict:
    """生成合法的财务报表单元格证据。"""

    return {
        "evidence_id": "ev_midea_2024_revenue",
        "report_id": "midea_2024",
        "document_id": "midea_2024_pdf_v1",
        "page_id": "midea_2024_pdf_v1_p0158",
        "chunk_id": None,
        "evidence_type": "financial_statement_cell",
        "attribution_type": "report_disclosure",
        "statement_type": "income_statement",
        "statement_scope": "consolidated",
        "section_title": "财务报告",
        "subsection_title": "合并及公司利润表",
        "table_name": "2024年度合并及公司利润表",
        "row_label": "营业收入",
        "column_label": "2024年度合并",
        "printed_page": 157,
        "pdf_page": 158,
        "evidence_text": (
            "表格单位为人民币千元，营业收入在"
            "2024年度合并列的原始值为407,149,600。"
        ),
        "cell_value": "407,149,600",
        "source_hash": "a" * 64,
        "validation_status": "verified",
        "validated_by": "human",
        "created_at": datetime.now(timezone.utc),
    }


def test_create_valid_financial_evidence() -> None:
    """合法财务报表证据应创建成功。"""

    evidence = SourceEvidence(
        **build_valid_financial_evidence_data()
    )

    assert (
        evidence.evidence_type
        is EvidenceType.FINANCIAL_STATEMENT_CELL
    )
    assert (
        evidence.attribution_type
        is AttributionType.REPORT_DISCLOSURE
    )
    assert evidence.validation_status is ValidationStatus.VERIFIED
    assert evidence.pdf_page == 158


def test_create_valid_management_statement() -> None:
    """合法管理层表述证据应创建成功。"""

    evidence = SourceEvidence(
        evidence_id="ev_midea_2025_management_001",
        report_id="midea_2025",
        document_id="midea_2025_pdf_v1",
        page_id="midea_2025_pdf_v1_p0020",
        chunk_id="midea_2025_pdf_v1_c000020",
        evidence_type="management_statement",
        attribution_type="management_statement",
        statement_type="management_discussion",
        statement_scope=None,
        section_title="第三节 管理层讨论与分析",
        subsection_title="主营业务分析",
        table_name=None,
        row_label=None,
        column_label=None,
        printed_page=19,
        pdf_page=20,
        evidence_text=(
            "管理层表示以旧换新政策带动国内需求恢复。"
        ),
        cell_value=None,
        source_hash="b" * 64,
        validation_status="verified",
        validated_by="human",
        created_at=datetime.now(timezone.utc),
    )

    assert (
        evidence.evidence_type
        is EvidenceType.MANAGEMENT_STATEMENT
    )
    assert evidence.printed_page == 19


def test_create_valid_printed_page_range() -> None:
    """跨页证据应支持页码范围字符串。"""

    data = build_valid_financial_evidence_data()
    data["printed_page"] = "48-49"

    evidence = SourceEvidence(**data)

    assert evidence.printed_page == "48-49"


def test_reject_system_calculation_as_source_evidence() -> None:
    """系统计算不能伪装成原始来源证据。"""

    data = build_valid_financial_evidence_data()
    data["attribution_type"] = "system_calculation"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_management_evidence_wrong_attribution() -> None:
    """管理层表述必须使用对应归属类型。"""

    data = build_valid_financial_evidence_data()
    data["evidence_type"] = "management_statement"
    data["attribution_type"] = "report_disclosure"
    data["section_title"] = "管理层讨论与分析"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


@pytest.mark.parametrize(
    "missing_field",
    [
        "statement_type",
        "statement_scope",
        "table_name",
        "row_label",
        "column_label",
        "cell_value",
    ],
)
def test_reject_incomplete_financial_table_evidence(
    missing_field: str,
) -> None:
    """财务表格证据不能缺少定位字段。"""

    data = build_valid_financial_evidence_data()
    data[missing_field] = None

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_narrative_evidence_without_section() -> None:
    """叙述性证据必须填写章节。"""

    data = build_valid_financial_evidence_data()
    data.update(
        {
            "evidence_type": "risk_disclosure",
            "attribution_type": "report_disclosure",
            "statement_type": "important_events",
            "statement_scope": None,
            "section_title": None,
            "table_name": None,
            "row_label": None,
            "column_label": None,
            "cell_value": None,
        }
    )

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_unknown_statement_scope() -> None:
    """证据不能显式使用 unknown 口径。"""

    data = build_valid_financial_evidence_data()
    data["statement_scope"] = "unknown"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_reversed_printed_page_range() -> None:
    """印刷页码范围不能前后颠倒。"""

    data = build_valid_financial_evidence_data()
    data["printed_page"] = "49-48"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_invalid_printed_page_format() -> None:
    """印刷页码字符串必须符合统一格式。"""

    data = build_valid_financial_evidence_data()
    data["printed_page"] = "第157页"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_invalid_source_hash() -> None:
    """source_hash 必须是 64 位小写十六进制字符串。"""

    data = build_valid_financial_evidence_data()
    data["source_hash"] = "not-a-valid-hash"

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_naive_created_at() -> None:
    """created_at 必须包含时区。"""

    data = build_valid_financial_evidence_data()
    data["created_at"] = datetime.now()

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_verified_evidence_without_validator() -> None:
    """已核验证据必须记录核验人。"""

    data = build_valid_financial_evidence_data()
    data["validated_by"] = None

    with pytest.raises(ValidationError):
        SourceEvidence(**data)


def test_reject_verified_evidence_without_printed_page() -> None:
    """已核验证据必须保留印刷页码。"""

    data = build_valid_financial_evidence_data()
    data["printed_page"] = None

    with pytest.raises(ValidationError):
        SourceEvidence(**data)