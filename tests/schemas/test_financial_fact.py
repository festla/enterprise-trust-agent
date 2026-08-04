from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.enums import (
    EvidenceSupportType,
    PeriodType,
    RestatementStatus,
    UnitCode,
    ValidationStatus,
)
from app.schemas.financial_fact import (
    FactEvidenceLink,
    FinancialFact,
)


def build_valid_duration_fact_data() -> dict:
    """生成合法的利润表期间指标事实。"""

    now = datetime.now(timezone.utc)

    return {
        "fact_id": (
            "fact_midea_2024_revenue_consolidated"
        ),
        "company_id": "midea",
        "report_id": "midea_2024",
        "metric_id": "revenue",
        "fiscal_year": 2024,
        "statement_type": "income_statement",
        "statement_scope": "consolidated",
        "period_type": "duration",
        "period_start": date(2024, 1, 1),
        "period_end": date(2024, 12, 31),
        "as_of_date": None,
        "raw_value": "407149600",
        "raw_unit": "CNY_thousand",
        "unit_multiplier": "1000",
        "normalized_value": "407149600000",
        "normalized_unit": "CNY",
        "currency": "CNY",
        "table_name": "2024年度合并及公司利润表",
        "row_label": "营业收入",
        "column_label": "2024年度合并",
        "is_comparative_value": False,
        "restatement_status": "not_applicable",
        "primary_evidence_id": (
            "ev_midea_2024_revenue"
        ),
        "validation_status": "verified",
        "validated_by": "human",
        "validated_at": now,
        "source_version": "midea_2024_pdf_v1",
        "created_at": now,
        "updated_at": now,
    }


def test_create_valid_duration_fact() -> None:
    """合法期间财务事实应创建成功。"""

    fact = FinancialFact(
        **build_valid_duration_fact_data()
    )

    assert fact.metric_id == "revenue"
    assert fact.period_type is PeriodType.DURATION
    assert fact.raw_unit is UnitCode.CNY_THOUSAND

    assert fact.normalized_value == Decimal(
        "407149600000"
    )

    assert (
        fact.validation_status
        is ValidationStatus.VERIFIED
    )


def test_create_valid_instant_fact() -> None:
    """资产负债表时点指标应创建成功。"""

    data = build_valid_duration_fact_data()

    data.update(
        {
            "fact_id": (
                "fact_midea_2024_inventory_consolidated"
            ),
            "metric_id": "inventory",
            "statement_type": "balance_sheet",
            "period_type": "instant",
            "period_start": None,
            "period_end": None,
            "as_of_date": date(2024, 12, 31),
            "table_name": "合并资产负债表",
            "row_label": "存货",
            "primary_evidence_id": (
                "ev_midea_2024_inventory"
            ),
        }
    )

    fact = FinancialFact(**data)

    assert fact.period_type is PeriodType.INSTANT
    assert fact.as_of_date == date(2024, 12, 31)


def test_create_valid_comparative_fact() -> None:
    """后续年报中的比较列数值应被允许。"""

    data = build_valid_duration_fact_data()

    data.update(
        {
            "fact_id": (
                "fact_robam_2024_revenue_from_2025"
            ),
            "company_id": "robam",
            "report_id": "robam_2025",
            "metric_id": "revenue",
            "fiscal_year": 2024,
            "raw_value": "11212654220.22",
            "raw_unit": "CNY",
            "unit_multiplier": "1",
            "normalized_value": "11212654220.22",
            "normalized_unit": "CNY",
            "table_name": "2025年度合并利润表",
            "row_label": "营业收入",
            "column_label": "2024年度",
            "is_comparative_value": True,
            "restatement_status": "not_restated",
            "primary_evidence_id": (
                "ev_robam_2025_revenue_2024"
            ),
            "source_version": "robam_2025_pdf_v1",
        }
    )

    fact = FinancialFact(**data)

    assert fact.is_comparative_value is True
    assert (
        fact.restatement_status
        is RestatementStatus.NOT_RESTATED
    )


def test_reject_non_comparative_report_year_mismatch() -> None:
    """当前期间事实必须来自相同年度的报告。"""

    data = build_valid_duration_fact_data()
    data["report_id"] = "midea_2025"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_report_company_mismatch() -> None:
    """report_id 中的公司必须与 company_id 一致。"""

    data = build_valid_duration_fact_data()
    data["report_id"] = "gree_2024"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_duration_without_period_start() -> None:
    """期间指标不能缺少 period_start。"""

    data = build_valid_duration_fact_data()
    data["period_start"] = None

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_duration_with_as_of_date() -> None:
    """期间指标不能同时填写 as_of_date。"""

    data = build_valid_duration_fact_data()
    data["as_of_date"] = date(2024, 12, 31)

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_instant_without_as_of_date() -> None:
    """时点指标必须填写 as_of_date。"""

    data = build_valid_duration_fact_data()

    data.update(
        {
            "metric_id": "inventory",
            "statement_type": "balance_sheet",
            "period_type": "instant",
            "period_start": None,
            "period_end": None,
            "as_of_date": None,
        }
    )

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_reversed_duration_dates() -> None:
    """期间结束日期不能早于开始日期。"""

    data = build_valid_duration_fact_data()
    data["period_start"] = date(2024, 12, 31)
    data["period_end"] = date(2024, 1, 1)

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_wrong_money_multiplier() -> None:
    """千元单位必须使用 1000 倍率。"""

    data = build_valid_duration_fact_data()
    data["unit_multiplier"] = "1"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_wrong_normalized_value() -> None:
    """归一化数值必须由原值和倍率精确计算。"""

    data = build_valid_duration_fact_data()
    data["normalized_value"] = "407149600001"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_money_normalized_to_ratio() -> None:
    """金额不能归一化为 ratio。"""

    data = build_valid_duration_fact_data()
    data["normalized_unit"] = "ratio"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_verified_fact_without_validator() -> None:
    """已核验事实必须记录核验人。"""

    data = build_valid_duration_fact_data()
    data["validated_by"] = None

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_verified_fact_without_validated_at() -> None:
    """已核验事实必须记录核验时间。"""

    data = build_valid_duration_fact_data()
    data["validated_at"] = None

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_current_fact_with_restatement_status() -> None:
    """非比较列不能设置重列状态。"""

    data = build_valid_duration_fact_data()
    data["restatement_status"] = "not_restated"

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_verified_comparative_unknown_restatement() -> None:
    """已核验比较列不能保留未知重列状态。"""

    data = build_valid_duration_fact_data()

    data.update(
        {
            "report_id": "midea_2025",
            "is_comparative_value": True,
            "restatement_status": "unknown",
        }
    )

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_reject_updated_before_created() -> None:
    """更新时间不能早于创建时间。"""

    now = datetime.now(timezone.utc)

    data = build_valid_duration_fact_data()
    data["created_at"] = now
    data["updated_at"] = now - timedelta(minutes=1)

    with pytest.raises(ValidationError):
        FinancialFact(**data)


def test_create_valid_fact_evidence_link() -> None:
    """财务事实与证据关联应创建成功。"""

    link = FactEvidenceLink(
        fact_id=(
            "fact_midea_2024_revenue_consolidated"
        ),
        evidence_id="ev_midea_2024_revenue",
        support_type="primary",
        notes="合并利润表直接证据",
    )

    assert (
        link.support_type
        is EvidenceSupportType.PRIMARY
    )