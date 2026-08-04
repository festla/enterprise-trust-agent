from datetime import date, datetime, timezone

import pytest

from app.schemas import (
    Company,
    FinancialFact,
    FinancialMetric,
    Report,
    SourceEvidence,
)
from app.services.registry import (
    CompanyRegistry,
    DuplicateRegistryKeyError,
    RegistryBundle,
    RegistryIntegrityError,
    RegistryItemNotFoundError,
)


def build_company() -> Company:
    now = datetime.now(timezone.utc)

    return Company(
        company_id="midea",
        legal_name_cn="美的集团股份有限公司",
        short_name_cn="美的集团",
        stock_code="000333",
        exchange="SZSE",
        industry="家电制造业",
        status="active",
        created_at=now,
        updated_at=now,
    )


def build_report() -> Report:
    now = datetime.now(timezone.utc)

    return Report(
        report_id="midea_2024",
        company_id="midea",
        fiscal_year=2024,
        report_type="annual_report",
        title="美的集团：2024年年度报告",
        publication_date=date(2025, 3, 29),
        source_name="公司官网",
        source_uri=None,
        quality_grade="A",
        citation_risk="low",
        active_document_id="midea_2024_pdf_v1",
        status="active",
        notes="PDF 页码比印刷页码大 1",
        created_at=now,
        updated_at=now,
    )


def build_metric() -> FinancialMetric:
    now = datetime.now(timezone.utc)

    return FinancialMetric(
        metric_id="revenue",
        display_name_cn="营业收入",
        display_name_en="Revenue",
        description=(
            "企业日常经营活动形成的收入，"
            "不等同于营业总收入"
        ),
        metric_origin="reported",
        statement_type="income_statement",
        period_type="duration",
        default_unit="CNY",
        allowed_scopes=[
            "consolidated",
            "parent_company",
        ],
        value_type="decimal",
        is_core_metric=True,
        confusable_metric_ids=[
            "total_operating_revenue"
        ],
        formula_id=None,
        status="active",
        created_at=now,
        updated_at=now,
    )


def build_evidence() -> SourceEvidence:
    return SourceEvidence(
        evidence_id="ev_midea_2024_revenue",
        report_id="midea_2024",
        document_id="midea_2024_pdf_v1",
        page_id="midea_2024_pdf_v1_p0158",
        chunk_id=None,
        evidence_type="financial_statement_cell",
        attribution_type="report_disclosure",
        statement_type="income_statement",
        statement_scope="consolidated",
        section_title="财务报告",
        subsection_title="合并及公司利润表",
        table_name="2024年度合并及公司利润表",
        row_label="营业收入",
        column_label="2024年度合并",
        printed_page=157,
        pdf_page=158,
        evidence_text=(
            "表格单位为人民币千元，营业收入在"
            "2024年度合并列的原始值为407,149,600。"
        ),
        cell_value="407,149,600",
        source_hash="a" * 64,
        validation_status="verified",
        validated_by="human",
        created_at=datetime.now(timezone.utc),
    )


def build_fact() -> FinancialFact:
    now = datetime.now(timezone.utc)

    return FinancialFact(
        fact_id="fact_midea_2024_revenue_consolidated",
        company_id="midea",
        report_id="midea_2024",
        metric_id="revenue",
        fiscal_year=2024,
        statement_type="income_statement",
        statement_scope="consolidated",
        period_type="duration",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        as_of_date=None,
        raw_value="407149600",
        raw_unit="CNY_thousand",
        unit_multiplier="1000",
        normalized_value="407149600000",
        normalized_unit="CNY",
        currency="CNY",
        table_name="2024年度合并及公司利润表",
        row_label="营业收入",
        column_label="2024年度合并",
        is_comparative_value=False,
        restatement_status="not_applicable",
        primary_evidence_id="ev_midea_2024_revenue",
        validation_status="verified",
        validated_by="human",
        validated_at=now,
        source_version="midea_2024_pdf_v1",
        created_at=now,
        updated_at=now,
    )


def build_valid_bundle() -> RegistryBundle:
    bundle = RegistryBundle()

    bundle.companies.add(build_company())
    bundle.reports.add(build_report())
    bundle.metrics.add(build_metric())
    bundle.evidences.add(build_evidence())
    bundle.financial_facts.add(build_fact())

    return bundle


def test_add_and_get_company() -> None:
    """Registry 应能够添加和查询 Company。"""

    registry = CompanyRegistry()
    company = build_company()

    registry.add(company)

    assert registry.get("midea") is company
    assert registry.contains("midea") is True
    assert len(registry) == 1


def test_reject_duplicate_company_id() -> None:
    """重复 company_id 应该被拒绝。"""

    registry = CompanyRegistry()
    company = build_company()

    registry.add(company)

    with pytest.raises(DuplicateRegistryKeyError):
        registry.add(company)


def test_require_missing_company() -> None:
    """require 查询不存在对象时应抛出明确异常。"""

    registry = CompanyRegistry()

    with pytest.raises(RegistryItemNotFoundError):
        registry.require("unknow_company")


def test_reject_report_with_missing_company() -> None:
    """Report 引用不存在 Company 时应失败。"""

    bundle = RegistryBundle()
    bundle.reports.add(build_report())

    with pytest.raises(RegistryIntegrityError) as exc_info:
        bundle.validate_relationships()

    assert "不存在的 Company" in str(exc_info.value)


def test_reject_fact_with_missing_metric() -> None:
    """Fact 引用不存在 Metric 时应失败。"""

    bundle = build_valid_bundle()
    bundle.metrics = type(bundle.metrics)()

    with pytest.raises(RegistryIntegrityError) as exc_info:
        bundle.validate_relationships()

    assert "不存在的 FinancialMetric" in str(
        exc_info.value
    )


def test_reject_fact_with_missing_evidence() -> None:
    """Fact 引用不存在 Evidence 时应失败。"""

    bundle = build_valid_bundle()
    bundle.evidences = type(bundle.evidences)()

    with pytest.raises(RegistryIntegrityError) as exc_info:
        bundle.validate_relationships()

    assert "不存在的 SourceEvidence" in str(
        exc_info.value
    )


def test_reject_evidence_report_mismatch() -> None:
    """Evidence 与 Fact 的 report_id 不一致时应失败。"""

    bundle = build_valid_bundle()

    original = bundle.evidences.require(
        "ev_midea_2024_revenue"
    )

    corrupted = original.model_copy(
        update={"report_id": "midea_2025"}
    )

    bundle.evidences = type(bundle.evidences)()
    bundle.evidences.add(corrupted)

    with pytest.raises(RegistryIntegrityError) as exc_info:
        bundle.validate_relationships()

    assert "report_id 不一致" in str(exc_info.value)


def test_find_financial_fact() -> None:
    """应能够按业务字段查询财务事实。"""

    bundle = build_valid_bundle()

    results = bundle.financial_facts.find(
        company_id="midea",
        fiscal_year=2024,
        metric_id="revenue",
        statement_scope="consolidated",
    )

    assert len(results) == 1

    fact = results[0]

    assert (
        fact.fact_id
        == "fact_midea_2024_revenue_consolidated"
    )

    assert str(fact.normalized_value) == "407149600000"


def test_find_financial_fact_returns_empty_list() -> None:
    """没有匹配事实时应返回空列表。"""

    bundle = build_valid_bundle()

    results = bundle.financial_facts.find(
        company_id="gree",
        metric_id="revenue",
    )

    assert results == []