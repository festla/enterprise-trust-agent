from dataclasses import dataclass

from app.schemas.financial_fact import FactEvidenceLink
from app.schemas.metric import MetricAlias
from app.schemas.report import PageMappingSegment
from app.services.registry import RegistryBundle


@dataclass(frozen=True, slots=True)
class Week2QualityReport:
    """Week 2 Registry 数据质量检查结果。"""

    company_count: int
    report_count: int
    page_mapping_count: int
    metric_count: int
    metric_alias_count: int
    evidence_count: int
    financial_fact_count: int
    fact_evidence_link_count: int
    verified_evidence_count: int
    verified_fact_count: int
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """没有质量问题时视为通过。"""

        return not self.issues


def build_week2_quality_report(
    *,
    bundle: RegistryBundle,
    page_mappings: list[PageMappingSegment],
    metric_aliases: list[MetricAlias],
    fact_evidence_links: list[FactEvidenceLink],
) -> Week2QualityReport:
    """检查 Week 2 最小数据集是否达到验收条件。"""

    issues: list[str] = []

    company_count = len(bundle.companies)
    report_count = len(bundle.reports)
    metric_count = len(bundle.metrics)
    evidence_count = len(bundle.evidences)
    financial_fact_count = len(
        bundle.financial_facts
    )

    verified_evidence_count = sum(
        evidence.validation_status.value
        == "verified"
        for evidence in bundle.evidences.values()
    )

    verified_fact_count = sum(
        fact.validation_status.value
        == "verified"
        for fact in bundle.financial_facts.values()
    )

    minimum_counts = {
        "Company": (company_count, 6),
        "Report": (report_count, 12),
        "PageMappingSegment": (
            len(page_mappings),
            14,
        ),
        "FinancialMetric": (
            metric_count,
            7,
        ),
        "MetricAlias": (
            len(metric_aliases),
            16,
        ),
        "SourceEvidence": (
            evidence_count,
            1,
        ),
        "FinancialFact": (
            financial_fact_count,
            1,
        ),
        "FactEvidenceLink": (
            len(fact_evidence_links),
            1,
        ),
    }

    for object_name, (
        actual_count,
        minimum_count,
    ) in minimum_counts.items():
        if actual_count < minimum_count:
            issues.append(
                f"{object_name} 数量不足："
                f"{actual_count} < {minimum_count}"
            )

    primary_link_pairs = {
        (
            link.fact_id,
            link.evidence_id,
        )
        for link in fact_evidence_links
        if link.support_type.value == "primary"
    }

    for fact in bundle.financial_facts.values():
        if not bundle.evidences.contains(
            fact.primary_evidence_id
        ):
            issues.append(
                "FinancialFact 的主要证据不存在："
                f"{fact.fact_id}"
            )

            continue

        evidence = bundle.evidences.require(
            fact.primary_evidence_id
        )

        if (
            fact.fact_id,
            fact.primary_evidence_id,
        ) not in primary_link_pairs:
            issues.append(
                "FinancialFact 缺少 primary Link："
                f"{fact.fact_id}"
            )

        if (
            fact.validation_status.value == "verified"
            and evidence.validation_status.value
            != "verified"
        ):
            issues.append(
                "verified FinancialFact 的主要证据"
                "尚未 verified："
                f"{fact.fact_id}"
            )

    for alias in metric_aliases:
        if not bundle.metrics.contains(
            alias.metric_id
        ):
            issues.append(
                "MetricAlias 引用了不存在的指标："
                f"{alias.alias_id}"
            )

    for mapping in page_mappings:
        if not bundle.reports.contains(
            mapping.report_id
        ):
            issues.append(
                "PageMappingSegment 引用了不存在的报告："
                f"{mapping.mapping_id}"
            )

    return Week2QualityReport(
        company_count=company_count,
        report_count=report_count,
        page_mapping_count=len(page_mappings),
        metric_count=metric_count,
        metric_alias_count=len(metric_aliases),
        evidence_count=evidence_count,
        financial_fact_count=financial_fact_count,
        fact_evidence_link_count=len(
            fact_evidence_links
        ),
        verified_evidence_count=(
            verified_evidence_count
        ),
        verified_fact_count=verified_fact_count,
        issues=tuple(issues),
    )