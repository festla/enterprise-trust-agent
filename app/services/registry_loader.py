from pathlib import Path
from typing import Any

import yaml

from app.schemas.enums import EvidenceSupportType
from app.schemas.metric import FinancialMetric, MetricAlias
from app.schemas.company import Company
from app.schemas.evidence import SourceEvidence
from app.schemas.financial_fact import (
    FactEvidenceLink,
    FinancialFact,
)
from app.services.registry import (
    CompanyRegistry,
    MetricRegistry,
    ReportRegistry,
    RegistryBundle,
    RegistryIntegrityError,
    EvidenceRegistry,
    FinancialFactRegistry,
)

from app.schemas.report import PageMappingSegment, Report



REGISTRY_SCHEMA_VERSION = 1


class RegistryLoaderError(ValueError):
    """Registry YAML 文件读取或结构校验失败。"""


def load_registry_yaml(path: Path) -> dict[str, Any]:
    """读取并检查一份 Registry YAML 文件。"""

    try:
        raw_data = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise RegistryLoaderError(
            f"Registry YAML 文件不存在：{path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RegistryLoaderError(
            f"Registry YAML 解析失败：{path}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise RegistryLoaderError(
            "Registry YAML 顶层必须是映射对象"
        )

    schema_version = raw_data.get("schema_version")

    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise RegistryLoaderError(
            "不支持的 Registry YAML schema_version："
            f"{schema_version!r}"
        )

    return raw_data


def load_companies(path: Path) -> CompanyRegistry:
    """从 YAML 文件加载 CompanyRegistry。"""

    raw_data = load_registry_yaml(path)
    raw_companies = raw_data.get("companies")

    if not isinstance(raw_companies, list):
        raise RegistryLoaderError(
            "companies 字段必须是列表"
        )

    if not raw_companies:
        raise RegistryLoaderError(
            "companies 字段不能为空"
        )

    companies = [
        Company.model_validate(item)
        for item in raw_companies
    ]

    registry = CompanyRegistry()
    registry.add_many(companies)

    return registry


def load_reports(
    path: Path,
) -> tuple[ReportRegistry, list[PageMappingSegment]]:
    """从 YAML 文件加载报告 Registry 和页码映射。"""

    raw_data = load_registry_yaml(path)

    raw_reports = raw_data.get("reports")
    raw_page_mappings = raw_data.get("page_mappings")

    if not isinstance(raw_reports, list):
        raise RegistryLoaderError(
            "reports 字段必须是列表"
        )

    if not raw_reports:
        raise RegistryLoaderError(
            "reports 字段不能为空"
        )

    if not isinstance(raw_page_mappings, list):
        raise RegistryLoaderError(
            "page_mappings 字段必须是列表"
        )

    reports = [
        Report.model_validate(item)
        for item in raw_reports
    ]

    page_mappings = [
        PageMappingSegment.model_validate(item)
        for item in raw_page_mappings
    ]

    registry = ReportRegistry()
    registry.add_many(reports)

    return registry, page_mappings


def load_metrics(
    path: Path,
) -> tuple[MetricRegistry, list[MetricAlias]]:
    """从 YAML 文件加载指标 Registry 和指标别名。"""

    raw_data = load_registry_yaml(path)

    raw_metrics = raw_data.get("metrics")
    raw_aliases = raw_data.get("metric_aliases")

    if not isinstance(raw_metrics, list):
        raise RegistryLoaderError(
            "metrics 字段必须是列表"
        )

    if not raw_metrics:
        raise RegistryLoaderError(
            "metrics 字段不能为空"
        )

    if not isinstance(raw_aliases, list):
        raise RegistryLoaderError(
            "metric_aliases 字段必须是列表"
        )

    metrics = [
        FinancialMetric.model_validate(item)
        for item in raw_metrics
    ]

    aliases = [
        MetricAlias.model_validate(item)
        for item in raw_aliases
    ]

    registry = MetricRegistry()
    registry.add_many(metrics)

    return registry, aliases


def load_evidences(
    path: Path,
) -> EvidenceRegistry:
    """从 YAML 加载来源证据注册表。"""

    raw_data = load_registry_yaml(path)

    raw_evidences = raw_data.get("evidences")

    if not isinstance(raw_evidences, list):
        raise RegistryLoaderError(
            "evidences 字段必须是列表"
        )

    if not raw_evidences:
        raise RegistryLoaderError(
            "evidences 字段不能为空"
        )

    registry = EvidenceRegistry()

    for raw_evidence in raw_evidences:
        evidence = SourceEvidence.model_validate(
            raw_evidence
        )
        registry.add(evidence)

    return registry


def load_financial_facts(
    path: Path,
) -> tuple[
    FinancialFactRegistry,
    list[FactEvidenceLink],
]:
    """从 YAML 加载财务事实及事实证据关联。"""

    raw_data = load_registry_yaml(path)

    raw_facts = raw_data.get("financial_facts")
    raw_links = raw_data.get(
        "fact_evidence_links"
    )

    if not isinstance(raw_facts, list):
        raise RegistryLoaderError(
            "financial_facts 字段必须是列表"
        )

    if not raw_facts:
        raise RegistryLoaderError(
            "financial_facts 字段不能为空"
        )

    if not isinstance(raw_links, list):
        raise RegistryLoaderError(
            "fact_evidence_links 字段必须是列表"
        )

    registry = FinancialFactRegistry()

    for raw_fact in raw_facts:
        fact = FinancialFact.model_validate(
            raw_fact
        )
        registry.add(fact)

    links = [
        FactEvidenceLink.model_validate(raw_link)
        for raw_link in raw_links
    ]

    return registry, links


def validate_metric_alias_relationships(
    metric_registry: MetricRegistry,
    metric_aliases: list[MetricAlias],
) -> None:
    """检查 MetricAlias 到 FinancialMetric 的引用。"""

    errors: list[str] = []

    for alias in metric_aliases:
        if not metric_registry.contains(alias.metric_id):
            errors.append(
                f"MetricAlias '{alias.alias_id}' "
                f"引用了不存在的 FinancialMetric "
                f"'{alias.metric_id}'"
            )

    if errors:
        raise RegistryIntegrityError(errors)


def validate_page_mapping_relationships(
    report_registry: ReportRegistry,
    page_mappings: list[PageMappingSegment],
) -> None:
    """检查 PageMappingSegment 到 Report 的引用。"""

    errors: list[str] = []

    for mapping in page_mappings:
        if not report_registry.contains(mapping.report_id):
            errors.append(
                f"PageMappingSegment '{mapping.mapping_id}' "
                f"引用了不存在的 Report "
                f"'{mapping.report_id}'"
            )

    if errors:
        raise RegistryIntegrityError(errors)


def validate_fact_evidence_link_relationships(
    bundle: RegistryBundle,
    links: list[FactEvidenceLink],
) -> None:
    """校验事实与证据关联记录的引用关系。"""

    errors: list[str] = []
    seen_links: set[tuple[str, str, str]] = set()
    facts_with_primary_link: set[str] = set()

    for link in links:
        link_key = (
            link.fact_id,
            link.evidence_id,
            link.support_type.value,
        )

        if link_key in seen_links:
            errors.append(
                "FactEvidenceLink 出现重复关联："
                f"{link.fact_id} -> {link.evidence_id} "
                f"({link.support_type.value})"
            )
            continue

        seen_links.add(link_key)

        if not bundle.financial_facts.contains(
            link.fact_id
        ):
            errors.append(
                "FactEvidenceLink 引用了不存在的 "
                f"FinancialFact：{link.fact_id}"
            )
            continue

        if not bundle.evidences.contains(
            link.evidence_id
        ):
            errors.append(
                "FactEvidenceLink 引用了不存在的 "
                f"SourceEvidence：{link.evidence_id}"
            )
            continue

        fact = bundle.financial_facts.require(
            link.fact_id
        )
        evidence = bundle.evidences.require(
            link.evidence_id
        )

        if fact.report_id != evidence.report_id:
            errors.append(
                "FinancialFact 与关联 SourceEvidence "
                "所属报告不一致："
                f"{fact.fact_id} -> {evidence.evidence_id}"
            )

        if (
            link.support_type
            is EvidenceSupportType.PRIMARY
        ):
            if (
                link.evidence_id
                != fact.primary_evidence_id
            ):
                errors.append(
                    "primary FactEvidenceLink 必须指向 "
                    "FinancialFact.primary_evidence_id："
                    f"{fact.fact_id}"
                )
            else:
                facts_with_primary_link.add(
                    fact.fact_id
                )

    for fact in bundle.financial_facts.values():
        if fact.fact_id not in facts_with_primary_link:
            errors.append(
                "FinancialFact 缺少与 "
                "primary_evidence_id 对应的 "
                f"primary Link：{fact.fact_id}"
            )

    if errors:
        raise RegistryIntegrityError(errors)


def load_registry_bundle(
    *,
    companies_path: Path,
    reports_path: Path,
    metrics_path: Path,
    evidences_path: Path | None = None,
    financial_facts_path: Path | None = None,
) -> tuple[
    RegistryBundle,
    list[PageMappingSegment],
    list[MetricAlias],
    list[FactEvidenceLink],
]:
    """加载完整 RegistryBundle 及附属关系数据。"""

    if (
        (evidences_path is None)
        != (financial_facts_path is None)
    ):
        raise RegistryLoaderError(
            "evidences_path 和 financial_facts_path "
            "必须同时提供"
        )

    companies = load_companies(
        companies_path
    )

    reports, page_mappings = load_reports(
        reports_path
    )

    metrics, metric_aliases = load_metrics(
        metrics_path
    )

    evidences = EvidenceRegistry()
    financial_facts = FinancialFactRegistry()
    fact_evidence_links: list[
        FactEvidenceLink
    ] = []

    if (
        evidences_path is not None
        and financial_facts_path is not None
    ):
        evidences = load_evidences(
            evidences_path
        )

        (
            financial_facts,
            fact_evidence_links,
        ) = load_financial_facts(
            financial_facts_path
        )

    bundle = RegistryBundle(
        companies=companies,
        reports=reports,
        metrics=metrics,
        evidences=evidences,
        financial_facts=financial_facts,
    )

    bundle.validate_relationships()

    validate_page_mapping_relationships(
        reports,
        page_mappings,
    )

    validate_metric_alias_relationships(
        metrics,
        metric_aliases,
    )

    validate_fact_evidence_link_relationships(
        bundle,
        fact_evidence_links,
    )

    return (
        bundle,
        page_mappings,
        metric_aliases,
        fact_evidence_links,
    )