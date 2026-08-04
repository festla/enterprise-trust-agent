from pathlib import Path

from app.services.registry_loader import (
    load_registry_bundle,
)
from app.services.week2_quality import (
    build_week2_quality_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRIES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)


def main() -> None:
    """执行 Week 2 Registry 数据质量检查。"""

    (
        bundle,
        page_mappings,
        metric_aliases,
        fact_evidence_links,
    ) = load_registry_bundle(
        companies_path=(
            REGISTRIES_DIR / "companies.yaml"
        ),
        reports_path=(
            REGISTRIES_DIR / "reports.yaml"
        ),
        metrics_path=(
            REGISTRIES_DIR / "metrics.yaml"
        ),
        evidences_path=(
            REGISTRIES_DIR / "evidences.yaml"
        ),
        financial_facts_path=(
            REGISTRIES_DIR
            / "financial_facts.yaml"
        ),
    )

    report = build_week2_quality_report(
        bundle=bundle,
        page_mappings=page_mappings,
        metric_aliases=metric_aliases,
        fact_evidence_links=fact_evidence_links,
    )

    print("Week 2 数据质量检查")
    print("=" * 40)
    print(f"Company: {report.company_count}")
    print(f"Report: {report.report_count}")
    print(
        "PageMappingSegment: "
        f"{report.page_mapping_count}"
    )
    print(f"FinancialMetric: {report.metric_count}")
    print(
        f"MetricAlias: {report.metric_alias_count}"
    )
    print(f"SourceEvidence: {report.evidence_count}")
    print(
        "FinancialFact: "
        f"{report.financial_fact_count}"
    )
    print(
        "FactEvidenceLink: "
        f"{report.fact_evidence_link_count}"
    )
    print(
        "Verified Evidence: "
        f"{report.verified_evidence_count}"
    )
    print(
        "Verified Fact: "
        f"{report.verified_fact_count}"
    )
    print("=" * 40)

    if report.passed:
        print("RESULT: PASSED")
        return

    print("RESULT: FAILED")

    for issue in report.issues:
        print(f"- {issue}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()