from pathlib import Path

import pytest
import yaml

from app.services.registry import (
    RegistryIntegrityError,
)
from app.services.registry_loader import (
    load_registry_bundle,
    load_registry_yaml,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRIES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

COMPANIES_YAML_PATH = (
    REGISTRIES_DIR / "companies.yaml"
)

REPORTS_YAML_PATH = (
    REGISTRIES_DIR / "reports.yaml"
)

METRICS_YAML_PATH = (
    REGISTRIES_DIR / "metrics.yaml"
)

EVIDENCES_YAML_PATH = (
    REGISTRIES_DIR / "evidences.yaml"
)

FINANCIAL_FACTS_YAML_PATH = (
    REGISTRIES_DIR / "financial_facts.yaml"
)


def write_yaml(
    path: Path,
    raw_data: dict,
) -> None:
    """把测试数据写入临时 YAML 文件。"""

    path.write_text(
        yaml.safe_dump(
            raw_data,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def load_bundle_with_facts(
    *,
    financial_facts_path: Path,
    evidences_path: Path = EVIDENCES_YAML_PATH,
) -> None:
    """使用指定事实和证据文件加载完整 Bundle。"""

    load_registry_bundle(
        companies_path=COMPANIES_YAML_PATH,
        reports_path=REPORTS_YAML_PATH,
        metrics_path=METRICS_YAML_PATH,
        evidences_path=evidences_path,
        financial_facts_path=financial_facts_path,
    )


def test_reject_link_referencing_missing_fact(
    tmp_path: Path,
) -> None:
    """Link 引用不存在的 Fact 时应失败。"""

    raw_data = load_registry_yaml(
        FINANCIAL_FACTS_YAML_PATH
    )

    raw_data["fact_evidence_links"][0][
        "fact_id"
    ] = "fact_missing"

    invalid_path = (
        tmp_path / "missing_fact_link.yaml"
    )

    write_yaml(
        invalid_path,
        raw_data,
    )

    with pytest.raises(
        RegistryIntegrityError,
        match="不存在的 FinancialFact",
    ):
        load_bundle_with_facts(
            financial_facts_path=invalid_path
        )


def test_reject_link_referencing_missing_evidence(
    tmp_path: Path,
) -> None:
    """Link 引用不存在的 Evidence 时应失败。"""

    raw_data = load_registry_yaml(
        FINANCIAL_FACTS_YAML_PATH
    )

    raw_data["fact_evidence_links"][0][
        "evidence_id"
    ] = "evidence_missing"

    invalid_path = (
        tmp_path / "missing_evidence_link.yaml"
    )

    write_yaml(
        invalid_path,
        raw_data,
    )

    with pytest.raises(
        RegistryIntegrityError,
        match="不存在的 SourceEvidence",
    ):
        load_bundle_with_facts(
            financial_facts_path=invalid_path
        )


def test_reject_fact_without_primary_link(
    tmp_path: Path,
) -> None:
    """Fact 缺少对应 primary Link 时应失败。"""

    raw_data = load_registry_yaml(
        FINANCIAL_FACTS_YAML_PATH
    )

    raw_data["fact_evidence_links"] = []

    invalid_path = (
        tmp_path / "missing_primary_link.yaml"
    )

    write_yaml(
        invalid_path,
        raw_data,
    )

    with pytest.raises(
        RegistryIntegrityError,
        match="缺少.*primary Link",
    ):
        load_bundle_with_facts(
            financial_facts_path=invalid_path
        )


def test_reject_duplicate_fact_evidence_link(
    tmp_path: Path,
) -> None:
    """完全相同的事实证据关联不能重复。"""

    raw_data = load_registry_yaml(
        FINANCIAL_FACTS_YAML_PATH
    )

    duplicate_link = dict(
        raw_data["fact_evidence_links"][0]
    )

    raw_data["fact_evidence_links"].append(
        duplicate_link
    )

    invalid_path = (
        tmp_path / "duplicate_link.yaml"
    )

    write_yaml(
        invalid_path,
        raw_data,
    )

    with pytest.raises(
        RegistryIntegrityError,
        match="重复关联",
    ):
        load_bundle_with_facts(
            financial_facts_path=invalid_path
        )