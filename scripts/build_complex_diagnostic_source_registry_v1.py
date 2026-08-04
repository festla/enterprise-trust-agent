# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from app.schemas.evidence import SourceEvidence
from app.schemas.financial_fact import (
    FactEvidenceLink,
    FinancialFact,
)
from app.services.registry_loader import (
    load_evidences,
    load_financial_facts,
    load_metrics,
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

EVIDENCE_PATH = REGISTRY_ROOT / "evidences.yaml"

FACT_PATH = (
    REGISTRY_ROOT / "financial_facts.yaml"
)

METRIC_PATH = REGISTRY_ROOT / "metrics.yaml"
COMPANY_PATH = REGISTRY_ROOT / "companies.yaml"
REPORT_PATH = REGISTRY_ROOT / "reports.yaml"

EVIDENCE_BACKUP_PATH = REGISTRY_ROOT / (
    "evidences.before_complex_"
    "diagnostic_source_v1.yaml"
)

FACT_BACKUP_PATH = REGISTRY_ROOT / (
    "financial_facts.before_complex_"
    "diagnostic_source_v1.yaml"
)

TEMP_EVIDENCE_PATH = REGISTRY_ROOT / (
    "evidences.complex_diagnostic_source_v1.tmp.yaml"
)

TEMP_FACT_PATH = REGISTRY_ROOT / (
    "financial_facts.complex_"
    "diagnostic_source_v1.tmp.yaml"
)

REGISTRY_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "services"
    / "test_registry_loader.py"
)

WEEK2_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "services"
    / "test_week2_quality.py"
)


def make_spec(
    *,
    company_id: str,
    metric_id: str,
    statement_type: str,
    table_name: str,
    row_label: str,
    column_label: str,
    raw_value: str,
    raw_unit: str,
    cell_value: str,
    chunk_id: str,
) -> dict[str, str]:
    return {
        "company_id": company_id,
        "report_id": f"{company_id}_2024",
        "metric_id": metric_id,
        "statement_type": statement_type,
        "table_name": table_name,
        "row_label": row_label,
        "column_label": column_label,
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "cell_value": cell_value,
        "chunk_id": chunk_id,
    }


SPECS = (
    make_spec(
        company_id="midea_group",
        metric_id="non_current_assets",
        statement_type="balance_sheet",
        table_name="合并及公司资产负债表",
        row_label="非流动资产合计",
        column_label="2024年12月31日合并",
        raw_value="215288067",
        raw_unit="CNY_thousand",
        cell_value="215,288,067",
        chunk_id=(
            "chunk_midea_group_2024_"
            "2806f974842689f22661037d"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id=(
            "cash_outflows_from_investing_"
            "activities_subtotal"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="投资活动现金流出小计",
        column_label="2024年度",
        raw_value="50412775368.32",
        raw_unit="CNY",
        cell_value="50,412,775,368.32",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "2c8f4084dced5a2a42cdb3da"
        ),
    ),
    make_spec(
        company_id="haier_smart_home",
        metric_id="taxes_and_surcharges",
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="税金及附加",
        column_label="2024年度",
        raw_value="1276040830.27",
        raw_unit="CNY",
        cell_value="1,276,040,830.27",
        chunk_id=(
            "chunk_haier_smart_home_2024_"
            "c4bcc0d67f61ba7aa731df88"
        ),
    ),
    make_spec(
        company_id="hisense_home",
        metric_id=(
            "cash_received_from_sales_of_goods_"
            "and_rendering_of_services"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="销售商品、提供劳务收到的现金",
        column_label="2024年度",
        raw_value="76577033223.57",
        raw_unit="CNY",
        cell_value="76,577,033,223.57",
        chunk_id=(
            "chunk_hisense_home_2024_"
            "d641659ee959c6e7c56f95e5"
        ),
    ),
    make_spec(
        company_id="hisense_home",
        metric_id="tax_refunds_received",
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="收到的税费返还",
        column_label="2024年度",
        raw_value="3516048170.30",
        raw_unit="CNY",
        cell_value="3,516,048,170.30",
        chunk_id=(
            "chunk_hisense_home_2024_"
            "d878c4bcb4ff1bda02d3a327"
        ),
    ),
    make_spec(
        company_id="hisense_home",
        metric_id=(
            "net_cash_flow_from_operating_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="经营活动产生的现金流量净额",
        column_label="2024年度",
        raw_value="5132164941.24",
        raw_unit="CNY",
        cell_value="5,132,164,941.24",
        chunk_id=(
            "chunk_hisense_home_2024_"
            "d878c4bcb4ff1bda02d3a327"
        ),
    ),
    make_spec(
        company_id="hisense_home",
        metric_id=(
            "net_cash_flow_from_investing_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="投资活动产生的现金流量净额",
        column_label="2024年度",
        raw_value="-619437088.11",
        raw_unit="CNY",
        cell_value="-619,437,088.11",
        chunk_id=(
            "chunk_hisense_home_2024_"
            "a9a5885b9c49f6302ac29676"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id=(
            "net_profit_attributable_to_parent"
        ),
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="归属于母公司股东的净利润",
        column_label="2024年度",
        raw_value="32184570372.28",
        raw_unit="CNY",
        cell_value="32,184,570,372.28",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "ede3eaaa02dfbdc6beb6b587"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id=(
            "other_comprehensive_income_net_of_tax"
        ),
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="其他综合收益的税后净额",
        column_label="2024年度",
        raw_value="180264674.95",
        raw_unit="CNY",
        cell_value="180,264,674.95",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "ede3eaaa02dfbdc6beb6b587"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id="total_comprehensive_income",
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="综合收益总额",
        column_label="2024年度",
        raw_value="32551333962.30",
        raw_unit="CNY",
        cell_value="32,551,333,962.30",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "76a82b9c61a9f9a86051014c"
        ),
    ),
    make_spec(
        company_id="haier_smart_home",
        metric_id=(
            "other_comprehensive_income_net_of_tax"
        ),
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="其他综合收益的税后净额",
        column_label="2024年度",
        raw_value="-1173713256.17",
        raw_unit="CNY",
        cell_value="-1,173,713,256.17",
        chunk_id=(
            "chunk_haier_smart_home_2024_"
            "d71046f6303d37c5388381a4"
        ),
    ),
    make_spec(
        company_id="haier_smart_home",
        metric_id="total_comprehensive_income",
        statement_type="income_statement",
        table_name="合并利润表",
        row_label="综合收益总额",
        column_label="2024年度",
        raw_value="18401899245.51",
        raw_unit="CNY",
        cell_value="18,401,899,245.51",
        chunk_id=(
            "chunk_haier_smart_home_2024_"
            "bfd720a101b506409615cff0"
        ),
    ),
    make_spec(
        company_id="haier_smart_home",
        metric_id="non_current_liabilities",
        statement_type="balance_sheet",
        table_name="合并资产负债表",
        row_label="非流动负债合计",
        column_label="2024年12月31日",
        raw_value="22153482887.26",
        raw_unit="CNY",
        cell_value="22,153,482,887.26",
        chunk_id=(
            "chunk_haier_smart_home_2024_"
            "cceb1aa581911957d77e7589"
        ),
    ),
    make_spec(
        company_id="midea_group",
        metric_id=(
            "net_cash_flow_from_investing_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="2024 年度合并及公司现金流量表",
        row_label="投资活动使用的现金流量净额",
        column_label="2024年度合并",
        raw_value="-87901802",
        raw_unit="CNY_thousand",
        cell_value="(87,901,802)",
        chunk_id=(
            "chunk_midea_group_2024_"
            "202c4c766817c2371d495230"
        ),
    ),
    make_spec(
        company_id="midea_group",
        metric_id=(
            "net_cash_flow_from_financing_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="2024 年度合并及公司现金流量表",
        row_label="筹资活动产生/(使用)的现金流量净额",
        column_label="2024年度合并",
        raw_value="22697954",
        raw_unit="CNY_thousand",
        cell_value="22,697,954",
        chunk_id=(
            "chunk_midea_group_2024_"
            "743e54374724e436306930f6"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id=(
            "net_cash_flow_from_investing_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="投资活动产生的现金流量净额",
        column_label="2024年度",
        raw_value="-15557909615.57",
        raw_unit="CNY",
        cell_value="-15,557,909,615.57",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "2c8f4084dced5a2a42cdb3da"
        ),
    ),
    make_spec(
        company_id="gree_electric",
        metric_id=(
            "net_cash_flow_from_financing_activities"
        ),
        statement_type="cash_flow_statement",
        table_name="合并现金流量表",
        row_label="筹资活动产生的现金流量净额",
        column_label="2024年度",
        raw_value="-23703212908.16",
        raw_unit="CNY",
        cell_value="-23,703,212,908.16",
        chunk_id=(
            "chunk_gree_electric_2024_"
            "2c8f4084dced5a2a42cdb3da"
        ),
    ),
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise SystemExit(
            f"yaml_root_not_mapping={path}"
        )

    return data


def write_yaml(
    path: Path,
    data: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        )


def find_target_chunks() -> dict[str, dict]:
    target_ids = {
        spec["chunk_id"]
        for spec in SPECS
    }

    chunks: dict[str, dict] = {}

    chunk_root = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "chunks"
    )

    for path in chunk_root.rglob("chunks.jsonl"):
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                chunk = json.loads(line)
                chunk_id = chunk.get("chunk_id")

                if chunk_id not in target_ids:
                    continue

                if chunk_id in chunks:
                    old = chunks[chunk_id]

                    identity_fields = (
                        "company_id",
                        "report_id",
                        "document_id",
                        "page_id",
                        "pdf_page",
                        "printed_page",
                        "text",
                    )

                    if any(
                        old.get(field)
                        != chunk.get(field)
                        for field in identity_fields
                    ):
                        raise SystemExit(
                            "conflicting_chunk_identity="
                            f"{chunk_id}"
                        )

                    continue

                chunks[chunk_id] = chunk

    missing_chunk_ids = sorted(
        target_ids - set(chunks)
    )

    if missing_chunk_ids:
        raise SystemExit(
            "missing_chunk_ids="
            + ",".join(missing_chunk_ids)
        )

    return chunks


def validate_specs_and_chunks(
    chunks: dict[str, dict],
) -> None:
    metrics, _ = load_metrics(METRIC_PATH)

    fact_ids = []
    evidence_ids = []

    for spec in SPECS:
        company_id = spec["company_id"]
        metric_id = spec["metric_id"]

        fact_ids.append(
            f"fact_{company_id}_2024_{metric_id}"
        )

        evidence_ids.append(
            f"evidence_{company_id}_2024_{metric_id}"
        )

        chunk = chunks[spec["chunk_id"]]

        if chunk["company_id"] != company_id:
            raise SystemExit(
                "chunk_company_mismatch="
                f"{spec['chunk_id']}"
            )

        if (
            chunk["report_id"]
            != spec["report_id"]
        ):
            raise SystemExit(
                "chunk_report_mismatch="
                f"{spec['chunk_id']}"
            )

        if chunk["fiscal_year"] != 2024:
            raise SystemExit(
                "chunk_year_mismatch="
                f"{spec['chunk_id']}"
            )

        if chunk["mapping_status"] != "mapped":
            raise SystemExit(
                "chunk_not_mapped="
                f"{spec['chunk_id']}"
            )

        if chunk["parse_status"] != "success":
            raise SystemExit(
                "chunk_parse_not_success="
                f"{spec['chunk_id']}"
            )

        if chunk.get("printed_page") is None:
            raise SystemExit(
                "chunk_missing_printed_page="
                f"{spec['chunk_id']}"
            )

        if (
            spec["row_label"]
            not in chunk["text"]
        ):
            raise SystemExit(
                "row_label_not_in_chunk="
                f"{metric_id}"
            )

        if (
            spec["cell_value"]
            not in chunk["text"]
        ):
            raise SystemExit(
                "cell_value_not_in_chunk="
                f"{metric_id}"
            )

        metric = metrics.require(metric_id)

        if (
            metric.statement_type.value
            != spec["statement_type"]
        ):
            raise SystemExit(
                "metric_statement_type_mismatch="
                f"{metric_id}"
            )

        if metric.default_unit.value != "CNY":
            raise SystemExit(
                "metric_default_unit_not_cny="
                f"{metric_id}"
            )

        if "consolidated" not in {
            scope.value
            for scope in metric.allowed_scopes
        }:
            raise SystemExit(
                "metric_disallows_consolidated="
                f"{metric_id}"
            )

    duplicate_fact_ids = sorted(
        item
        for item, count
        in Counter(fact_ids).items()
        if count > 1
    )

    duplicate_evidence_ids = sorted(
        item
        for item, count
        in Counter(evidence_ids).items()
        if count > 1
    )

    if duplicate_fact_ids:
        raise SystemExit(
            "duplicate_target_fact_ids="
            + ",".join(duplicate_fact_ids)
        )

    if duplicate_evidence_ids:
        raise SystemExit(
            "duplicate_target_evidence_ids="
            + ",".join(duplicate_evidence_ids)
        )


def build_records(
    chunks: dict[str, dict],
    timestamp: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    evidences = []
    facts = []
    links = []

    for spec in SPECS:
        company_id = spec["company_id"]
        report_id = spec["report_id"]
        metric_id = spec["metric_id"]

        evidence_id = (
            f"evidence_{company_id}_2024_"
            f"{metric_id}"
        )

        fact_id = (
            f"fact_{company_id}_2024_"
            f"{metric_id}"
        )

        chunk = chunks[spec["chunk_id"]]

        raw_value = Decimal(spec["raw_value"])

        if spec["raw_unit"] == "CNY_thousand":
            multiplier = Decimal("1000")
            unit_text = "人民币千元"
        elif spec["raw_unit"] == "CNY":
            multiplier = Decimal("1")
            unit_text = "人民币元"
        else:
            raise SystemExit(
                "unsupported_raw_unit="
                f"{spec['raw_unit']}"
            )

        normalized_value = (
            raw_value * multiplier
        )

        evidence_text = "｜".join(
            (
                spec["table_name"],
                spec["row_label"],
                spec["column_label"],
                spec["cell_value"],
                unit_text,
            )
        )

        source_hash = hashlib.sha256(
            evidence_text.encode("utf-8")
        ).hexdigest()

        evidence_data = {
            "evidence_id": evidence_id,
            "report_id": report_id,
            "document_id": chunk["document_id"],
            "page_id": chunk["page_id"],
            "chunk_id": chunk["chunk_id"],
            "evidence_type": (
                "financial_statement_cell"
            ),
            "attribution_type": (
                "report_disclosure"
            ),
            "statement_type": (
                spec["statement_type"]
            ),
            "statement_scope": "consolidated",
            "section_title": "财务报告",
            "table_name": spec["table_name"],
            "row_label": spec["row_label"],
            "column_label": spec["column_label"],
            "printed_page": (
                chunk["printed_page"]
            ),
            "pdf_page": chunk["pdf_page"],
            "evidence_text": evidence_text,
            "cell_value": spec["cell_value"],
            "source_hash": source_hash,
            "validation_status": "pending",
            "created_at": timestamp,
        }

        if (
            spec["statement_type"]
            == "balance_sheet"
        ):
            period_fields = {
                "period_type": "instant",
                "as_of_date": "2024-12-31",
            }
        else:
            period_fields = {
                "period_type": "duration",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }

        fact_data = {
            "fact_id": fact_id,
            "company_id": company_id,
            "report_id": report_id,
            "metric_id": metric_id,
            "fiscal_year": 2024,
            "statement_type": (
                spec["statement_type"]
            ),
            "statement_scope": "consolidated",
            "raw_value": format(
                raw_value,
                "f",
            ),
            "raw_unit": spec["raw_unit"],
            "unit_multiplier": format(
                multiplier,
                "f",
            ),
            "normalized_value": format(
                normalized_value,
                "f",
            ),
            "normalized_unit": "CNY",
            "currency": "CNY",
            "table_name": spec["table_name"],
            "row_label": spec["row_label"],
            "column_label": spec["column_label"],
            "is_comparative_value": False,
            "restatement_status": (
                "not_applicable"
            ),
            "primary_evidence_id": evidence_id,
            "validation_status": "pending",
            "source_version": (
                f"{report_id}_v1"
            ),
            "created_at": timestamp,
            "updated_at": timestamp,
            **period_fields,
        }

        link_data = {
            "fact_id": fact_id,
            "evidence_id": evidence_id,
            "support_type": "primary",
        }

        evidence_model = (
            SourceEvidence.model_validate(
                evidence_data
            )
        )

        fact_model = FinancialFact.model_validate(
            fact_data
        )

        link_model = (
            FactEvidenceLink.model_validate(
                link_data
            )
        )

        evidences.append(
            evidence_model.model_dump(
                mode="json",
                exclude_none=True,
            )
        )

        facts.append(
            fact_model.model_dump(
                mode="json",
                exclude_none=True,
            )
        )

        links.append(
            link_model.model_dump(
                mode="json",
                exclude_none=True,
            )
        )

    return evidences, facts, links


def validate_raw_candidate(
    evidence_data: dict,
    fact_data: dict,
) -> None:
    evidence_models = [
        SourceEvidence.model_validate(item)
        for item in evidence_data["evidences"]
    ]

    fact_models = [
        FinancialFact.model_validate(item)
        for item in fact_data[
            "financial_facts"
        ]
    ]

    link_models = [
        FactEvidenceLink.model_validate(item)
        for item in fact_data[
            "fact_evidence_links"
        ]
    ]

    checks = (
        (
            "evidence_id",
            [
                item.evidence_id
                for item in evidence_models
            ],
        ),
        (
            "fact_id",
            [
                item.fact_id
                for item in fact_models
            ],
        ),
    )

    for name, values in checks:
        duplicates = sorted(
            value
            for value, count
            in Counter(values).items()
            if count > 1
        )

        if duplicates:
            raise SystemExit(
                f"duplicate_{name}s="
                + ",".join(duplicates)
            )

    link_pairs = [
        (
            link.fact_id,
            link.evidence_id,
        )
        for link in link_models
    ]

    duplicate_link_pairs = sorted(
        pair
        for pair, count
        in Counter(link_pairs).items()
        if count > 1
    )

    if duplicate_link_pairs:
        raise SystemExit(
            "duplicate_link_pairs="
            f"{duplicate_link_pairs}"
        )


def validate_bundle(
    evidence_path: Path,
    fact_path: Path,
) -> tuple[Any, list, list, list]:
    return load_registry_bundle(
        companies_path=COMPANY_PATH,
        reports_path=REPORT_PATH,
        metrics_path=METRIC_PATH,
        evidences_path=evidence_path,
        financial_facts_path=fact_path,
    )


def synchronize_test_counts() -> None:
    paths = (
        REGISTRY_TEST_PATH,
        WEEK2_TEST_PATH,
    )

    for path in paths:
        if not path.is_file():
            raise SystemExit(
                f"missing_test_file={path}"
            )

    registry_text = (
        REGISTRY_TEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    registry_transitions = (
        (
            "assert len(registry.values()) == 61",
            "assert len(registry.values()) == 78",
            2,
        ),
        (
            "assert len(links) == 61",
            "assert len(links) == 78",
            1,
        ),
        (
            "assert len(bundle.evidences) == 61",
            "assert len(bundle.evidences) == 78",
            1,
        ),
        (
            (
                "assert "
                "len(bundle.financial_facts) == 61"
            ),
            (
                "assert "
                "len(bundle.financial_facts) == 78"
            ),
            1,
        ),
        (
            (
                "assert "
                "len(fact_evidence_links) == 61"
            ),
            (
                "assert "
                "len(fact_evidence_links) == 78"
            ),
            1,
        ),
    )

    registry_changed = False

    for old, new, expected_count in (
        registry_transitions
    ):
        old_count = registry_text.count(old)
        new_count = registry_text.count(new)

        if old_count == expected_count:
            registry_text = registry_text.replace(
                old,
                new,
            )
            registry_changed = True

        elif (
            old_count == 0
            and new_count == expected_count
        ):
            continue

        else:
            raise SystemExit(
                "unexpected_registry_test_"
                "transition_count="
                f"{old!r}:"
                f"old={old_count},new={new_count}"
            )

    if registry_changed:
        backup_path = (
            REGISTRY_TEST_PATH.with_name(
                REGISTRY_TEST_PATH.name
                + ".before_complex_diagnostic_"
                + "source_v1"
            )
        )

        if not backup_path.exists():
            shutil.copy2(
                REGISTRY_TEST_PATH,
                backup_path,
            )

        REGISTRY_TEST_PATH.write_text(
            registry_text,
            encoding="utf-8",
            newline="\n",
        )

    week2_text = WEEK2_TEST_PATH.read_text(
        encoding="utf-8"
    )

    week2_transitions = (
        (
            "assert report.evidence_count == 61",
            "assert report.evidence_count == 78",
        ),
        (
            (
                "assert "
                "report.financial_fact_count == 61"
            ),
            (
                "assert "
                "report.financial_fact_count == 78"
            ),
        ),
    )

    week2_changed = False

    for old, new in week2_transitions:
        old_count = week2_text.count(old)
        new_count = week2_text.count(new)

        if old_count == 1:
            week2_text = week2_text.replace(
                old,
                new,
            )
            week2_changed = True

        elif old_count == 0 and new_count == 1:
            continue

        else:
            raise SystemExit(
                "unexpected_week2_test_"
                "transition_count="
                f"{old!r}:"
                f"old={old_count},new={new_count}"
            )

    if week2_changed:
        backup_path = WEEK2_TEST_PATH.with_name(
            WEEK2_TEST_PATH.name
            + ".before_complex_diagnostic_"
            + "source_v1"
        )

        if not backup_path.exists():
            shutil.copy2(
                WEEK2_TEST_PATH,
                backup_path,
            )

        WEEK2_TEST_PATH.write_text(
            week2_text,
            encoding="utf-8",
            newline="\n",
        )

    print(
        "registry_test_counts_synced=true"
    )
    print("week2_test_counts_synced=true")
    print(
        "verified_count_expectation_preserved=61"
    )


def main() -> None:
    required_paths = (
        EVIDENCE_PATH,
        FACT_PATH,
        METRIC_PATH,
        COMPANY_PATH,
        REPORT_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            raise SystemExit(
                f"missing_required_file={path}"
            )

    chunks = find_target_chunks()

    validate_specs_and_chunks(chunks)

    evidence_raw = read_yaml(EVIDENCE_PATH)
    fact_raw = read_yaml(FACT_PATH)

    old_evidences = deepcopy(
        evidence_raw["evidences"]
    )

    old_facts = deepcopy(
        fact_raw["financial_facts"]
    )

    old_links = deepcopy(
        fact_raw["fact_evidence_links"]
    )

    target_evidence_ids = {
        (
            f"evidence_{spec['company_id']}_"
            f"2024_{spec['metric_id']}"
        )
        for spec in SPECS
    }

    target_fact_ids = {
        (
            f"fact_{spec['company_id']}_"
            f"2024_{spec['metric_id']}"
        )
        for spec in SPECS
    }

    target_link_pairs = {
        (
            (
                f"fact_{spec['company_id']}_"
                f"2024_{spec['metric_id']}"
            ),
            (
                f"evidence_{spec['company_id']}_"
                f"2024_{spec['metric_id']}"
            ),
        )
        for spec in SPECS
    }

    existing_evidence_ids = {
        item["evidence_id"]
        for item in old_evidences
    }

    existing_fact_ids = {
        item["fact_id"]
        for item in old_facts
    }

    existing_link_pairs = {
        (
            item["fact_id"],
            item["evidence_id"],
        )
        for item in old_links
    }

    existing_target_evidence_ids = (
        target_evidence_ids
        & existing_evidence_ids
    )

    existing_target_fact_ids = (
        target_fact_ids
        & existing_fact_ids
    )

    existing_target_link_pairs = (
        target_link_pairs
        & existing_link_pairs
    )

    all_already_present = (
        existing_target_evidence_ids
        == target_evidence_ids
        and existing_target_fact_ids
        == target_fact_ids
        and existing_target_link_pairs
        == target_link_pairs
    )

    any_already_present = any(
        (
            existing_target_evidence_ids,
            existing_target_fact_ids,
            existing_target_link_pairs,
        )
    )

    added = False

    if all_already_present:
        print(
            "diagnostic_source_records_"
            "already_present=true"
        )

    elif any_already_present:
        raise SystemExit(
            "partial_diagnostic_source_"
            "records_detected=true"
        )

    else:
        if len(old_evidences) != 61:
            raise SystemExit(
                "unexpected_before_evidence_count="
                f"{len(old_evidences)}"
            )

        if len(old_facts) != 61:
            raise SystemExit(
                "unexpected_before_fact_count="
                f"{len(old_facts)}"
            )

        if len(old_links) != 61:
            raise SystemExit(
                "unexpected_before_link_count="
                f"{len(old_links)}"
            )

        timestamp = datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat()

        (
            new_evidences,
            new_facts,
            new_links,
        ) = build_records(
            chunks,
            timestamp,
        )

        evidence_candidate = deepcopy(
            evidence_raw
        )

        fact_candidate = deepcopy(fact_raw)

        evidence_candidate["evidences"].extend(
            new_evidences
        )

        fact_candidate[
            "financial_facts"
        ].extend(new_facts)

        fact_candidate[
            "fact_evidence_links"
        ].extend(new_links)

        if (
            evidence_candidate["evidences"][:61]
            != old_evidences
        ):
            raise SystemExit(
                "old_evidence_records_changed=true"
            )

        if (
            fact_candidate[
                "financial_facts"
            ][:61]
            != old_facts
        ):
            raise SystemExit(
                "old_fact_records_changed=true"
            )

        if (
            fact_candidate[
                "fact_evidence_links"
            ][:61]
            != old_links
        ):
            raise SystemExit(
                "old_link_records_changed=true"
            )

        validate_raw_candidate(
            evidence_candidate,
            fact_candidate,
        )

        write_yaml(
            TEMP_EVIDENCE_PATH,
            evidence_candidate,
        )

        write_yaml(
            TEMP_FACT_PATH,
            fact_candidate,
        )

        try:
            (
                temporary_bundle,
                _,
                _,
                temporary_links,
            ) = validate_bundle(
                TEMP_EVIDENCE_PATH,
                TEMP_FACT_PATH,
            )

            if (
                len(temporary_bundle.evidences)
                != 78
            ):
                raise SystemExit(
                    "temporary_evidence_count_"
                    "not_78"
                )

            if (
                len(
                    temporary_bundle.financial_facts
                )
                != 78
            ):
                raise SystemExit(
                    "temporary_fact_count_not_78"
                )

            if len(temporary_links) != 78:
                raise SystemExit(
                    "temporary_link_count_not_78"
                )

            if not EVIDENCE_BACKUP_PATH.exists():
                shutil.copy2(
                    EVIDENCE_PATH,
                    EVIDENCE_BACKUP_PATH,
                )

            if not FACT_BACKUP_PATH.exists():
                shutil.copy2(
                    FACT_PATH,
                    FACT_BACKUP_PATH,
                )

            TEMP_EVIDENCE_PATH.replace(
                EVIDENCE_PATH
            )

            TEMP_FACT_PATH.replace(FACT_PATH)

            added = True

        except Exception:
            if EVIDENCE_BACKUP_PATH.exists():
                shutil.copy2(
                    EVIDENCE_BACKUP_PATH,
                    EVIDENCE_PATH,
                )

            if FACT_BACKUP_PATH.exists():
                shutil.copy2(
                    FACT_BACKUP_PATH,
                    FACT_PATH,
                )

            raise

        finally:
            if TEMP_EVIDENCE_PATH.exists():
                TEMP_EVIDENCE_PATH.unlink()

            if TEMP_FACT_PATH.exists():
                TEMP_FACT_PATH.unlink()

    (
        bundle,
        _,
        _,
        final_links,
    ) = validate_bundle(
        EVIDENCE_PATH,
        FACT_PATH,
    )

    pending_target_evidence_count = sum(
        bundle.evidences.require(
            evidence_id
        ).validation_status.value
        == "pending"
        for evidence_id in target_evidence_ids
    )

    pending_target_fact_count = sum(
        bundle.financial_facts.require(
            fact_id
        ).validation_status.value
        == "pending"
        for fact_id in target_fact_ids
    )

    final_link_pairs = {
        (
            link.fact_id,
            link.evidence_id,
        )
        for link in final_links
    }

    missing_target_links = sorted(
        target_link_pairs - final_link_pairs
    )

    if len(bundle.evidences) != 78:
        raise SystemExit(
            "final_evidence_count_not_78"
        )

    if len(bundle.financial_facts) != 78:
        raise SystemExit(
            "final_fact_count_not_78"
        )

    if len(final_links) != 78:
        raise SystemExit(
            "final_link_count_not_78"
        )

    if pending_target_evidence_count != 17:
        raise SystemExit(
            "pending_target_evidence_"
            "count_not_17"
        )

    if pending_target_fact_count != 17:
        raise SystemExit(
            "pending_target_fact_count_not_17"
        )

    if missing_target_links:
        raise SystemExit(
            "missing_target_links="
            f"{missing_target_links}"
        )

    synchronize_test_counts()

    print("-" * 72)
    print(f"evidence_path={EVIDENCE_PATH}")
    print(f"fact_path={FACT_PATH}")
    print(
        f"evidence_backup_path="
        f"{EVIDENCE_BACKUP_PATH}"
    )
    print(
        f"fact_backup_path={FACT_BACKUP_PATH}"
    )
    print(f"records_added={added}")
    print(
        f"source_spec_count={len(SPECS)}"
    )
    print(
        f"source_chunk_count={len(chunks)}"
    )
    print(
        f"total_evidence_count="
        f"{len(bundle.evidences)}"
    )
    print(
        f"total_financial_fact_count="
        f"{len(bundle.financial_facts)}"
    )
    print(
        f"total_link_count={len(final_links)}"
    )
    print(
        "pending_target_evidence_count="
        f"{pending_target_evidence_count}"
    )
    print(
        "pending_target_fact_count="
        f"{pending_target_fact_count}"
    )
    verified_existing_evidence_count = sum(
        item.validation_status.value
        == "verified"
        for item in bundle.evidences.values()
    )

    verified_existing_fact_count = sum(
        item.validation_status.value
        == "verified"
        for item
        in bundle.financial_facts.values()
    )

    print(
        "verified_existing_evidence_count="
        f"{verified_existing_evidence_count}"
    )
    print(
        "verified_existing_fact_count="
        f"{verified_existing_fact_count}"
    )
    print("source_chunk_validation_passed=true")
    print(
        "registry_relationship_validation_"
        "passed=true"
    )
    print(
        "complex_diagnostic_source_batch_"
        "written=true"
    )

    for spec in SPECS:
        fact_id = (
            f"fact_{spec['company_id']}_"
            f"2024_{spec['metric_id']}"
        )

        fact = bundle.financial_facts.require(
            fact_id
        )

        evidence = bundle.evidences.require(
            fact.primary_evidence_id
        )

        print("-" * 72)
        print(f"fact_id={fact.fact_id}")
        print(f"metric_id={fact.metric_id}")
        print(
            f"raw={fact.raw_value} "
            f"{fact.raw_unit.value}"
        )
        print(
            f"normalized="
            f"{fact.normalized_value} "
            f"{fact.normalized_unit.value}"
        )
        print(
            f"pdf_page={evidence.pdf_page}, "
            f"printed_page="
            f"{evidence.printed_page}"
        )
        print(f"chunk_id={evidence.chunk_id}")
        print(
            f"status="
            f"{fact.validation_status.value}"
        )


if __name__ == "__main__":
    main()