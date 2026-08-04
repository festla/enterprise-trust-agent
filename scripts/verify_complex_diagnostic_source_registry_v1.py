from __future__ import annotations

import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from app.services.registry_loader import (
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

COMPANY_PATH = REGISTRY_ROOT / "companies.yaml"
REPORT_PATH = REGISTRY_ROOT / "reports.yaml"
METRIC_PATH = REGISTRY_ROOT / "metrics.yaml"

EVIDENCE_BACKUP_PATH = REGISTRY_ROOT / (
    "evidences.before_complex_diagnostic_"
    "manual_verification_v1.yaml"
)

FACT_BACKUP_PATH = REGISTRY_ROOT / (
    "financial_facts.before_complex_"
    "diagnostic_manual_verification_v1.yaml"
)

TEMP_EVIDENCE_PATH = REGISTRY_ROOT / (
    "evidences.complex_diagnostic_"
    "manual_verification_v1.tmp.yaml"
)

TEMP_FACT_PATH = REGISTRY_ROOT / (
    "financial_facts.complex_diagnostic_"
    "manual_verification_v1.tmp.yaml"
)

WEEK2_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "services"
    / "test_week2_quality.py"
)

VALIDATED_BY = "manual_review"


TARGETS = (
    (
        "midea_group",
        "non_current_assets",
    ),
    (
        "gree_electric",
        (
            "cash_outflows_from_investing_"
            "activities_subtotal"
        ),
    ),
    (
        "haier_smart_home",
        "taxes_and_surcharges",
    ),
    (
        "hisense_home",
        (
            "cash_received_from_sales_of_goods_"
            "and_rendering_of_services"
        ),
    ),
    (
        "hisense_home",
        "tax_refunds_received",
    ),
    (
        "hisense_home",
        (
            "net_cash_flow_from_operating_"
            "activities"
        ),
    ),
    (
        "hisense_home",
        (
            "net_cash_flow_from_investing_"
            "activities"
        ),
    ),
    (
        "gree_electric",
        (
            "net_profit_attributable_to_parent"
        ),
    ),
    (
        "gree_electric",
        (
            "other_comprehensive_income_"
            "net_of_tax"
        ),
    ),
    (
        "gree_electric",
        "total_comprehensive_income",
    ),
    (
        "haier_smart_home",
        (
            "other_comprehensive_income_"
            "net_of_tax"
        ),
    ),
    (
        "haier_smart_home",
        "total_comprehensive_income",
    ),
    (
        "haier_smart_home",
        "non_current_liabilities",
    ),
    (
        "midea_group",
        (
            "net_cash_flow_from_investing_"
            "activities"
        ),
    ),
    (
        "midea_group",
        (
            "net_cash_flow_from_financing_"
            "activities"
        ),
    ),
    (
        "gree_electric",
        (
            "net_cash_flow_from_investing_"
            "activities"
        ),
    ),
    (
        "gree_electric",
        (
            "net_cash_flow_from_financing_"
            "activities"
        ),
    ),
)


TARGET_EVIDENCE_IDS = {
    f"evidence_{company_id}_2024_{metric_id}"
    for company_id, metric_id in TARGETS
}

TARGET_FACT_IDS = {
    f"fact_{company_id}_2024_{metric_id}"
    for company_id, metric_id in TARGETS
}


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


def validate_bundle(
    evidence_path: Path,
    fact_path: Path,
):
    return load_registry_bundle(
        companies_path=COMPANY_PATH,
        reports_path=REPORT_PATH,
        metrics_path=METRIC_PATH,
        evidences_path=evidence_path,
        financial_facts_path=fact_path,
    )


def synchronize_verified_test_counts() -> None:
    if not WEEK2_TEST_PATH.is_file():
        raise SystemExit(
            f"missing_test_file={WEEK2_TEST_PATH}"
        )

    text = WEEK2_TEST_PATH.read_text(
        encoding="utf-8"
    )

    transitions = (
        (
            (
                "assert "
                "report.verified_evidence_count == 61"
            ),
            (
                "assert "
                "report.verified_evidence_count == 78"
            ),
        ),
        (
            (
                "assert "
                "report.verified_fact_count == 61"
            ),
            (
                "assert "
                "report.verified_fact_count == 78"
            ),
        ),
    )

    changed = False

    for old, new in transitions:
        old_count = text.count(old)
        new_count = text.count(new)

        if old_count == 1:
            text = text.replace(old, new)
            changed = True

        elif old_count == 0 and new_count == 1:
            continue

        else:
            raise SystemExit(
                "unexpected_verified_test_"
                "transition_count="
                f"{old!r}:"
                f"old={old_count},"
                f"new={new_count}"
            )

    if changed:
        backup_path = WEEK2_TEST_PATH.with_name(
            WEEK2_TEST_PATH.name
            + ".before_complex_diagnostic_"
            + "manual_verification_v1"
        )

        if not backup_path.exists():
            shutil.copy2(
                WEEK2_TEST_PATH,
                backup_path,
            )

        WEEK2_TEST_PATH.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )

    print(
        "verified_test_counts_synced=true"
    )


def main() -> None:
    for path in (
        EVIDENCE_PATH,
        FACT_PATH,
        COMPANY_PATH,
        REPORT_PATH,
        METRIC_PATH,
    ):
        if not path.is_file():
            raise SystemExit(
                f"missing_required_file={path}"
            )

    if len(TARGET_EVIDENCE_IDS) != 17:
        raise SystemExit(
            "target_evidence_id_count_not_17"
        )

    if len(TARGET_FACT_IDS) != 17:
        raise SystemExit(
            "target_fact_id_count_not_17"
        )

    evidence_raw = read_yaml(EVIDENCE_PATH)
    fact_raw = read_yaml(FACT_PATH)

    evidences = evidence_raw["evidences"]
    facts = fact_raw["financial_facts"]
    links = fact_raw["fact_evidence_links"]

    if len(evidences) != 78:
        raise SystemExit(
            "unexpected_evidence_count="
            f"{len(evidences)}"
        )

    if len(facts) != 78:
        raise SystemExit(
            "unexpected_fact_count="
            f"{len(facts)}"
        )

    if len(links) != 78:
        raise SystemExit(
            "unexpected_link_count="
            f"{len(links)}"
        )

    evidence_by_id = {
        item["evidence_id"]: item
        for item in evidences
    }

    fact_by_id = {
        item["fact_id"]: item
        for item in facts
    }

    missing_evidence_ids = sorted(
        TARGET_EVIDENCE_IDS
        - set(evidence_by_id)
    )

    missing_fact_ids = sorted(
        TARGET_FACT_IDS - set(fact_by_id)
    )

    if missing_evidence_ids:
        raise SystemExit(
            "missing_target_evidence_ids="
            + ",".join(missing_evidence_ids)
        )

    if missing_fact_ids:
        raise SystemExit(
            "missing_target_fact_ids="
            + ",".join(missing_fact_ids)
        )

    evidence_status_counts = Counter(
        evidence_by_id[
            evidence_id
        ]["validation_status"]
        for evidence_id
        in TARGET_EVIDENCE_IDS
    )

    fact_status_counts = Counter(
        fact_by_id[
            fact_id
        ]["validation_status"]
        for fact_id in TARGET_FACT_IDS
    )

    all_pending = (
        evidence_status_counts == {"pending": 17}
        and fact_status_counts == {"pending": 17}
    )

    all_verified = (
        evidence_status_counts == {"verified": 17}
        and fact_status_counts == {"verified": 17}
    )

    written = False

    if all_verified:
        print(
            "manual_verification_"
            "already_written=true"
        )

    elif not all_pending:
        raise SystemExit(
            "mixed_target_validation_statuses="
            f"evidence:{dict(evidence_status_counts)},"
            f"fact:{dict(fact_status_counts)}"
        )

    else:
        timestamp = datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat()

        evidence_candidate = deepcopy(
            evidence_raw
        )

        fact_candidate = deepcopy(fact_raw)

        for item in evidence_candidate[
            "evidences"
        ]:
            if (
                item["evidence_id"]
                not in TARGET_EVIDENCE_IDS
            ):
                continue

            item["validation_status"] = (
                "verified"
            )

            item["validated_by"] = VALIDATED_BY

        for item in fact_candidate[
            "financial_facts"
        ]:
            if (
                item["fact_id"]
                not in TARGET_FACT_IDS
            ):
                continue

            item["validation_status"] = (
                "verified"
            )

            item["validated_by"] = VALIDATED_BY
            item["validated_at"] = timestamp
            item["updated_at"] = timestamp

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

            temporary_verified_evidence = sum(
                evidence.validation_status.value
                == "verified"
                for evidence
                in temporary_bundle.evidences.values()
            )

            temporary_verified_facts = sum(
                fact.validation_status.value
                == "verified"
                for fact
                in temporary_bundle.financial_facts.values()
            )

            if temporary_verified_evidence != 78:
                raise SystemExit(
                    "temporary_verified_evidence_"
                    "count_not_78"
                )

            if temporary_verified_facts != 78:
                raise SystemExit(
                    "temporary_verified_fact_"
                    "count_not_78"
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

            written = True

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
        final_bundle,
        _,
        _,
        final_links,
    ) = validate_bundle(
        EVIDENCE_PATH,
        FACT_PATH,
    )

    target_verified_evidence_count = sum(
        final_bundle.evidences.require(
            evidence_id
        ).validation_status.value
        == "verified"
        for evidence_id
        in TARGET_EVIDENCE_IDS
    )

    target_verified_fact_count = sum(
        final_bundle.financial_facts.require(
            fact_id
        ).validation_status.value
        == "verified"
        for fact_id in TARGET_FACT_IDS
    )

    total_verified_evidence_count = sum(
        item.validation_status.value
        == "verified"
        for item
        in final_bundle.evidences.values()
    )

    total_verified_fact_count = sum(
        item.validation_status.value
        == "verified"
        for item
        in final_bundle.financial_facts.values()
    )

    remaining_pending_evidence_count = sum(
        item.validation_status.value
        == "pending"
        for item
        in final_bundle.evidences.values()
    )

    remaining_pending_fact_count = sum(
        item.validation_status.value
        == "pending"
        for item
        in final_bundle.financial_facts.values()
    )

    if target_verified_evidence_count != 17:
        raise SystemExit(
            "target_verified_evidence_"
            "count_not_17"
        )

    if target_verified_fact_count != 17:
        raise SystemExit(
            "target_verified_fact_count_not_17"
        )

    if total_verified_evidence_count != 78:
        raise SystemExit(
            "total_verified_evidence_"
            "count_not_78"
        )

    if total_verified_fact_count != 78:
        raise SystemExit(
            "total_verified_fact_count_not_78"
        )

    if remaining_pending_evidence_count != 0:
        raise SystemExit(
            "remaining_pending_evidence_"
            "count_not_0"
        )

    if remaining_pending_fact_count != 0:
        raise SystemExit(
            "remaining_pending_fact_"
            "count_not_0"
        )

    if len(final_links) != 78:
        raise SystemExit(
            "final_link_count_not_78"
        )

    synchronize_verified_test_counts()

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
    print(f"verification_written={written}")
    print(
        "target_verified_evidence_count="
        f"{target_verified_evidence_count}"
    )
    print(
        "target_verified_fact_count="
        f"{target_verified_fact_count}"
    )
    print(
        "total_verified_evidence_count="
        f"{total_verified_evidence_count}"
    )
    print(
        "total_verified_fact_count="
        f"{total_verified_fact_count}"
    )
    print(
        "remaining_pending_evidence_count="
        f"{remaining_pending_evidence_count}"
    )
    print(
        "remaining_pending_fact_count="
        f"{remaining_pending_fact_count}"
    )
    print(f"validated_by={VALIDATED_BY}")
    print(
        "registry_relationship_validation_"
        "passed=true"
    )
    print(
        "complex_diagnostic_manual_"
        "verification_written=true"
    )


if __name__ == "__main__":
    main()