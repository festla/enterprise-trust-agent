from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    validate_complex_plan_eval_integrity,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPLEX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
)

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

DATASET_PATH = (
    COMPLEX_ROOT
    / "complex_plan_diagnostic_dev_v1.jsonl"
)

MANIFEST_PATH = (
    COMPLEX_ROOT
    / "complex_plan_diagnostic_dev_v1_manifest.json"
)

DEV_V2_PATH = (
    COMPLEX_ROOT
    / "complex_plan_dev_v2.jsonl"
)

TEST_V1_PATH = (
    COMPLEX_ROOT
    / "complex_plan_test_v1.jsonl"
)

DATASET_BACKUP_PATH = (
    COMPLEX_ROOT
    / (
        "complex_plan_diagnostic_dev_v1"
        ".before_manual_verification.jsonl"
    )
)

MANIFEST_BACKUP_PATH = (
    COMPLEX_ROOT
    / (
        "complex_plan_diagnostic_dev_v1_manifest"
        ".before_manual_verification.json"
    )
)

EXPECTED_CASE_IDS = [
    f"complex_{number:03d}"
    for number in range(31, 39)
]

EXPECTED_TARGETS = {
    ("complex_031", "q1"): (
        "fact_midea_group_2024_non_current_assets"
    ),
    ("complex_032", "q2"): (
        "fact_gree_electric_2024_"
        "cash_outflows_from_investing_activities_subtotal"
    ),
    ("complex_033", "q2"): (
        "fact_haier_smart_home_2024_taxes_and_surcharges"
    ),
    ("complex_034", "q2"): (
        "fact_hisense_home_2024_tax_refunds_received"
    ),
    ("complex_035", "q2"): (
        "fact_gree_electric_2024_"
        "net_profit_attributable_to_parent"
    ),
    ("complex_036", "q4"): (
        "fact_haier_smart_home_2024_"
        "total_comprehensive_income"
    ),
    ("complex_037", "q4"): (
        "fact_haier_smart_home_2024_"
        "non_current_liabilities"
    ),
    ("complex_038", "q2"): (
        "fact_midea_group_2024_"
        "net_cash_flow_from_financing_activities"
    ),
}

ALLOWED_CHANGED_FIELDS = {
    "validation_status",
    "validated_by",
    "validated_at",
    "updated_at",
    "review_notes",
}

CHINA_TIMEZONE = timezone(
    timedelta(hours=8)
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_raw_records(
    path: Path,
) -> list[dict[str, Any]]:
    records = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            raise SystemExit(
                f"blank_line={line_number}"
            )

        records.append(
            json.loads(line)
        )

    return records


def semantic_payload(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in ALLOWED_CHANGED_FIELDS
    }


def validate_target_queries(
    cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ],
) -> None:
    query_by_key = {
        (
            case.case_id,
            query.query_id,
        ): query
        for case in cases
        for query
        in case.gold_rewrite.retrieval_queries
    }

    if set(query_by_key).issuperset(
        EXPECTED_TARGETS
    ) is False:
        raise SystemExit(
            "diagnostic_target_query_missing=true"
        )

    for key, expected_fact_id in (
        EXPECTED_TARGETS.items()
    ):
        query = query_by_key[key]

        if (
            query.target_fact_id
            != expected_fact_id
        ):
            raise SystemExit(
                f"target_fact_changed={key}:"
                f"{query.target_fact_id}"
            )


def main() -> None:
    for required_path in (
        DATASET_PATH,
        MANIFEST_PATH,
        DEV_V2_PATH,
        TEST_V1_PATH,
    ):
        if not required_path.is_file():
            raise SystemExit(
                f"missing_file={required_path}"
            )

    dev_hash_before = sha256_file(
        DEV_V2_PATH
    )

    test_hash_before = sha256_file(
        TEST_V1_PATH
    )

    original_records = load_raw_records(
        DATASET_PATH
    )

    if [
        record["case_id"]
        for record in original_records
    ] != EXPECTED_CASE_IDS:
        raise SystemExit(
            "unexpected_case_ids=true"
        )

    original_by_case_id = {
        record["case_id"]: record
        for record in original_records
    }

    status_counts_before = Counter(
        record["validation_status"]
        for record in original_records
    )

    invalid_statuses = (
        set(status_counts_before)
        - {"pending", "verified"}
    )

    if invalid_statuses:
        raise SystemExit(
            "invalid_statuses="
            f"{sorted(invalid_statuses)}"
        )

    timestamp = datetime.now(
        CHINA_TIMEZONE
    ).isoformat()

    updated_records: list[
        dict[str, Any]
    ] = []

    updated_case_count = 0

    for original in original_records:
        updated = dict(original)

        if (
            original["validation_status"]
            == "pending"
        ):
            updated[
                "validation_status"
            ] = "verified"

            updated[
                "validated_by"
            ] = "manual_review"

            updated[
                "validated_at"
            ] = timestamp

            updated[
                "updated_at"
            ] = timestamp

            updated[
                "review_notes"
            ] = (
                "Diagnostic Dev v1：底层来源以及 "
                "Gold Rewrite、Gold Plan 和 Gold Answer "
                "均已完成人工语义核验。"
            )

            updated_case_count += 1

        if (
            semantic_payload(original)
            != semantic_payload(updated)
        ):
            raise SystemExit(
                "unexpected_semantic_change="
                f"{original['case_id']}"
            )

        updated_records.append(
            updated
        )

    updated_cases = tuple(
        ComplexFinancialEvalCase
        .model_validate(record)
        for record in updated_records
    )

    if [
        case.case_id
        for case in updated_cases
    ] != EXPECTED_CASE_IDS:
        raise SystemExit(
            "case_order_changed=true"
        )

    status_counts_after = Counter(
        case.validation_status.value
        for case in updated_cases
    )

    if status_counts_after != Counter(
        {"verified": 8}
    ):
        raise SystemExit(
            "unexpected_final_statuses="
            f"{dict(status_counts_after)}"
        )

    for case in updated_cases:
        if case.validated_by != "manual_review":
            raise SystemExit(
                "invalid_validated_by="
                f"{case.case_id}"
            )

        if case.validated_at is None:
            raise SystemExit(
                "missing_validated_at="
                f"{case.case_id}"
            )

    validate_target_queries(
        updated_cases
    )

    query_count = sum(
        len(
            case.gold_rewrite.retrieval_queries
        )
        for case in updated_cases
    )

    fact_reference_count = sum(
        len(case.gold_fact_ids)
        for case in updated_cases
    )

    evidence_reference_count = sum(
        len(case.gold_evidence_ids)
        for case in updated_cases
    )

    calculation_count = sum(
        len(case.gold_calculation_ids)
        for case in updated_cases
    )

    if query_count != 24:
        raise SystemExit(
            f"query_count_not_24={query_count}"
        )

    if fact_reference_count != 24:
        raise SystemExit(
            "fact_reference_count_not_24="
            f"{fact_reference_count}"
        )

    if evidence_reference_count != 24:
        raise SystemExit(
            "evidence_reference_count_not_24="
            f"{evidence_reference_count}"
        )

    if calculation_count != 0:
        raise SystemExit(
            "unexpected_calculation_count="
            f"{calculation_count}"
        )

    (
        registry_bundle,
        _page_mappings,
        _metric_aliases,
        fact_evidence_links,
    ) = load_registry_bundle(
        companies_path=(
            REGISTRY_ROOT / "companies.yaml"
        ),
        reports_path=(
            REGISTRY_ROOT / "reports.yaml"
        ),
        metrics_path=(
            REGISTRY_ROOT / "metrics.yaml"
        ),
        evidences_path=(
            REGISTRY_ROOT / "evidences.yaml"
        ),
        financial_facts_path=(
            REGISTRY_ROOT
            / "financial_facts.yaml"
        ),
    )

    if len(registry_bundle.evidences) != 78:
        raise SystemExit(
            "evidence_count_not_78="
            f"{len(registry_bundle.evidences)}"
        )

    if (
        len(
            registry_bundle.financial_facts
        )
        != 78
    ):
        raise SystemExit(
            "financial_fact_count_not_78="
            f"{len(registry_bundle.financial_facts)}"
        )

    if len(fact_evidence_links) != 78:
        raise SystemExit(
            "fact_evidence_link_count_not_78="
            f"{len(fact_evidence_links)}"
        )

    validate_complex_plan_eval_integrity(
        cases=updated_cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )

    output_text = "\n".join(
        case.model_dump_json()
        for case in updated_cases
    ) + "\n"

    original_text = DATASET_PATH.read_text(
        encoding="utf-8"
    )

    if output_text != original_text:
        if not DATASET_BACKUP_PATH.exists():
            shutil.copy2(
                DATASET_PATH,
                DATASET_BACKUP_PATH,
            )

        temporary_path = (
            DATASET_PATH.with_suffix(
                ".jsonl.tmp"
            )
        )

        temporary_path.write_text(
            output_text,
            encoding="utf-8",
        )

        temporary_cases = (
            load_complex_financial_eval_cases(
                temporary_path
            )
        )

        validate_complex_plan_eval_integrity(
            cases=temporary_cases,
            calculations=[],
            registry_bundle=registry_bundle,
        )

        temporary_path.replace(
            DATASET_PATH
        )

    reloaded_cases = (
        load_complex_financial_eval_cases(
            DATASET_PATH
        )
    )

    validate_complex_plan_eval_integrity(
        cases=reloaded_cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )

    final_status_counts = Counter(
        case.validation_status.value
        for case in reloaded_cases
    )

    if final_status_counts != Counter(
        {"verified": 8}
    ):
        raise SystemExit(
            "dataset_verification_not_persisted=true"
        )

    for case in reloaded_cases:
        original = original_by_case_id[
            case.case_id
        ]

        final_record = json.loads(
            case.model_dump_json()
        )

        if (
            semantic_payload(original)
            != semantic_payload(final_record)
        ):
            raise SystemExit(
                "persisted_semantic_change="
                f"{case.case_id}"
            )

    dataset_hash = sha256_file(
        DATASET_PATH
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if manifest["dataset_id"] != (
        "complex_plan_diagnostic_dev_v1"
    ):
        raise SystemExit(
            "unexpected_manifest_dataset_id="
            f"{manifest['dataset_id']}"
        )

    if manifest["case_count"] != 8:
        raise SystemExit(
            "manifest_case_count_not_8=true"
        )

    stored_dev_hash = manifest[
        "frozen_inputs"
    ][
        "complex_plan_dev_v2_sha256"
    ]

    stored_test_hash = manifest[
        "frozen_inputs"
    ][
        "complex_plan_test_v1_sha256"
    ]

    if stored_dev_hash != dev_hash_before:
        raise SystemExit(
            "stored_dev_v2_hash_mismatch=true"
        )

    if stored_test_hash != test_hash_before:
        raise SystemExit(
            "stored_test_v1_hash_mismatch=true"
        )

    if not MANIFEST_BACKUP_PATH.exists():
        shutil.copy2(
            MANIFEST_PATH,
            MANIFEST_BACKUP_PATH,
        )

    validation_times = [
        case.validated_at
        for case in reloaded_cases
        if case.validated_at is not None
    ]

    latest_validation_time = max(
        validation_times
    ).isoformat()

    manifest["status"] = "verified"
    manifest["status_counts"] = {
        "verified": 8,
    }
    manifest[
        "manual_semantic_review_required"
    ] = False
    manifest["validated_by"] = (
        "manual_review"
    )
    manifest["validated_at"] = (
        latest_validation_time
    )
    manifest["updated_at"] = (
        latest_validation_time
    )
    manifest["dataset_sha256"] = (
        dataset_hash
    )
    manifest["verification"] = {
        "case_count": 8,
        "verified_case_count": 8,
        "query_count": 24,
        "fact_reference_count": 24,
        "evidence_reference_count": 24,
        "calculation_count": 0,
        "semantic_fields_preserved": True,
        "gold_plan_preserved": True,
        "gold_answer_preserved": True,
        "frozen_test_target_overlap": 0,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    dev_hash_after = sha256_file(
        DEV_V2_PATH
    )

    test_hash_after = sha256_file(
        TEST_V1_PATH
    )

    if dev_hash_before != dev_hash_after:
        raise SystemExit(
            "complex_plan_dev_v2_modified=true"
        )

    if test_hash_before != test_hash_after:
        raise SystemExit(
            "complex_plan_test_v1_modified=true"
        )

    print("-" * 80)
    print(f"dataset_path={DATASET_PATH}")
    print(f"manifest_path={MANIFEST_PATH}")
    print(
        f"status_counts_before="
        f"{dict(status_counts_before)}"
    )
    print(
        f"updated_case_count="
        f"{updated_case_count}"
    )
    print(
        f"status_counts_after="
        f"{dict(final_status_counts)}"
    )
    print(f"case_count={len(reloaded_cases)}")
    print(f"query_count={query_count}")
    print(
        "fact_reference_count="
        f"{fact_reference_count}"
    )
    print(
        "evidence_reference_count="
        f"{evidence_reference_count}"
    )
    print(
        f"calculation_count={calculation_count}"
    )
    print(
        f"validated_by=manual_review"
    )
    print(
        f"validated_at={latest_validation_time}"
    )
    print(
        "semantic_fields_preserved=true"
    )
    print("gold_plan_preserved=true")
    print("gold_answer_preserved=true")
    print("dev_v2_preserved=true")
    print("test_v1_preserved=true")
    print(
        "diagnostic_dev_verification_passed=true"
    )


if __name__ == "__main__":
    main()