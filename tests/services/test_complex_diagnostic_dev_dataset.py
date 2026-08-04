from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    validate_complex_plan_eval_integrity,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

EXPECTED_LAYOUT = {
    "complex_031": (
        "single_company_multi_metric",
        "medium",
        2,
    ),
    "complex_032": (
        "single_company_multi_metric",
        "medium",
        2,
    ),
    "complex_033": (
        "single_company_multi_metric",
        "medium",
        2,
    ),
    "complex_034": (
        "single_company_multi_metric",
        "hard",
        4,
    ),
    "complex_035": (
        "single_company_multi_metric",
        "medium",
        2,
    ),
    "complex_036": (
        "cross_company_comparison",
        "hard",
        4,
    ),
    "complex_037": (
        "single_company_multi_metric",
        "hard",
        4,
    ),
    "complex_038": (
        "cross_company_comparison",
        "hard",
        4,
    ),
}

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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_diagnostic_dev_manifest() -> None:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == 1
    assert manifest["dataset_id"] == (
        "complex_plan_diagnostic_dev_v1"
    )
    assert manifest["split"] == (
        "diagnostic_dev"
    )
    assert manifest["status"] == "verified"
    assert manifest["case_count"] == 8
    assert manifest["query_count"] == 24
    assert manifest[
        "diagnostic_target_count"
    ] == 8
    assert manifest["status_counts"] == {
        "verified": 8,
    }
    assert manifest[
        "manual_semantic_review_required"
    ] is False
    assert manifest["validated_by"] == (
        "manual_review"
    )
    assert manifest["validated_at"]
    assert manifest["dataset_sha256"] == (
        sha256_file(DATASET_PATH)
    )

    verification = manifest[
        "verification"
    ]

    assert verification[
        "verified_case_count"
    ] == 8
    assert verification[
        "semantic_fields_preserved"
    ] is True
    assert verification[
        "gold_plan_preserved"
    ] is True
    assert verification[
        "gold_answer_preserved"
    ] is True


def test_diagnostic_dev_layout() -> None:
    cases = (
        load_complex_financial_eval_cases(
            DATASET_PATH
        )
    )

    assert [
        case.case_id
        for case in cases
    ] == list(EXPECTED_LAYOUT)

    fact_reference_count = 0
    query_count = 0
    evidence_reference_count = 0

    all_fact_ids: set[str] = set()

    for case in cases:
        (
            expected_type,
            expected_difficulty,
            expected_fact_count,
        ) = EXPECTED_LAYOUT[
            case.case_id
        ]

        assert case.question_type == (
            expected_type
        )
        assert case.difficulty == (
            expected_difficulty
        )
        assert len(case.gold_fact_ids) == (
            expected_fact_count
        )
        assert len(
            case.gold_rewrite.retrieval_queries
        ) == expected_fact_count
        assert len(
            case.gold_evidence_ids
        ) == expected_fact_count
        assert not case.gold_calculation_ids

        assert (
            case.validation_status.value
            == "verified"
        )
        assert case.validated_by == (
            "manual_review"
        )
        assert case.validated_at is not None
        assert case.source_version == (
            "complex_plan_diagnostic_dev_v1"
        )

        assert set(
            case.gold_answer.supporting_fact_ids
        ) == set(case.gold_fact_ids)

        assert set(
            case.gold_answer.evidence_ids
        ) == set(case.gold_evidence_ids)

        assert not (
            case.gold_answer
            .supporting_calculation_ids
        )

        fact_reference_count += len(
            case.gold_fact_ids
        )
        query_count += len(
            case.gold_rewrite.retrieval_queries
        )
        evidence_reference_count += len(
            case.gold_evidence_ids
        )
        all_fact_ids.update(
            case.gold_fact_ids
        )

    assert fact_reference_count == 24
    assert query_count == 24
    assert evidence_reference_count == 24
    assert len(all_fact_ids) == 23


def test_diagnostic_dev_integrity() -> None:
    cases = (
        load_complex_financial_eval_cases(
            DATASET_PATH
        )
    )

    (
        registry_bundle,
        _,
        _,
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

    assert len(
        registry_bundle.evidences
    ) == 78

    assert len(
        registry_bundle.financial_facts
    ) == 78

    assert len(
        fact_evidence_links
    ) == 78

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )


def test_diagnostic_target_contract() -> None:
    cases = (
        load_complex_financial_eval_cases(
            DATASET_PATH
        )
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    query_by_key = {
        (
            case.case_id,
            query.query_id,
        ): query
        for case in cases
        for query
        in case.gold_rewrite.retrieval_queries
    }

    manifest_targets = {
        (
            target["case_id"],
            target["query_id"],
        ): target
        for target in manifest[
            "diagnostic_targets"
        ]
    }

    assert set(manifest_targets) == set(
        EXPECTED_TARGETS
    )

    for key, expected_fact_id in (
        EXPECTED_TARGETS.items()
    ):
        query = query_by_key[key]
        target = manifest_targets[key]

        assert query.target_fact_id == (
            expected_fact_id
        )

        assert target[
            "target_fact_id"
        ] == expected_fact_id

        assert target[
            "target_evidence_id"
        ] == expected_fact_id.replace(
            "fact_",
            "evidence_",
            1,
        )

        assert target[
            "semantic_query"
        ] == query.semantic_query

        assert target[
            "diagnostic_category"
        ]


def test_diagnostic_dev_isolation() -> None:
    diagnostic_cases = (
        load_complex_financial_eval_cases(
            DATASET_PATH
        )
    )

    dev_cases = (
        load_complex_financial_eval_cases(
            DEV_V2_PATH
        )
    )

    test_cases = (
        load_complex_financial_eval_cases(
            TEST_V1_PATH
        )
    )

    diagnostic_case_ids = {
        case.case_id
        for case in diagnostic_cases
    }

    existing_case_ids = {
        case.case_id
        for case in dev_cases + test_cases
    }

    assert not (
        diagnostic_case_ids
        & existing_case_ids
    )

    diagnostic_questions = {
        case.question
        for case in diagnostic_cases
    }

    existing_questions = {
        case.question
        for case in dev_cases + test_cases
    }

    assert not (
        diagnostic_questions
        & existing_questions
    )

    target_fact_ids = set(
        EXPECTED_TARGETS.values()
    )

    target_evidence_ids = {
        fact_id.replace(
            "fact_",
            "evidence_",
            1,
        )
        for fact_id in target_fact_ids
    }

    frozen_test_fact_ids = {
        fact_id
        for case in test_cases
        for fact_id in case.gold_fact_ids
    }

    frozen_test_evidence_ids = {
        evidence_id
        for case in test_cases
        for evidence_id
        in case.gold_evidence_ids
    }

    assert not (
        target_fact_ids
        & frozen_test_fact_ids
    )

    assert not (
        target_evidence_ids
        & frozen_test_evidence_ids
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    frozen_inputs = manifest[
        "frozen_inputs"
    ]

    assert frozen_inputs[
        "complex_plan_dev_v2_sha256"
    ] == sha256_file(DEV_V2_PATH)

    assert frozen_inputs[
        "complex_plan_test_v1_sha256"
    ] == sha256_file(TEST_V1_PATH)