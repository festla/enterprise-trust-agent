from collections import Counter
from decimal import Decimal
from pathlib import Path
import hashlib
import json

from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    validate_complex_plan_eval_integrity,
)
from app.services.derived_calculation_dataset import (
    load_derived_calculations,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

COMPLEX_PLAN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
)

TEST_CASE_PATH = (
    COMPLEX_PLAN_ROOT
    / "complex_plan_test_v1.jsonl"
)

TEST_CALCULATION_PATH = (
    COMPLEX_PLAN_ROOT
    / "gold_calculations_test_v1.jsonl"
)

TEST_MANIFEST_PATH = (
    COMPLEX_PLAN_ROOT
    / "complex_plan_test_v1_manifest.json"
)

DEV_CASE_PATH = (
    COMPLEX_PLAN_ROOT
    / "complex_plan_dev_v2.jsonl"
)


EXPECTED_CASE_SHA256 = (
    "ea65ac71fd9cdee06db1c9eef0a733e061fd828869775a6b17f90231307c1049"
)

EXPECTED_CALCULATION_SHA256 = (
    "36217aa94b2da6792d41ac09db24999c12427d3913c59262c9cdb6500811228d"
)


EXPECTED_LAYOUT = {
    "complex_021": (
        "single_company_multi_metric",
        "medium",
        3,
        0,
    ),
    "complex_022": (
        "single_company_multi_metric",
        "medium",
        3,
        0,
    ),
    "complex_023": (
        "single_company_multi_metric",
        "medium",
        3,
        0,
    ),
    "complex_024": (
        "single_company_multi_metric",
        "hard",
        4,
        2,
    ),
    "complex_025": (
        "cross_company_comparison",
        "medium",
        2,
        0,
    ),
    "complex_026": (
        "cross_company_comparison",
        "medium",
        2,
        0,
    ),
    "complex_027": (
        "cross_company_comparison",
        "hard",
        4,
        2,
    ),
    "complex_028": (
        "multi_company_ranking",
        "medium",
        3,
        0,
    ),
    "complex_029": (
        "multi_company_ranking",
        "hard",
        4,
        0,
    ),
    "complex_030": (
        "multi_company_ranking",
        "hard",
        6,
        3,
    ),
}


EXPECTED_CALCULATION_RESULTS = {
    (
        "calculation_hisense_home_2024_"
        "current_ratio"
    ): ("1.0958", "ratio"),
    (
        "calculation_hisense_home_2024_"
        "debt_to_equity_ratio"
    ): ("2.5976", "ratio"),
    (
        "calculation_midea_group_2024_"
        "effective_income_tax_rate"
    ): ("16.9899", "percent"),
    (
        "calculation_gree_electric_2024_"
        "effective_income_tax_rate"
    ): ("12.2640", "percent"),
    (
        "calculation_midea_group_2024_"
        "current_ratio"
    ): ("1.1059", "ratio"),
    (
        "calculation_gree_electric_2024_"
        "current_ratio"
    ): ("1.1177", "ratio"),
    (
        "calculation_haier_smart_home_2024_"
        "current_ratio"
    ): ("1.0142", "ratio"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_complex_plan_test_v1_is_frozen() -> None:
    manifest = json.loads(
        TEST_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == 1
    assert manifest["dataset_id"] == (
        "complex_plan_test_v1"
    )
    assert manifest["split"] == "test"
    assert manifest["status"] == "frozen"

    assert (
        manifest["case_file"]["sha256"]
        == EXPECTED_CASE_SHA256
    )

    assert (
        manifest["calculation_file"]["sha256"]
        == EXPECTED_CALCULATION_SHA256
    )

    assert sha256_file(
        TEST_CASE_PATH
    ) == EXPECTED_CASE_SHA256

    assert sha256_file(
        TEST_CALCULATION_PATH
    ) == EXPECTED_CALCULATION_SHA256

    policy = manifest[
        "evaluation_policy"
    ]

    assert (
        policy[
            "allow_post_test_query_tuning"
        ]
        is False
    )

    assert (
        policy[
            "allow_post_test_parameter_tuning"
        ]
        is False
    )

    assert (
        policy[
            "allow_post_test_gold_edit"
        ]
        is False
    )


def test_complex_plan_test_v1_layout() -> None:
    cases = load_complex_financial_eval_cases(
        TEST_CASE_PATH
    )

    assert len(cases) == 10

    assert [
        case.case_id
        for case in cases
    ] == list(EXPECTED_LAYOUT)

    total_fact_count = 0
    total_query_count = 0
    total_evidence_count = 0
    total_calculation_count = 0

    for case in cases:
        (
            expected_type,
            expected_difficulty,
            expected_fact_count,
            expected_calculation_count,
        ) = EXPECTED_LAYOUT[case.case_id]

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

        assert len(
            case.gold_calculation_ids
        ) == expected_calculation_count

        assert (
            case.validation_status.value
            == "verified"
        )

        assert case.validated_by == (
            "manual_review"
        )

        assert case.validated_at is not None

        assert case.source_version == (
            "complex_plan_test_v1"
        )

        assert set(
            case.gold_answer.supporting_fact_ids
        ) == set(case.gold_fact_ids)

        assert set(
            case.gold_answer.evidence_ids
        ) == set(case.gold_evidence_ids)

        assert set(
            case.gold_answer
            .supporting_calculation_ids
        ) == set(
            case.gold_calculation_ids
        )

        total_fact_count += len(
            case.gold_fact_ids
        )

        total_query_count += len(
            case.gold_rewrite.retrieval_queries
        )

        total_evidence_count += len(
            case.gold_evidence_ids
        )

        total_calculation_count += len(
            case.gold_calculation_ids
        )

    assert total_fact_count == 34
    assert total_query_count == 34
    assert total_evidence_count == 34
    assert total_calculation_count == 7


def test_complex_plan_test_v1_integrity() -> None:
    cases = load_complex_financial_eval_cases(
        TEST_CASE_PATH
    )

    calculations = (
        load_derived_calculations(
            TEST_CALCULATION_PATH
        )
    )

    (
        bundle,
        _,
        _,
        _,
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

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=calculations,
        registry_bundle=bundle,
    )


def test_complex_plan_test_v1_has_no_dev_leakage(
) -> None:
    dev_cases = (
        load_complex_financial_eval_cases(
            DEV_CASE_PATH
        )
    )

    test_cases = (
        load_complex_financial_eval_cases(
            TEST_CASE_PATH
        )
    )

    dev_case_ids = {
        case.case_id
        for case in dev_cases
    }

    test_case_ids = {
        case.case_id
        for case in test_cases
    }

    dev_questions = {
        case.question
        for case in dev_cases
    }

    test_questions = {
        case.question
        for case in test_cases
    }

    dev_fact_ids = {
        fact_id
        for case in dev_cases
        for fact_id in case.gold_fact_ids
    }

    test_fact_ids = {
        fact_id
        for case in test_cases
        for fact_id in case.gold_fact_ids
    }

    assert not (
        dev_case_ids & test_case_ids
    )

    assert not (
        dev_questions & test_questions
    )

    assert not (
        dev_fact_ids & test_fact_ids
    )

    assert len(test_fact_ids) == 34


def test_complex_plan_test_v1_calculations(
) -> None:
    cases = load_complex_financial_eval_cases(
        TEST_CASE_PATH
    )

    calculations = (
        load_derived_calculations(
            TEST_CALCULATION_PATH
        )
    )

    assert len(calculations) == 7

    calculation_by_id = {
        calculation.calculation_id: calculation
        for calculation in calculations
    }

    assert set(calculation_by_id) == set(
        EXPECTED_CALCULATION_RESULTS
    )

    referenced_calculation_ids = [
        calculation_id
        for case in cases
        for calculation_id
        in case.gold_calculation_ids
    ]

    assert len(
        referenced_calculation_ids
    ) == 7

    assert set(
        referenced_calculation_ids
    ) == set(calculation_by_id)

    assert [
        case.case_id
        for case in cases
        if case.gold_calculation_ids
    ] == [
        "complex_024",
        "complex_027",
        "complex_030",
    ]

    for calculation_id, expected in (
        EXPECTED_CALCULATION_RESULTS.items()
    ):
        calculation = calculation_by_id[
            calculation_id
        ]

        expected_value, expected_unit = (
            expected
        )

        assert calculation.result_value == (
            Decimal(expected_value)
        )

        assert calculation.result_unit == (
            expected_unit
        )

        assert (
            calculation.validation_status
            == "verified"
        )

        assert calculation.validated_by == (
            "deterministic_calculator_v1"
        )
