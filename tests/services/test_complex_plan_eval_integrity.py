from pathlib import Path

import pytest

from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    ComplexPlanEvalIntegrityError,
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

CASES_PATH = (
    COMPLEX_PLAN_ROOT
    / "complex_plan_pilot_v1.jsonl"
)

CALCULATIONS_PATH = (
    COMPLEX_PLAN_ROOT
    / "gold_calculations_pilot_v1.jsonl"
)


@pytest.fixture
def valid_inputs():
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    calculations = load_derived_calculations(
        CALCULATIONS_PATH
    )

    bundle, _, _, _ = load_registry_bundle(
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

    return cases, calculations, bundle


def replace_case(
    cases,
    index,
    replacement,
):
    return (
        *cases[:index],
        replacement,
        *cases[index + 1:],
    )


def test_valid_pilot_dataset_passes(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=calculations,
        registry_bundle=bundle,
    )


def test_rejects_missing_gold_fact(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    case = cases[0]

    invalid_case = case.model_copy(
        update={
            "gold_fact_ids": (
                case.gold_fact_ids[0],
                "fact_missing",
            ),
        }
    )

    invalid_cases = replace_case(
        cases,
        0,
        invalid_case,
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=invalid_cases,
            calculations=calculations,
            registry_bundle=bundle,
        )

    assert any(
        "不存在的 FinancialFact 'fact_missing'"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_query_metric_mismatch(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    case = cases[0]
    queries = (
        case.gold_rewrite.retrieval_queries
    )

    invalid_query = queries[0].model_copy(
        update={
            "metric_id": "operating_cost",
        }
    )

    invalid_rewrite = (
        case.gold_rewrite.model_copy(
            update={
                "retrieval_queries": (
                    invalid_query,
                    *queries[1:],
                ),
            }
        )
    )

    invalid_case = case.model_copy(
        update={
            "gold_rewrite": invalid_rewrite,
        }
    )

    invalid_cases = replace_case(
        cases,
        0,
        invalid_case,
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=invalid_cases,
            calculations=calculations,
            registry_bundle=bundle,
        )

    assert any(
        "metric_id 与目标 FinancialFact 不一致"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_missing_primary_evidence(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    case = cases[0]

    invalid_case = case.model_copy(
        update={
            "gold_evidence_ids": (
                case.gold_evidence_ids[1],
            ),
        }
    )

    invalid_cases = replace_case(
        cases,
        0,
        invalid_case,
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=invalid_cases,
            calculations=calculations,
            registry_bundle=bundle,
        )

    assert any(
        "未列入 gold_evidence_ids"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_wrong_gold_pdf_page(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    case = cases[0]
    queries = (
        case.gold_rewrite.retrieval_queries
    )

    invalid_query = queries[0].model_copy(
        update={
            "gold_pdf_pages": (999,),
        }
    )

    invalid_rewrite = (
        case.gold_rewrite.model_copy(
            update={
                "retrieval_queries": (
                    invalid_query,
                    *queries[1:],
                ),
            }
        )
    )

    invalid_case = case.model_copy(
        update={
            "gold_rewrite": invalid_rewrite,
        }
    )

    invalid_cases = replace_case(
        cases,
        0,
        invalid_case,
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=invalid_cases,
            calculations=calculations,
            registry_bundle=bundle,
        )

    assert any(
        "未包含 primary Evidence 页码"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_missing_calculation(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    case = cases[1]

    invalid_case = case.model_copy(
        update={
            "gold_calculation_ids": (
                "calculation_missing",
            ),
        }
    )

    invalid_cases = replace_case(
        cases,
        1,
        invalid_case,
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=invalid_cases,
            calculations=calculations,
            registry_bundle=bundle,
        )

    assert any(
        "不存在的 DerivedCalculation "
        "'calculation_missing'"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_calculation_input_order_mismatch(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    calculation = calculations[0]

    invalid_calculation = calculation.model_copy(
        update={
            "input_fact_ids": list(
                reversed(
                    calculation.input_fact_ids
                )
            ),
        }
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=cases,
            calculations=(
                invalid_calculation,
            ),
            registry_bundle=bundle,
        )

    assert any(
        "input_refs 与 "
        "DerivedCalculation.input_fact_ids "
        "顺序不一致"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_formula_mismatch(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    invalid_calculation = (
        calculations[0].model_copy(
            update={
                "formula_id": "wrong_formula",
            }
        )
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=cases,
            calculations=(
                invalid_calculation,
            ),
            registry_bundle=bundle,
        )

    assert any(
        "formula_id 与 "
        "FinancialMetric.formula_id 不一致"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_unverified_calculation(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    invalid_calculation = (
        calculations[0].model_copy(
            update={
                "validation_status": "pending",
            }
        )
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=cases,
            calculations=(
                invalid_calculation,
            ),
            registry_bundle=bundle,
        )

    assert any(
        "不是 verified"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_orphan_calculation(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    orphan_calculation = (
        calculations[0].model_copy(
            update={
                "calculation_id": (
                    "calculation_orphan"
                ),
            }
        )
    )

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=cases,
            calculations=(
                *calculations,
                orphan_calculation,
            ),
            registry_bundle=bundle,
        )

    assert any(
        "DerivedCalculation "
        "'calculation_orphan' "
        "没有被任何 Case 引用"
        in error
        for error in exc_info.value.errors
    )


def test_rejects_duplicate_calculation_id(
    valid_inputs,
) -> None:
    cases, calculations, bundle = valid_inputs

    with pytest.raises(
        ComplexPlanEvalIntegrityError
    ) as exc_info:
        validate_complex_plan_eval_integrity(
            cases=cases,
            calculations=(
                calculations[0],
                calculations[0],
            ),
            registry_bundle=bundle,
        )

    assert any(
        "calculation_id 重复"
        in error
        for error in exc_info.value.errors
    )