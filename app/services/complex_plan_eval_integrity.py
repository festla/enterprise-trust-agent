from __future__ import annotations

from collections.abc import Sequence

from app.schemas.calculation import DerivedCalculation
from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
    GoldRetrievalQuery,
)
from app.schemas.enums import ValidationStatus
from app.services.registry import RegistryBundle


class ComplexPlanEvalIntegrityError(ValueError):
    """复杂评测数据跨文件引用不完整。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors

        message = (
            "复杂评测数据跨文件完整性检查失败：\n- "
            + "\n- ".join(errors)
        )

        super().__init__(message)


def validate_complex_plan_eval_integrity(
    *,
    cases: Sequence[ComplexFinancialEvalCase],
    calculations: Sequence[DerivedCalculation],
    registry_bundle: RegistryBundle,
) -> None:
    """校验 Case、Calculation、Fact 和 Evidence 的引用关系。"""

    errors: list[str] = []

    calculations_by_id = _build_calculation_index(
        calculations,
        errors,
    )

    _validate_calculations_against_registry(
        calculations_by_id,
        registry_bundle,
        errors,
    )

    referenced_calculation_ids: set[str] = set()
    seen_case_ids: set[str] = set()

    for case in cases:
        if case.case_id in seen_case_ids:
            errors.append(
                f"case_id 重复：'{case.case_id}'"
            )
        else:
            seen_case_ids.add(case.case_id)

        _validate_case_scope(
            case,
            registry_bundle,
            errors,
        )

        _validate_case_facts(
            case,
            registry_bundle,
            errors,
        )

        _validate_case_evidences(
            case,
            registry_bundle,
            errors,
        )

        _validate_case_queries(
            case,
            registry_bundle,
            errors,
        )

        _validate_case_calculations(
            case,
            calculations_by_id,
            referenced_calculation_ids,
            errors,
        )

    unreferenced_calculation_ids = (
        set(calculations_by_id)
        - referenced_calculation_ids
    )

    for calculation_id in sorted(
        unreferenced_calculation_ids
    ):
        errors.append(
            "DerivedCalculation "
            f"'{calculation_id}' "
            "没有被任何 Case 引用"
        )

    if errors:
        raise ComplexPlanEvalIntegrityError(
            errors
        )


def _build_calculation_index(
    calculations: Sequence[DerivedCalculation],
    errors: list[str],
) -> dict[str, DerivedCalculation]:
    calculations_by_id: dict[
        str,
        DerivedCalculation,
    ] = {}

    for calculation in calculations:
        calculation_id = (
            calculation.calculation_id
        )

        if calculation_id in calculations_by_id:
            errors.append(
                "calculation_id 重复："
                f"'{calculation_id}'"
            )
            continue

        calculations_by_id[
            calculation_id
        ] = calculation

    return calculations_by_id


def _validate_calculations_against_registry(
    calculations_by_id: dict[
        str,
        DerivedCalculation,
    ],
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    for calculation in calculations_by_id.values():
        calculation_id = (
            calculation.calculation_id
        )

        if calculation.validation_status != "verified":
            errors.append(
                "DerivedCalculation "
                f"'{calculation_id}' "
                "不是 verified"
            )

        if (
            len(calculation.input_fact_ids)
            != len(set(calculation.input_fact_ids))
        ):
            errors.append(
                "DerivedCalculation "
                f"'{calculation_id}' "
                "的 input_fact_ids 包含重复 ID"
            )

        metric = registry_bundle.metrics.get(
            calculation.metric_id
        )

        if metric is None:
            errors.append(
                "DerivedCalculation "
                f"'{calculation_id}' "
                "引用了不存在的 FinancialMetric "
                f"'{calculation.metric_id}'"
            )
        else:
            if metric.formula_id != calculation.formula_id:
                errors.append(
                    "DerivedCalculation "
                    f"'{calculation_id}' 的 formula_id "
                    "与 FinancialMetric.formula_id "
                    "不一致"
                )

            if (
                metric.default_unit.value
                != calculation.result_unit
            ):
                errors.append(
                    "DerivedCalculation "
                    f"'{calculation_id}' 的 result_unit "
                    "与 FinancialMetric.default_unit "
                    "不一致"
                )

        for fact_id in calculation.input_fact_ids:
            fact = (
                registry_bundle
                .financial_facts
                .get(fact_id)
            )

            if fact is None:
                errors.append(
                    "DerivedCalculation "
                    f"'{calculation_id}' "
                    "引用了不存在的 FinancialFact "
                    f"'{fact_id}'"
                )
                continue

            if (
                fact.validation_status
                is not ValidationStatus.VERIFIED
            ):
                errors.append(
                    "DerivedCalculation "
                    f"'{calculation_id}' 的输入事实 "
                    f"'{fact_id}' 不是 verified"
                )


def _validate_case_scope(
    case: ComplexFinancialEvalCase,
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    for company_id in case.company_ids:
        if not registry_bundle.companies.contains(
            company_id
        ):
            errors.append(
                f"Case '{case.case_id}' "
                "引用了不存在的 Company "
                f"'{company_id}'"
            )

    for report_id in case.report_ids:
        report = registry_bundle.reports.get(
            report_id
        )

        if report is None:
            errors.append(
                f"Case '{case.case_id}' "
                "引用了不存在的 Report "
                f"'{report_id}'"
            )
            continue

        if report.company_id not in case.company_ids:
            errors.append(
                f"Case '{case.case_id}' 中 Report "
                f"'{report_id}' 的 company_id "
                "未列入 company_ids"
            )

        if report.fiscal_year not in case.fiscal_years:
            errors.append(
                f"Case '{case.case_id}' 中 Report "
                f"'{report_id}' 的 fiscal_year "
                "未列入 fiscal_years"
            )


def _validate_case_facts(
    case: ComplexFinancialEvalCase,
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    for fact_id in case.gold_fact_ids:
        fact = (
            registry_bundle
            .financial_facts
            .get(fact_id)
        )

        if fact is None:
            errors.append(
                f"Case '{case.case_id}' "
                "引用了不存在的 FinancialFact "
                f"'{fact_id}'"
            )
            continue

        if fact.company_id not in case.company_ids:
            errors.append(
                f"Case '{case.case_id}' 中 Fact "
                f"'{fact_id}' 的 company_id "
                "未列入 company_ids"
            )

        if fact.report_id not in case.report_ids:
            errors.append(
                f"Case '{case.case_id}' 中 Fact "
                f"'{fact_id}' 的 report_id "
                "未列入 report_ids"
            )

        if fact.fiscal_year not in case.fiscal_years:
            errors.append(
                f"Case '{case.case_id}' 中 Fact "
                f"'{fact_id}' 的 fiscal_year "
                "未列入 fiscal_years"
            )

        if (
            fact.validation_status
            is not ValidationStatus.VERIFIED
        ):
            errors.append(
                f"Case '{case.case_id}' 中 Fact "
                f"'{fact_id}' 不是 verified"
            )

        if (
            fact.primary_evidence_id
            not in case.gold_evidence_ids
        ):
            errors.append(
                f"Case '{case.case_id}' 中 Fact "
                f"'{fact_id}' 的 primary Evidence "
                f"'{fact.primary_evidence_id}' "
                "未列入 gold_evidence_ids"
            )


def _validate_case_evidences(
    case: ComplexFinancialEvalCase,
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    for evidence_id in case.gold_evidence_ids:
        evidence = (
            registry_bundle
            .evidences
            .get(evidence_id)
        )

        if evidence is None:
            errors.append(
                f"Case '{case.case_id}' "
                "引用了不存在的 SourceEvidence "
                f"'{evidence_id}'"
            )
            continue

        if evidence.report_id not in case.report_ids:
            errors.append(
                f"Case '{case.case_id}' 中 Evidence "
                f"'{evidence_id}' 的 report_id "
                "未列入 report_ids"
            )

        if (
            evidence.validation_status
            is not ValidationStatus.VERIFIED
        ):
            errors.append(
                f"Case '{case.case_id}' 中 Evidence "
                f"'{evidence_id}' 不是 verified"
            )


def _validate_case_queries(
    case: ComplexFinancialEvalCase,
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    for query in (
        case.gold_rewrite.retrieval_queries
    ):
        _validate_query(
            case,
            query,
            registry_bundle,
            errors,
        )


def _validate_query(
    case: ComplexFinancialEvalCase,
    query: GoldRetrievalQuery,
    registry_bundle: RegistryBundle,
    errors: list[str],
) -> None:
    fact = (
        registry_bundle
        .financial_facts
        .get(query.target_fact_id)
    )

    if fact is None:
        errors.append(
            f"Case '{case.case_id}' Query "
            f"'{query.query_id}' 的 target_fact_id "
            "不存在："
            f"'{query.target_fact_id}'"
        )
        return

    comparisons = (
        (
            "company_id",
            query.company_id,
            fact.company_id,
        ),
        (
            "report_id",
            query.report_id,
            fact.report_id,
        ),
        (
            "metric_id",
            query.metric_id,
            fact.metric_id,
        ),
        (
            "fiscal_year",
            query.fiscal_year,
            fact.fiscal_year,
        ),
        (
            "statement_type",
            query.statement_type,
            fact.statement_type,
        ),
        (
            "statement_scope",
            query.statement_scope,
            fact.statement_scope,
        ),
    )

    for field_name, query_value, fact_value in comparisons:
        if query_value != fact_value:
            errors.append(
                f"Case '{case.case_id}' Query "
                f"'{query.query_id}' 的 "
                f"{field_name} 与目标 "
                "FinancialFact 不一致"
            )

    report = registry_bundle.reports.get(
        query.report_id
    )

    if (
        report is not None
        and query.report_type
        is not report.report_type
    ):
        errors.append(
            f"Case '{case.case_id}' Query "
            f"'{query.query_id}' 的 report_type "
            "与 Report 不一致"
        )

    evidence = registry_bundle.evidences.get(
        fact.primary_evidence_id
    )

    if evidence is None:
        return

    if evidence.pdf_page not in query.gold_pdf_pages:
        errors.append(
            f"Case '{case.case_id}' Query "
            f"'{query.query_id}' 的 gold_pdf_pages "
            "未包含 primary Evidence 页码 "
            f"{evidence.pdf_page}"
        )

    if (
        evidence.statement_type is not None
        and evidence.statement_type
        is not fact.statement_type
    ):
        errors.append(
            f"Case '{case.case_id}' Query "
            f"'{query.query_id}' 的 primary Evidence "
            "与目标 Fact 的 statement_type 不一致"
        )

    if (
        evidence.statement_scope is not None
        and evidence.statement_scope
        is not fact.statement_scope
    ):
        errors.append(
            f"Case '{case.case_id}' Query "
            f"'{query.query_id}' 的 primary Evidence "
            "与目标 Fact 的 statement_scope 不一致"
        )


def _validate_case_calculations(
    case: ComplexFinancialEvalCase,
    calculations_by_id: dict[
        str,
        DerivedCalculation,
    ],
    referenced_calculation_ids: set[str],
    errors: list[str],
) -> None:
    for calculation_id in case.gold_calculation_ids:
        referenced_calculation_ids.add(
            calculation_id
        )

        calculation = calculations_by_id.get(
            calculation_id
        )

        if calculation is None:
            errors.append(
                f"Case '{case.case_id}' "
                "引用了不存在的 DerivedCalculation "
                f"'{calculation_id}'"
            )
            continue

        unexpected_input_fact_ids = (
            set(calculation.input_fact_ids)
            - set(case.gold_fact_ids)
        )

        if unexpected_input_fact_ids:
            errors.append(
                f"Case '{case.case_id}' 的 Calculation "
                f"'{calculation_id}' 使用了未列入 "
                "gold_fact_ids 的输入事实："
                f"{sorted(unexpected_input_fact_ids)}"
            )

        calculation_steps = [
            step
            for step in case.gold_plan.steps
            if step.calculation_id
            == calculation_id
        ]

        if len(calculation_steps) != 1:
            errors.append(
                f"Case '{case.case_id}' 的 Calculation "
                f"'{calculation_id}' 必须对应且只能对应 "
                "一个 Gold Plan 计算步骤"
            )
            continue

        step = calculation_steps[0]

        if step.formula_id != calculation.formula_id:
            errors.append(
                f"Case '{case.case_id}' 的 Calculation "
                f"'{calculation_id}' 与 Gold Plan "
                "计算步骤的 formula_id 不一致"
            )

        if (
            tuple(step.input_refs)
            != tuple(calculation.input_fact_ids)
        ):
            errors.append(
                f"Case '{case.case_id}' 的 Calculation "
                f"'{calculation_id}'：Gold Plan 的 "
                "input_refs 与 "
                "DerivedCalculation.input_fact_ids "
                "顺序不一致"
            )