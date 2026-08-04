from pathlib import Path

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalTrace,
)
from app.schemas.enums import (
    ValidationStatus,
)
from app.services.complex_oracle_answer_generator_v2 import (
    ComplexOracleAnswerGeneratorV2,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_oracle import (
    execute_gold_oracle_case,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_dev_v1.jsonl"
)


@pytest.fixture(scope="module")
def bundle():
    registry_bundle, _, _, _ = (
        load_registry_bundle(
            companies_path=(
                REGISTRY_ROOT
                / "companies.yaml"
            ),
            reports_path=(
                REGISTRY_ROOT
                / "reports.yaml"
            ),
            metrics_path=(
                REGISTRY_ROOT
                / "metrics.yaml"
            ),
            evidences_path=(
                REGISTRY_ROOT
                / "evidences.yaml"
            ),
            financial_facts_path=(
                REGISTRY_ROOT
                / "financial_facts.yaml"
            ),
        )
    )

    return registry_bundle


@pytest.fixture(scope="module")
def cases_by_id():
    cases = (
        load_complex_financial_eval_cases(
            CASES_PATH
        )
    )

    return {
        case.case_id: case
        for case in cases
    }


class RegistryRetriever:
    def __init__(self, bundle):
        self.bundle = bundle

    @property
    def retriever_id(self) -> str:
        return "registry_answer_v2_test"

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        matches = [
            fact
            for fact
            in self.bundle.financial_facts.find(
                company_id=query.company_id,
                report_id=query.report_id,
                metric_id=query.metric_id,
                fiscal_year=query.fiscal_year,
                statement_scope=(
                    query.statement_scope.value
                ),
            )
            if (
                fact.statement_type
                is query.statement_type
                and fact.validation_status
                is ValidationStatus.VERIFIED
            )
        ]

        assert len(matches) == 1

        fact = matches[0]

        return ComplexRetrievalTrace(
            query_id=query.query_id,
            status="completed",
            retrieved_fact_ids=(
                fact.fact_id,
            ),
            retrieved_evidence_ids=(
                fact.primary_evidence_id,
            ),
            retrieved_chunk_ids=(
                f"{query.report_id}_"
                f"answer_v2_{query.query_id}",
            ),
            top_k=top_k,
            latency_ms=0.0,
        )


EXPECTED_PHRASES = {
    "complex_005": (
        "资产总计更高",
    ),
    "complex_006": (
        "应收账款更高",
    ),
    "complex_007": (
        "销售费用更高",
    ),
    "complex_008": (
        "净流入",
        "净流出",
    ),
    "complex_010": (
        "毛利率",
        "经营活动现金流量净额与净利润比率",
    ),
    "complex_014": (
        "美的集团更高",
    ),
    "complex_015": (
        "海尔智家更高",
    ),
    "complex_016": (
        "营业收入",
        "经营活动产生的现金流量净额",
        "美的集团更高",
    ),
    "complex_019": (
        "毛利率从高到低排序",
    ),
    "complex_020": (
        "经营活动现金流量净额与净利润比率"
        "从高到低排序",
    ),
}


EXPECTED_RANKINGS = {
    "complex_019": (
        "格力电器",
        "海尔智家",
        "美的集团",
        "海信家电",
    ),
    "complex_020": (
        "美的集团",
        "海尔智家",
        "格力电器",
    ),
}


@pytest.mark.parametrize(
    "case_id",
    [
        f"complex_{number:03d}"
        for number in range(1, 21)
    ],
)
def test_registry_oracle_completes_all_cases(
    bundle,
    cases_by_id,
    case_id,
) -> None:
    case = cases_by_id[case_id]

    result = execute_gold_oracle_case(
        run_id=(
            "complex_run_answer_v2_"
            f"{case_id}"
        ),
        case=case,
        retriever=RegistryRetriever(
            bundle
        ),
        calculator=(
            ComplexOracleCalculatorAdapter(
                registry_bundle=bundle
            )
        ),
        generator=(
            ComplexOracleAnswerGeneratorV2(
                registry_bundle=bundle
            )
        ),
        top_k=5,
    )

    assert result.status == "completed"
    assert result.error_stage is None
    assert result.error_message is None
    assert result.answer is not None

    assert result.generator_id == (
        "deterministic_financial_"
        "answer_generator_v2"
    )

    assert (
        result.answer.supporting_fact_ids
        == case.gold_fact_ids
    )

    assert (
        result.answer.citation_evidence_ids
        == case.gold_evidence_ids
    )

    assert (
        result.answer
        .supporting_calculation_ids
        == case.gold_calculation_ids
    )

    # 每个Gold事实的实际数值都必须出现在答案中
    plan_actions = {
        step.action
        for step in case.gold_plan.steps
    }

    is_calculation_comparison_or_ranking = (
        bool(case.gold_calculation_ids)
        and bool(
            plan_actions.intersection(
                {
                    "compare",
                    "rank",
                }
            )
        )
    )

    # 普通事实题和计算汇总题需要展示原始事实值。
    # 计算结果比较/排名题允许正文只展示派生结果，
    # 原始输入通过 supporting_fact_ids 保持可追踪。
    if not (
        is_calculation_comparison_or_ranking
    ):
        for fact_id in case.gold_fact_ids:
            fact = (
                bundle
                .financial_facts
                .require(fact_id)
            )

            formatted_value = format(
                fact.normalized_value,
                ",f",
            )

            assert (
                formatted_value
                in result.answer.answer_text
            )

    # 只要执行了计算，所有计算结果都必须出现在答案中。
    for calculation_trace in (
        result.calculation_traces
    ):
        assert (
            calculation_trace.result_value
            is not None
        )

        formatted_result = format(
            calculation_trace.result_value,
            ",f",
        )

        assert (
            formatted_result
            in result.answer.answer_text
        )

    for expected_phrase in (
        EXPECTED_PHRASES.get(
            case_id,
            (),
        )
    ):
        assert (
            expected_phrase
            in result.answer.answer_text
        )

    expected_ranking = (
        EXPECTED_RANKINGS.get(case_id)
    )

    if expected_ranking is not None:
        positions = [
            result.answer.answer_text.index(
                company_name
            )
            for company_name
            in expected_ranking
        ]

        assert positions == sorted(positions)


def test_v2_uses_new_runtime_identity(
    bundle,
) -> None:
    generator = (
        ComplexOracleAnswerGeneratorV2(
            registry_bundle=bundle
        )
    )

    assert generator.generator_id == (
        "deterministic_financial_"
        "answer_generator_v2"
    )