from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexRetrievalTrace,
)
from app.schemas.enums import ValidationStatus
from app.services.complex_oracle_answer_generator import (
    ComplexOracleAnswerGenerator,
    ComplexOracleAnswerGeneratorError,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_oracle import (
    build_gold_oracle_plan,
    build_gold_oracle_rewrite,
    execute_gold_oracle_case,
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

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_pilot_v1.jsonl"
)

TEST_TIME = datetime(
    2026,
    8,
    2,
    18,
    0,
    tzinfo=timezone.utc,
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
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    return {
        case.case_id: case
        for case in cases
    }


def resolve_fact(
    bundle,
    query,
):
    matches = [
        fact
        for fact
        in bundle.financial_facts.find(
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

    return matches[0]


def build_runtime_context(
    bundle,
    case,
):
    rewrite = build_gold_oracle_rewrite(
        case
    )

    plan = build_gold_oracle_plan(
        case
    )

    traces = []

    for query in (
        rewrite.retrieval_queries
    ):
        fact = resolve_fact(
            bundle,
            query,
        )

        traces.append(
            ComplexRetrievalTrace(
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
                    f"test_{query.query_id}",
                ),
                top_k=5,
                latency_ms=1.0,
            )
        )

    return (
        rewrite,
        plan,
        tuple(traces),
    )


def generate_answer(
    *,
    bundle,
    case,
    calculation_traces=(),
):
    rewrite, plan, traces = (
        build_runtime_context(
            bundle,
            case,
        )
    )

    generator = (
        ComplexOracleAnswerGenerator(
            registry_bundle=bundle
        )
    )

    answer = generator.generate(
        question=case.question,
        rewrite=rewrite,
        plan=plan,
        retrieval_traces=traces,
        calculation_traces=(
            calculation_traces
        ),
    )

    return answer, traces


def test_generates_multi_metric_answer(
    bundle,
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_001"
    ]

    answer, traces = generate_answer(
        bundle=bundle,
        case=case,
    )

    assert answer.answer_text == (
        "美的集团2024年合并口径"
        "营业收入为407,149,600,000元，"
        "归属于母公司所有者的净利润"
        "为38,537,237,000元。"
    )

    assert answer.supporting_fact_ids == (
        "fact_midea_group_2024_revenue",
        "fact_midea_group_2024_"
        "net_profit_attributable_to_parent",
    )

    assert answer.citation_evidence_ids == (
        "evidence_midea_group_2024_revenue",
        "evidence_midea_group_2024_"
        "net_profit_attributable_to_parent",
    )

    assert len(traces) == 2


def test_generates_calculation_answer(
    bundle,
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_002"
    ]

    rewrite, plan, traces = (
        build_runtime_context(
            bundle,
            case,
        )
    )

    calculation_step = next(
        step
        for step in plan.steps
        if step.action == "calculate"
    )

    calculator = (
        ComplexOracleCalculatorAdapter(
            registry_bundle=bundle,
            clock=lambda: TEST_TIME,
        )
    )

    calculation_trace = (
        calculator.calculate(
            calculation_id=(
                calculation_step
                .calculation_id
            ),
            formula_id=(
                calculation_step.formula_id
            ),
            input_fact_ids=tuple(
                trace.retrieved_fact_ids[0]
                for trace in traces
            ),
        )
    )

    generator = (
        ComplexOracleAnswerGenerator(
            registry_bundle=bundle
        )
    )

    answer = generator.generate(
        question=case.question,
        rewrite=rewrite,
        plan=plan,
        retrieval_traces=traces,
        calculation_traces=(
            calculation_trace,
        ),
    )

    assert answer.answer_text == (
        "海信家电2024年合并口径"
        "营业收入为92,745,611,109.52元，"
        "营业成本为73,476,062,734.50元，"
        "毛利率为20.7768%。"
    )

    assert (
        answer.supporting_calculation_ids
        == (
            "calculation_hisense_home_"
            "2024_gross_profit_margin",
        )
    )


def test_generates_comparison_answer(
    bundle,
    cases_by_id,
) -> None:
    answer, _ = generate_answer(
        bundle=bundle,
        case=cases_by_id[
            "complex_003"
        ],
    )

    assert answer.answer_text == (
        "美的集团2024年合并口径"
        "经营活动产生的现金流量净额"
        "为60,511,572,000元，"
        "格力电器为29,369,250,570.66元；"
        "美的集团更高。"
    )


def test_generates_ranking_answer(
    bundle,
    cases_by_id,
) -> None:
    answer, _ = generate_answer(
        bundle=bundle,
        case=cases_by_id[
            "complex_004"
        ],
    )

    assert answer.answer_text == (
        "按2024年合并口径营业收入"
        "从高到低排序："
        "1. 美的集团407,149,600,000元；"
        "2. 海尔智家285,981,225,203.93元；"
        "3. 格力电器189,163,654,064.64元；"
        "4. 海信家电92,745,611,109.52元。"
    )


def test_failed_retrieval_trace_is_rejected(
    bundle,
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_001"
    ]

    rewrite, plan, traces = (
        build_runtime_context(
            bundle,
            case,
        )
    )

    failed_trace = (
        ComplexRetrievalTrace(
            query_id="q1",
            status="failed",
            retrieved_fact_ids=(),
            retrieved_evidence_ids=(),
            retrieved_chunk_ids=(),
            top_k=5,
            latency_ms=1.0,
            error_message="test failure",
        )
    )

    generator = (
        ComplexOracleAnswerGenerator(
            registry_bundle=bundle
        )
    )

    with pytest.raises(
        ComplexOracleAnswerGeneratorError,
        match="检索尚未完成",
    ):
        generator.generate(
            question=case.question,
            rewrite=rewrite,
            plan=plan,
            retrieval_traces=(
                failed_trace,
                traces[1],
            ),
            calculation_traces=(),
        )


def test_missing_primary_evidence_is_rejected(
    bundle,
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_001"
    ]

    rewrite, plan, traces = (
        build_runtime_context(
            bundle,
            case,
        )
    )

    invalid_trace = (
        traces[0].model_copy(
            update={
                "retrieved_evidence_ids": (
                    "evidence_unrelated",
                ),
            }
        )
    )

    generator = (
        ComplexOracleAnswerGenerator(
            registry_bundle=bundle
        )
    )

    with pytest.raises(
        ComplexOracleAnswerGeneratorError,
        match="没有检索到其主证据",
    ):
        generator.generate(
            question=case.question,
            rewrite=rewrite,
            plan=plan,
            retrieval_traces=(
                invalid_trace,
                traces[1],
            ),
            calculation_traces=(),
        )


def test_empty_generator_id_is_rejected(
    bundle,
) -> None:
    with pytest.raises(
        ComplexOracleAnswerGeneratorError,
        match="generator_id 不能为空",
    ):
        ComplexOracleAnswerGenerator(
            registry_bundle=bundle,
            generator_id=" ",
        )


class RegistryTestRetriever:
    def __init__(self, bundle):
        self.bundle = bundle

    @property
    def retriever_id(self) -> str:
        return "registry_test_retriever_v1"

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        fact = resolve_fact(
            self.bundle,
            query,
        )

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
                f"integration_{query.query_id}",
            ),
            top_k=top_k,
            latency_ms=1.0,
        )


def test_generator_integrates_with_executor(
    bundle,
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_002"
    ]

    result = execute_gold_oracle_case(
        run_id=(
            "complex_run_generator_integration"
        ),
        case=case,
        retriever=(
            RegistryTestRetriever(bundle)
        ),
        calculator=(
            ComplexOracleCalculatorAdapter(
                registry_bundle=bundle,
                clock=lambda: TEST_TIME,
            )
        ),
        generator=(
            ComplexOracleAnswerGenerator(
                registry_bundle=bundle
            )
        ),
        top_k=5,
    )

    assert result.status == "completed"
    assert result.answer is not None

    assert result.answer.answer_text == (
        "海信家电2024年合并口径"
        "营业收入为92,745,611,109.52元，"
        "营业成本为73,476,062,734.50元，"
        "毛利率为20.7768%。"
    )

    assert result.generator_id == (
        "deterministic_financial_"
        "answer_generator_v1"
    )