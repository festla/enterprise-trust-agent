from pathlib import Path

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexCalculationTrace,
    ComplexFinalAnswerOutput,
    ComplexRetrievalTrace,
)
from app.schemas.enums import ValidationStatus
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_oracle import (
    build_gold_oracle_plan,
    build_gold_oracle_rewrite,
    execute_gold_oracle_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_pilot_v1.jsonl"
)


@pytest.fixture(scope="module")
def cases_by_id():
    cases = load_complex_financial_eval_cases(
        CASES_PATH
    )

    return {
        case.case_id: case
        for case in cases
    }


def build_retrieval_mapping(case):
    mapping = {}

    for query in (
        case.gold_rewrite.retrieval_queries
    ):
        fact_id = query.target_fact_id

        evidence_id = (
            "evidence_"
            + fact_id.removeprefix("fact_")
        )

        assert (
            evidence_id
            in case.gold_evidence_ids
        )

        mapping[query.query_id] = (
            fact_id,
            evidence_id,
        )

    return mapping


class FakeRetriever:
    def __init__(
        self,
        mapping,
        *,
        missed_query_ids=(),
    ):
        self.mapping = mapping
        self.missed_query_ids = set(
            missed_query_ids
        )
        self.calls = []

    @property
    def retriever_id(self) -> str:
        return "fake_hybrid_reranker_v1"

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        self.calls.append(
            (query, top_k)
        )

        if (
            query.query_id
            in self.missed_query_ids
        ):
            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="completed",
                retrieved_fact_ids=(),
                retrieved_evidence_ids=(),
                retrieved_chunk_ids=(),
                top_k=top_k,
                latency_ms=1.0,
            )

        fact_id, evidence_id = (
            self.mapping[query.query_id]
        )

        return ComplexRetrievalTrace(
            query_id=query.query_id,
            status="completed",
            retrieved_fact_ids=(
                fact_id,
            ),
            retrieved_evidence_ids=(
                evidence_id,
            ),
            retrieved_chunk_ids=(
                f"{query.report_id}_"
                f"chunk_{query.query_id}",
            ),
            top_k=top_k,
            latency_ms=1.0,
        )


class WrongQueryIdRetriever(
    FakeRetriever
):
    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        return ComplexRetrievalTrace(
            query_id="q9",
            status="completed",
            retrieved_fact_ids=(
                "fact_unexpected",
            ),
            retrieved_evidence_ids=(
                "evidence_unexpected",
            ),
            retrieved_chunk_ids=(
                "unexpected_chunk",
            ),
            top_k=top_k,
            latency_ms=1.0,
        )


class FakeCalculator:
    def __init__(self):
        self.calls = []

    @property
    def calculator_id(self) -> str:
        return (
            "deterministic_calculator_v1"
        )

    def calculate(
        self,
        *,
        calculation_id,
        formula_id,
        input_fact_ids,
    ):
        self.calls.append(
            (
                calculation_id,
                formula_id,
                input_fact_ids,
            )
        )

        return ComplexCalculationTrace(
            calculation_id=calculation_id,
            metric_id=(
                "gross_profit_margin"
            ),
            formula_id=formula_id,
            input_fact_ids=input_fact_ids,
            status="completed",
            result_value="20.7768",
            result_unit="percent",
            latency_ms=1.0,
        )


class FakeGenerator:
    def __init__(self):
        self.call_count = 0

    @property
    def generator_id(self) -> str:
        return (
            "deterministic_test_generator_v1"
        )

    def generate(
        self,
        *,
        question,
        rewrite,
        plan,
        retrieval_traces,
        calculation_traces,
    ):
        self.call_count += 1

        fact_ids = tuple(
            trace.retrieved_fact_ids[0]
            for trace in retrieval_traces
        )

        evidence_ids = tuple(
            trace.retrieved_evidence_ids[0]
            for trace in retrieval_traces
        )

        calculation_ids = tuple(
            trace.calculation_id
            for trace in calculation_traces
        )

        return ComplexFinalAnswerOutput(
            answer_text=(
                f"已根据 {len(fact_ids)} "
                "条事实生成测试答案。"
            ),
            supporting_fact_ids=fact_ids,
            supporting_calculation_ids=(
                calculation_ids
            ),
            citation_evidence_ids=(
                evidence_ids
            ),
        )


class InvalidReferenceGenerator(
    FakeGenerator
):
    def generate(
        self,
        *,
        question,
        rewrite,
        plan,
        retrieval_traces,
        calculation_traces,
    ):
        return ComplexFinalAnswerOutput(
            answer_text="无效引用测试。",
            supporting_fact_ids=(
                "fact_not_retrieved",
            ),
            citation_evidence_ids=(
                "evidence_not_retrieved",
            ),
        )


def test_oracle_rewrite_removes_gold_only_fields(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_001"]

    rewrite = build_gold_oracle_rewrite(
        case
    )

    dumped = rewrite.model_dump(
        mode="json"
    )

    query = dumped[
        "retrieval_queries"
    ][0]

    assert "target_fact_id" not in query
    assert "gold_pdf_pages" not in query
    assert "baseline_query" not in query


def test_oracle_plan_hides_fact_ids(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_002"]

    plan = build_gold_oracle_plan(
        case
    )

    retrieve_steps = [
        step
        for step in plan.steps
        if step.action == "retrieve"
    ]

    assert [
        step.output_ref
        for step in retrieve_steps
    ] == [
        "retrieval_result_q1",
        "retrieval_result_q2",
    ]

    calculation_step = plan.steps[2]

    assert calculation_step.input_refs == (
        "retrieval_result_q1",
        "retrieval_result_q2",
    )


def test_executes_non_calculation_case(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_001"]

    retriever = FakeRetriever(
        build_retrieval_mapping(case)
    )

    generator = FakeGenerator()

    result = execute_gold_oracle_case(
        run_id="complex_run_test_001",
        case=case,
        retriever=retriever,
        generator=generator,
        top_k=5,
    )

    assert result.status == "completed"
    assert result.error_stage is None
    assert len(result.retrieval_traces) == 2
    assert len(result.calculation_traces) == 0
    assert result.calculator_id is None
    assert generator.call_count == 1


def test_executes_calculation_case(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_002"]

    retriever = FakeRetriever(
        build_retrieval_mapping(case)
    )

    calculator = FakeCalculator()
    generator = FakeGenerator()

    result = execute_gold_oracle_case(
        run_id="complex_run_test_002",
        case=case,
        retriever=retriever,
        calculator=calculator,
        generator=generator,
        top_k=5,
    )

    assert result.status == "completed"
    assert len(result.retrieval_traces) == 2
    assert len(result.calculation_traces) == 1
    assert result.calculator_id == (
        "deterministic_calculator_v1"
    )

    assert len(calculator.calls) == 1

    _, _, input_fact_ids = (
        calculator.calls[0]
    )

    assert input_fact_ids == (
        "fact_hisense_home_2024_revenue",
        "fact_hisense_home_2024_operating_cost",
    )


def test_retrieval_miss_returns_failed_run(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_001"]

    retriever = FakeRetriever(
        build_retrieval_mapping(case),
        missed_query_ids=("q1",),
    )

    generator = FakeGenerator()

    result = execute_gold_oracle_case(
        run_id="complex_run_test_003",
        case=case,
        retriever=retriever,
        generator=generator,
    )

    assert result.status == "failed"
    assert result.error_stage == "retrieval"
    assert "没有解析出" in (
        result.error_message or ""
    )
    assert generator.call_count == 0


def test_wrong_query_id_returns_failed_run(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_001"]

    retriever = WrongQueryIdRetriever(
        build_retrieval_mapping(case)
    )

    result = execute_gold_oracle_case(
        run_id="complex_run_test_004",
        case=case,
        retriever=retriever,
        generator=FakeGenerator(),
    )

    assert result.status == "failed"
    assert result.error_stage == "retrieval"
    assert "错误的 query_id" in (
        result.error_message or ""
    )


def test_missing_calculator_returns_failed_run(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_002"]

    result = execute_gold_oracle_case(
        run_id="complex_run_test_005",
        case=case,
        retriever=FakeRetriever(
            build_retrieval_mapping(case)
        ),
        generator=FakeGenerator(),
        calculator=None,
    )

    assert result.status == "failed"
    assert result.error_stage == "calculation"
    assert "未提供" in (
        result.error_message or ""
    )


def test_invalid_answer_reference_is_blocked(
    cases_by_id,
) -> None:
    case = cases_by_id["complex_001"]

    result = execute_gold_oracle_case(
        run_id="complex_run_test_006",
        case=case,
        retriever=FakeRetriever(
            build_retrieval_mapping(case)
        ),
        generator=(
            InvalidReferenceGenerator()
        ),
    )

    assert result.status == "failed"
    assert result.error_stage == "answer"
    assert "未通过审计约束" in (
        result.error_message or ""
    )


def test_unverified_case_is_blocked(
    cases_by_id,
) -> None:
    case = cases_by_id[
        "complex_001"
    ].model_copy(
        update={
            "validation_status": (
                ValidationStatus.PENDING
            ),
            "validated_by": None,
            "validated_at": None,
        }
    )

    result = execute_gold_oracle_case(
        run_id="complex_run_test_007",
        case=case,
        retriever=FakeRetriever(
            build_retrieval_mapping(case)
        ),
        generator=FakeGenerator(),
    )

    assert result.status == "failed"
    assert result.error_stage == "planning"
    assert "尚未 verified" in (
        result.error_message or ""
    )