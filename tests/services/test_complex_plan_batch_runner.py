from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.complex_plan_eval_result import (
    ComplexPlanRunResult,
    ComplexRetrievalTrace,
)
from app.schemas.enums import ValidationStatus
from app.services.complex_oracle_answer_generator import (
    ComplexOracleAnswerGenerator,
)
from app.services.complex_oracle_calculator_adapter import (
    ComplexOracleCalculatorAdapter,
)
from app.services.complex_plan_batch_runner import (
    ComplexPlanBatchRunnerError,
    ComplexPlanBatchWriteError,
    run_complex_plan_batch,
    write_complex_plan_batch_results,
)
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
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
    19,
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
def cases():
    return load_complex_financial_eval_cases(
        CASES_PATH
    )


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


class RegistryTestRetriever:
    def __init__(
        self,
        bundle,
        *,
        miss_operating_cash_flow=False,
    ):
        self.bundle = bundle
        self.miss_operating_cash_flow = (
            miss_operating_cash_flow
        )

    @property
    def retriever_id(self) -> str:
        return "registry_test_retriever_v1"

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        should_miss = (
            self.miss_operating_cash_flow
            and query.company_id
            == "gree_electric"
            and query.metric_id
            == (
                "net_cash_flow_from_"
                "operating_activities"
            )
        )

        if should_miss:
            return ComplexRetrievalTrace(
                query_id=query.query_id,
                status="completed",
                retrieved_fact_ids=(),
                retrieved_evidence_ids=(),
                retrieved_chunk_ids=(),
                top_k=top_k,
                latency_ms=1.0,
            )

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
                f"batch_{query.query_id}",
            ),
            top_k=top_k,
            latency_ms=1.0,
        )


def build_batch(
    *,
    bundle,
    cases,
    retriever=None,
):
    return run_complex_plan_batch(
        cases=cases,
        run_id_prefix=(
            "complex_run_batch_test"
        ),
        retriever=(
            retriever
            or RegistryTestRetriever(bundle)
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


def test_runs_all_four_cases(
    bundle,
    cases,
) -> None:
    batch = build_batch(
        bundle=bundle,
        cases=cases,
    )

    assert batch.case_count == 4
    assert batch.completed_count == 4
    assert batch.failed_count == 0
    assert batch.refused_count == 0
    assert batch.all_completed is True

    assert [
        result.case_id
        for result in batch.results
    ] == [
        "complex_001",
        "complex_002",
        "complex_003",
        "complex_004",
    ]

    assert [
        result.run_id
        for result in batch.results
    ] == [
        (
            "complex_run_batch_test_"
            "complex_001"
        ),
        (
            "complex_run_batch_test_"
            "complex_002"
        ),
        (
            "complex_run_batch_test_"
            "complex_003"
        ),
        (
            "complex_run_batch_test_"
            "complex_004"
        ),
    ]


def test_single_case_failure_does_not_stop_batch(
    bundle,
    cases,
) -> None:
    retriever = RegistryTestRetriever(
        bundle,
        miss_operating_cash_flow=True,
    )

    batch = build_batch(
        bundle=bundle,
        cases=cases,
        retriever=retriever,
    )

    assert batch.case_count == 4
    assert batch.completed_count == 3
    assert batch.failed_count == 1
    assert batch.all_completed is False

    result_by_case_id = {
        result.case_id: result
        for result in batch.results
    }

    assert (
        result_by_case_id[
            "complex_003"
        ].status
        == "failed"
    )

    assert (
        result_by_case_id[
            "complex_004"
        ].status
        == "completed"
    )


def test_writes_valid_jsonl(
    bundle,
    cases,
    tmp_path,
) -> None:
    batch = build_batch(
        bundle=bundle,
        cases=cases,
    )

    output_path = (
        tmp_path
        / "complex_results.jsonl"
    )

    written_path = (
        write_complex_plan_batch_results(
            batch=batch,
            output_path=output_path,
        )
    )

    assert written_path == output_path
    assert output_path.is_file()

    lines = (
        output_path
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert len(lines) == 4

    loaded_results = tuple(
        ComplexPlanRunResult
        .model_validate_json(line)
        for line in lines
    )

    assert [
        result.case_id
        for result in loaded_results
    ] == [
        "complex_001",
        "complex_002",
        "complex_003",
        "complex_004",
    ]


def test_existing_output_is_not_overwritten(
    bundle,
    cases,
    tmp_path,
) -> None:
    batch = build_batch(
        bundle=bundle,
        cases=cases,
    )

    output_path = (
        tmp_path
        / "complex_results.jsonl"
    )

    write_complex_plan_batch_results(
        batch=batch,
        output_path=output_path,
    )

    original_content = (
        output_path.read_bytes()
    )

    with pytest.raises(
        ComplexPlanBatchWriteError,
        match="拒绝覆盖",
    ):
        write_complex_plan_batch_results(
            batch=batch,
            output_path=output_path,
        )

    assert (
        output_path.read_bytes()
        == original_content
    )


def test_duplicate_case_ids_are_rejected(
    bundle,
    cases,
) -> None:
    with pytest.raises(
        ComplexPlanBatchRunnerError,
        match="重复 case_id",
    ):
        build_batch(
            bundle=bundle,
            cases=(
                cases[0],
                cases[0],
            ),
        )


def test_invalid_run_id_prefix_is_rejected(
    bundle,
    cases,
) -> None:
    with pytest.raises(
        ComplexPlanBatchRunnerError,
        match="complex_run_",
    ):
        run_complex_plan_batch(
            cases=cases,
            run_id_prefix="batch_test",
            retriever=(
                RegistryTestRetriever(bundle)
            ),
            generator=(
                ComplexOracleAnswerGenerator(
                    registry_bundle=bundle
                )
            ),
        )


def test_invalid_top_k_is_rejected(
    bundle,
    cases,
) -> None:
    with pytest.raises(
        ComplexPlanBatchRunnerError,
        match="top_k 必须大于 0",
    ):
        run_complex_plan_batch(
            cases=cases,
            run_id_prefix=(
                "complex_run_batch_test"
            ),
            retriever=(
                RegistryTestRetriever(bundle)
            ),
            generator=(
                ComplexOracleAnswerGenerator(
                    registry_bundle=bundle
                )
            ),
            top_k=0,
        )