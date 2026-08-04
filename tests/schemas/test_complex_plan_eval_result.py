from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.schemas.complex_plan_eval_result import (
    ComplexPlanRunResult,
)


def build_valid_completed_run() -> dict:
    return {
        "schema_version": 1,
        "run_id": (
            "complex_run_pilot_001"
        ),
        "case_id": "complex_001",
        "execution_mode": "gold_oracle",
        "status": "completed",
        "rewrite": {
            "normalized_question": (
                "查询美的集团2024年营业收入"
            ),
            "retrieval_queries": [
                {
                    "query_id": "q1",
                    "semantic_query": (
                        "美的集团 2024年 "
                        "合并利润表 营业收入"
                    ),
                    "company_id": "midea_group",
                    "report_id": (
                        "midea_group_2024"
                    ),
                    "metric_id": "revenue",
                    "fiscal_year": 2024,
                    "report_type": (
                        "annual_report"
                    ),
                    "statement_type": (
                        "income_statement"
                    ),
                    "statement_scope": (
                        "consolidated"
                    ),
                }
            ],
        },
        "plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "action": "retrieve",
                    "description": "检索营业收入",
                    "output_ref": (
                        "fact_request_1"
                    ),
                    "retrieval_query_id": "q1",
                },
                {
                    "step_id": "s2",
                    "action": "synthesize",
                    "description": "生成回答",
                    "input_refs": [
                        "fact_request_1"
                    ],
                    "depends_on": ["s1"],
                    "output_ref": (
                        "answer_result"
                    ),
                },
            ],
            "final_step_id": "s2",
        },
        "retrieval_traces": [
            {
                "query_id": "q1",
                "status": "completed",
                "retrieved_fact_ids": [
                    "fact_midea_group_2024_revenue"
                ],
                "retrieved_evidence_ids": [
                    "evidence_midea_group_2024_revenue"
                ],
                "retrieved_chunk_ids": [
                    "midea_group_2024_chunk_001"
                ],
                "top_k": 5,
                "latency_ms": 25.5,
            }
        ],
        "calculation_traces": [],
        "answer": {
            "answer_text": (
                "美的集团2024年营业收入为"
                "407,149,600,000元。"
            ),
            "supporting_fact_ids": [
                "fact_midea_group_2024_revenue"
            ],
            "supporting_calculation_ids": [],
            "citation_evidence_ids": [
                "evidence_midea_group_2024_revenue"
            ],
        },
        "planner_id": "gold_oracle_v1",
        "retriever_id": (
            "hybrid_reranker_v1"
        ),
        "generator_id": (
            "deterministic_synthesizer_v1"
        ),
        "started_at": (
            "2026-08-02T16:00:00+08:00"
        ),
        "completed_at": (
            "2026-08-02T16:00:01+08:00"
        ),
        "latency_ms": 1000.0,
    }


def test_valid_completed_run() -> None:
    result = ComplexPlanRunResult.model_validate(
        build_valid_completed_run()
    )

    assert result.status == "completed"
    assert result.execution_mode == (
        "gold_oracle"
    )
    assert len(result.retrieval_traces) == 1
    assert result.answer is not None


def test_rejects_duplicate_rewrite_query_ids() -> None:
    data = build_valid_completed_run()

    duplicate_query = deepcopy(
        data["rewrite"]["retrieval_queries"][0]
    )

    data["rewrite"][
        "retrieval_queries"
    ].append(duplicate_query)

    with pytest.raises(
        ValidationError,
        match="query_id 必须唯一",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_rejects_retrieve_without_query_id() -> None:
    data = build_valid_completed_run()

    del data["plan"]["steps"][0][
        "retrieval_query_id"
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "retrieve 步骤必须填写 "
            "retrieval_query_id"
        ),
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_rejects_unavailable_input_ref() -> None:
    data = build_valid_completed_run()

    data["plan"]["steps"][1][
        "input_refs"
    ] = ["missing_result"]

    with pytest.raises(
        ValidationError,
        match="尚未生成",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_completed_run_requires_answer() -> None:
    data = build_valid_completed_run()
    data["answer"] = None

    with pytest.raises(
        ValidationError,
        match="必须包含 answer",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_rewrite_and_trace_must_match() -> None:
    data = build_valid_completed_run()
    data["retrieval_traces"] = []

    with pytest.raises(
        ValidationError,
        match=(
            "Rewrite Query 与 Retrieval Trace"
        ),
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_answer_cannot_use_unretrieved_fact() -> None:
    data = build_valid_completed_run()

    data["answer"][
        "supporting_fact_ids"
    ] = [
        "fact_not_retrieved"
    ]

    with pytest.raises(
        ValidationError,
        match="未检索到的 Fact",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_answer_cannot_use_unretrieved_evidence() -> None:
    data = build_valid_completed_run()

    data["answer"][
        "citation_evidence_ids"
    ] = [
        "evidence_not_retrieved"
    ]

    with pytest.raises(
        ValidationError,
        match="未检索到的 Evidence",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_failed_run_requires_error() -> None:
    data = build_valid_completed_run()

    data["status"] = "failed"
    data["answer"] = None

    with pytest.raises(
        ValidationError,
        match="必须填写 error_stage",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )


def test_valid_failed_run() -> None:
    data = {
        "schema_version": 1,
        "run_id": (
            "complex_run_failed_001"
        ),
        "case_id": "complex_001",
        "execution_mode": "agent",
        "status": "failed",
        "rewrite": None,
        "plan": None,
        "retrieval_traces": [],
        "calculation_traces": [],
        "answer": None,
        "planner_id": "planner_v1",
        "retriever_id": None,
        "generator_id": None,
        "started_at": (
            "2026-08-02T16:00:00+08:00"
        ),
        "completed_at": (
            "2026-08-02T16:00:00+08:00"
        ),
        "latency_ms": 10.0,
        "error_stage": "rewrite",
        "error_message": (
            "无法识别目标公司"
        ),
    }

    result = ComplexPlanRunResult.model_validate(
        data
    )

    assert result.status == "failed"
    assert result.error_stage == "rewrite"


def test_rejects_naive_datetime() -> None:
    data = build_valid_completed_run()

    data["started_at"] = (
        "2026-08-02T16:00:00"
    )

    with pytest.raises(
        ValidationError,
        match="必须包含时区",
    ):
        ComplexPlanRunResult.model_validate(
            data
        )