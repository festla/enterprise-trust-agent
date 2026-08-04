from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)
from app.schemas.enums import (
    ReportType,
    ValidationStatus,
)


CHINA_TIMEZONE = timezone(
    timedelta(hours=8)
)

QUESTION = (
    "美的集团和格力电器2024年营业收入谁更高？"
)

MIDEA_REVENUE_FACT_ID = (
    "fact_midea_group_2024_revenue_consolidated"
)

GREE_REVENUE_FACT_ID = (
    "fact_gree_electric_2024_revenue_consolidated"
)

MIDEA_REVENUE_EVIDENCE_ID = (
    "evidence_midea_group_2024_revenue"
)

GREE_REVENUE_EVIDENCE_ID = (
    "evidence_gree_electric_2024_revenue"
)


def build_valid_complex_case() -> dict:
    """构造一个合法的跨公司比较评测 Case。"""

    created_at = datetime(
        2026,
        8,
        2,
        16,
        0,
        tzinfo=CHINA_TIMEZONE,
    )

    return {
        "schema_version": 1,
        "case_id": "complex_001",
        "question": QUESTION,
        "question_type": (
            "cross_company_comparison"
        ),
        "difficulty": "medium",
        "company_ids": [
            "midea_group",
            "gree_electric",
        ],
        "report_ids": [
            "midea_group_2024",
            "gree_electric_2024",
        ],
        "fiscal_years": [
            2024,
        ],
        "gold_rewrite": {
            "normalized_question": (
                "比较美的集团与格力电器2024年"
                "合并口径营业收入"
            ),
            "resolved_aliases": [
                {
                    "source_text": "美的集团",
                    "alias_type": "company",
                    "normalized_value": (
                        "midea_group"
                    ),
                },
                {
                    "source_text": "格力电器",
                    "alias_type": "company",
                    "normalized_value": (
                        "gree_electric"
                    ),
                },
                {
                    "source_text": "营业收入",
                    "alias_type": "metric",
                    "normalized_value": "revenue",
                },
                {
                    "source_text": "2024年",
                    "alias_type": "fiscal_year",
                    "normalized_value": "2024",
                },
            ],
            "retrieval_queries": [
                {
                    "query_id": "q1",
                    "target_fact_id": (
                        MIDEA_REVENUE_FACT_ID
                    ),
                    "baseline_query": QUESTION,
                    "semantic_query": (
                        "美的集团 2024 年 "
                        "合并利润表 营业收入"
                    ),
                    "company_id": "midea_group",
                    "report_id": (
                        "midea_group_2024"
                    ),
                    "metric_id": "revenue",
                    "fiscal_year": 2024,
                    # 不填写 report_type，
                    # 用于测试默认值。
                    "statement_type": (
                        "income_statement"
                    ),
                    "statement_scope": (
                        "consolidated"
                    ),
                    "gold_pdf_pages": [
                        158,
                    ],
                },
                {
                    "query_id": "q2",
                    "target_fact_id": (
                        GREE_REVENUE_FACT_ID
                    ),
                    "baseline_query": QUESTION,
                    "semantic_query": (
                        "格力电器 2024 年 "
                        "合并利润表 营业收入"
                    ),
                    "company_id": (
                        "gree_electric"
                    ),
                    "report_id": (
                        "gree_electric_2024"
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
                    "gold_pdf_pages": [
                        113,
                    ],
                },
            ],
        },
        "gold_plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "action": "retrieve",
                    "description": (
                        "检索美的集团2024年营业收入"
                    ),
                    "target_fact_ids": [
                        MIDEA_REVENUE_FACT_ID,
                    ],
                    "input_refs": [],
                    "depends_on": [],
                    "output_ref": (
                        MIDEA_REVENUE_FACT_ID
                    ),
                },
                {
                    "step_id": "s2",
                    "action": "retrieve",
                    "description": (
                        "检索格力电器2024年营业收入"
                    ),
                    "target_fact_ids": [
                        GREE_REVENUE_FACT_ID,
                    ],
                    "input_refs": [],
                    "depends_on": [],
                    "output_ref": (
                        GREE_REVENUE_FACT_ID
                    ),
                },
                {
                    "step_id": "s3",
                    "action": "compare",
                    "description": (
                        "比较两家公司2024年营业收入"
                    ),
                    "target_fact_ids": [],
                    "input_refs": [
                        MIDEA_REVENUE_FACT_ID,
                        GREE_REVENUE_FACT_ID,
                    ],
                    "depends_on": [
                        "s1",
                        "s2",
                    ],
                    "output_ref": (
                        "comparison_revenue_2024"
                    ),
                },
            ],
            "final_step_id": "s3",
        },
        "gold_fact_ids": [
            MIDEA_REVENUE_FACT_ID,
            GREE_REVENUE_FACT_ID,
        ],
        "gold_evidence_ids": [
            MIDEA_REVENUE_EVIDENCE_ID,
            GREE_REVENUE_EVIDENCE_ID,
        ],
        "gold_calculation_ids": [],
        "gold_answer": {
            "answer_text": (
                "美的集团2024年营业收入"
                "高于格力电器。"
            ),
            "conclusion": (
                "在2024年合并报表口径下，"
                "美的集团营业收入更高。"
            ),
            "supporting_fact_ids": [
                MIDEA_REVENUE_FACT_ID,
                GREE_REVENUE_FACT_ID,
            ],
            "evidence_ids": [
                MIDEA_REVENUE_EVIDENCE_ID,
                GREE_REVENUE_EVIDENCE_ID,
            ],
            "supporting_calculation_ids": [],
        },
        "validation_status": "pending",
        "validated_by": None,
        "validated_at": None,
        "source_version": (
            "complex_financial_gold_plan_v1"
        ),
        "created_at": created_at,
        "updated_at": created_at,
        "review_notes": None,
    }


def build_valid_calculation_case() -> dict:
    """构造包含计算步骤的合法评测 Case。"""

    data = build_valid_complex_case()

    calculation_id = (
        "calculation_midea_gree_2024_"
        "revenue_difference"
    )

    calculation_step = (
        data["gold_plan"]["steps"][2]
    )

    calculation_step.update(
        {
            "action": "calculate",
            "description": (
                "计算美的集团与格力电器"
                "营业收入之差"
            ),
            "output_ref": calculation_id,
            "calculation_id": calculation_id,
            "formula_id": "subtract_formula",
        }
    )

    data["gold_calculation_ids"] = [
        calculation_id,
    ]

    data["gold_answer"][
        "supporting_calculation_ids"
    ] = [
        calculation_id,
    ]

    return data


def test_create_valid_cross_company_case() -> None:
    """合法跨公司比较 Case 应创建成功。"""

    case = ComplexFinancialEvalCase.model_validate(
        build_valid_complex_case()
    )

    assert case.case_id == "complex_001"

    assert case.question_type == (
        "cross_company_comparison"
    )

    assert len(
        case.gold_rewrite.retrieval_queries
    ) == 2

    assert case.gold_plan.final_step_id == "s3"

    assert (
        case.gold_rewrite
        .retrieval_queries[0]
        .report_type
        is ReportType.ANNUAL_REPORT
    )

    assert (
        case.validation_status
        is ValidationStatus.PENDING
    )


def test_complex_case_json_round_trip() -> None:
    """Case 序列化为 JSON 后应能够恢复。"""

    original = (
        ComplexFinancialEvalCase.model_validate(
            build_valid_complex_case()
        )
    )

    json_text = original.model_dump_json()

    restored = (
        ComplexFinancialEvalCase
        .model_validate_json(json_text)
    )

    assert restored == original


def test_create_valid_calculation_plan() -> None:
    """包含规范计算步骤的 Plan 应创建成功。"""

    case = ComplexFinancialEvalCase.model_validate(
        build_valid_calculation_case()
    )

    calculation_step = case.gold_plan.steps[2]

    assert calculation_step.action == "calculate"

    assert calculation_step.calculation_id == (
        "calculation_midea_gree_2024_"
        "revenue_difference"
    )

    assert calculation_step.formula_id == (
        "subtract_formula"
    )

    assert len(
        case.gold_calculation_ids
    ) == 1


def test_complex_case_is_frozen() -> None:
    """复杂评测 Case 创建后不应被直接修改。"""

    case = ComplexFinancialEvalCase.model_validate(
        build_valid_complex_case()
    )

    with pytest.raises(ValidationError):
        case.question = "修改后的问题"


def test_reject_unknown_extra_field() -> None:
    """Schema 契约外的字段应被拒绝。"""

    data = build_valid_complex_case()
    data["unexpected_field"] = "unexpected"

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_duplicate_query_ids() -> None:
    """Gold Rewrite 的 query_id 必须唯一。"""

    data = build_valid_complex_case()

    data["gold_rewrite"][
        "retrieval_queries"
    ][1]["query_id"] = "q1"

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_rewrite_fact_set_mismatch() -> None:
    """Rewrite 目标事实必须匹配 Case 事实集合。"""

    data = build_valid_complex_case()

    data["gold_rewrite"][
        "retrieval_queries"
    ][1]["target_fact_id"] = (
        "fact_other_company_2024_revenue"
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_missing_plan_dependency() -> None:
    """步骤消费某个输入时必须依赖其生产步骤。"""

    data = build_valid_complex_case()

    compare_step = data["gold_plan"]["steps"][2]

    # compare 同时使用 s1、s2 的输出，
    # 但这里故意遗漏 s2。
    compare_step["depends_on"] = [
        "s1",
    ]

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_wrong_final_step() -> None:
    """final_step_id 必须指向最后一个步骤。"""

    data = build_valid_complex_case()

    data["gold_plan"]["final_step_id"] = "s2"

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_calculation_output_mismatch() -> None:
    """计算步骤的输出必须等于 calculation_id。"""

    data = build_valid_calculation_case()

    calculation_step = (
        data["gold_plan"]["steps"][2]
    )

    calculation_step["output_ref"] = (
        "calculation_unexpected_output"
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_wrong_single_company_contract() -> None:
    """单公司问题不能同时声明两家公司。"""

    data = build_valid_complex_case()

    data["question_type"] = (
        "single_company_multi_metric"
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_hard_case_without_complexity() -> None:
    """hard 问题必须满足最低复杂度约束。"""

    data = build_valid_complex_case()
    data["difficulty"] = "hard"

    # 当前 Case 只有两家公司、两个事实，
    # 不满足 hard 的复杂度要求。
    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_answer_unknown_evidence() -> None:
    """答案不能引用 Case 未声明的证据。"""

    data = build_valid_complex_case()

    data["gold_answer"]["evidence_ids"] = [
        "evidence_unknown",
    ]

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_verified_case_without_metadata() -> None:
    """verified Case 必须记录审核者和审核时间。"""

    data = build_valid_complex_case()

    data["validation_status"] = "verified"

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_pending_case_with_review_metadata() -> None:
    """pending Case 不应提前填写人工审核信息。"""

    data = build_valid_complex_case()

    data["validated_by"] = "human_reviewer"

    data["validated_at"] = datetime(
        2026,
        8,
        2,
        17,
        0,
        tzinfo=CHINA_TIMEZONE,
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_naive_timestamp() -> None:
    """时间字段必须包含时区信息。"""

    data = build_valid_complex_case()

    data["created_at"] = datetime(
        2026,
        8,
        2,
        16,
        0,
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )


def test_reject_report_from_undeclared_company() -> None:
    """报告必须唯一归属于 Case 声明的公司。"""

    data = build_valid_complex_case()

    data["report_ids"][0] = (
        "haier_smart_home_2024"
    )

    with pytest.raises(ValidationError):
        ComplexFinancialEvalCase.model_validate(
            data
        )