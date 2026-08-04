import json
from pathlib import Path

import pytest

from app.services.complex_plan_eval_dataset import (
    ComplexPlanEvalDatasetNotFoundError,
    InvalidComplexPlanEvalDatasetError,
    load_complex_financial_eval_cases,
)


QUESTION = (
    "美的集团2024年营业收入和"
    "归母净利润分别是多少？"
)

REVENUE_FACT_ID = (
    "fact_midea_group_2024_"
    "revenue_consolidated"
)

PARENT_NET_PROFIT_FACT_ID = (
    "fact_midea_group_2024_"
    "parent_net_profit_consolidated"
)

REVENUE_EVIDENCE_ID = (
    "evidence_midea_group_2024_revenue"
)

PARENT_NET_PROFIT_EVIDENCE_ID = (
    "evidence_midea_group_2024_"
    "parent_net_profit"
)


def build_valid_case(
    case_id: str = "complex_001",
) -> dict:
    """构造一条合法的复杂财务评测记录。"""

    return {
        "schema_version": 1,
        "case_id": case_id,
        "question": QUESTION,
        "question_type": (
            "single_company_multi_metric"
        ),
        "difficulty": "medium",
        "company_ids": [
            "midea_group",
        ],
        "report_ids": [
            "midea_group_2024",
        ],
        "fiscal_years": [
            2024,
        ],
        "gold_rewrite": {
            "normalized_question": (
                "查询美的集团2024年合并口径"
                "营业收入和归母净利润"
            ),
            "retrieval_queries": [
                {
                    "query_id": "q1",
                    "target_fact_id": (
                        REVENUE_FACT_ID
                    ),
                    "baseline_query": QUESTION,
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
                        PARENT_NET_PROFIT_FACT_ID
                    ),
                    "baseline_query": QUESTION,
                    "semantic_query": (
                        "美的集团 2024年 "
                        "合并利润表 "
                        "归属于母公司股东的净利润"
                    ),
                    "company_id": "midea_group",
                    "report_id": (
                        "midea_group_2024"
                    ),
                    "metric_id": (
                        "parent_net_profit"
                    ),
                    "fiscal_year": 2024,
                    "statement_type": (
                        "income_statement"
                    ),
                    "statement_scope": (
                        "consolidated"
                    ),
                    "gold_pdf_pages": [
                        159,
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
                        "检索美的集团营业收入"
                    ),
                    "target_fact_ids": [
                        REVENUE_FACT_ID,
                    ],
                    "output_ref": (
                        REVENUE_FACT_ID
                    ),
                },
                {
                    "step_id": "s2",
                    "action": "retrieve",
                    "description": (
                        "检索美的集团归母净利润"
                    ),
                    "target_fact_ids": [
                        PARENT_NET_PROFIT_FACT_ID,
                    ],
                    "output_ref": (
                        PARENT_NET_PROFIT_FACT_ID
                    ),
                },
                {
                    "step_id": "s3",
                    "action": "synthesize",
                    "description": (
                        "汇总营业收入和归母净利润"
                    ),
                    "input_refs": [
                        REVENUE_FACT_ID,
                        PARENT_NET_PROFIT_FACT_ID,
                    ],
                    "depends_on": [
                        "s1",
                        "s2",
                    ],
                    "output_ref": (
                        "answer_midea_group_2024_"
                        "revenue_and_parent_net_profit"
                    ),
                },
            ],
            "final_step_id": "s3",
        },
        "gold_fact_ids": [
            REVENUE_FACT_ID,
            PARENT_NET_PROFIT_FACT_ID,
        ],
        "gold_evidence_ids": [
            REVENUE_EVIDENCE_ID,
            PARENT_NET_PROFIT_EVIDENCE_ID,
        ],
        "gold_answer": {
            "answer_text": (
                "美的集团2024年营业收入和"
                "归母净利润见对应年报。"
            ),
            "conclusion": (
                "两个指标均来自2024年"
                "合并利润表。"
            ),
            "supporting_fact_ids": [
                REVENUE_FACT_ID,
                PARENT_NET_PROFIT_FACT_ID,
            ],
            "evidence_ids": [
                REVENUE_EVIDENCE_ID,
                PARENT_NET_PROFIT_EVIDENCE_ID,
            ],
        },
        "source_version": (
            "complex_financial_gold_plan_v1"
        ),
        "created_at": (
            "2026-08-02T16:00:00+08:00"
        ),
        "updated_at": (
            "2026-08-02T16:00:00+08:00"
        ),
    }


def serialize_case(data: dict) -> str:
    """将一条 Case 序列化为单行 JSON。"""

    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_cases(
    path: Path,
    *cases: dict,
) -> None:
    """向临时 JSONL 文件写入多条 Case。"""

    text = "\n".join(
        serialize_case(case)
        for case in cases
    )

    path.write_text(
        text + "\n",
        encoding="utf-8",
    )


def test_load_valid_complex_case(
    tmp_path: Path,
) -> None:
    """应成功加载一条合法记录。"""

    path = tmp_path / "complex_cases.jsonl"

    write_cases(
        path,
        build_valid_case(),
    )

    cases = load_complex_financial_eval_cases(
        path
    )

    assert isinstance(cases, tuple)
    assert len(cases) == 1
    assert cases[0].case_id == "complex_001"
    assert len(cases[0].gold_plan.steps) == 3


def test_load_multiple_complex_cases(
    tmp_path: Path,
) -> None:
    """应按 JSONL 原始顺序加载多条记录。"""

    path = tmp_path / "complex_cases.jsonl"

    write_cases(
        path,
        build_valid_case("complex_001"),
        build_valid_case("complex_002"),
    )

    cases = load_complex_financial_eval_cases(
        path
    )

    assert [
        case.case_id
        for case in cases
    ] == [
        "complex_001",
        "complex_002",
    ]


def test_reject_missing_dataset(
    tmp_path: Path,
) -> None:
    """文件不存在时应抛出专用异常。"""

    path = tmp_path / "missing.jsonl"

    with pytest.raises(
        ComplexPlanEvalDatasetNotFoundError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    assert str(path) in str(exc_info.value)


def test_reject_empty_dataset(
    tmp_path: Path,
) -> None:
    """空文件不能作为复杂评测数据集。"""

    path = tmp_path / "empty.jsonl"

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    assert "不能为空" in str(exc_info.value)
    assert str(path) in str(exc_info.value)


def test_reject_blank_record(
    tmp_path: Path,
) -> None:
    """JSONL 中间不能出现空记录。"""

    path = tmp_path / "blank_line.jsonl"

    first_line = serialize_case(
        build_valid_case("complex_001")
    )

    third_line = serialize_case(
        build_valid_case("complex_002")
    )

    path.write_text(
        first_line
        + "\n\n"
        + third_line
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    assert "第 2 行" in str(exc_info.value)


def test_reject_non_utf8_dataset(
    tmp_path: Path,
) -> None:
    """数据集必须使用 UTF-8 编码。"""

    path = tmp_path / "invalid_encoding.jsonl"

    path.write_bytes(
        b"\xff\xfe\x00\x00"
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    assert "UTF-8" in str(exc_info.value)
    assert str(path) in str(exc_info.value)


def test_reject_invalid_json_with_line_number(
    tmp_path: Path,
) -> None:
    """非法 JSON 应报告准确行号。"""

    path = tmp_path / "invalid_json.jsonl"

    first_line = serialize_case(
        build_valid_case("complex_001")
    )

    path.write_text(
        first_line
        + "\n"
        + "{not-valid-json}"
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    message = str(exc_info.value)

    assert "第 2 行" in message
    assert str(path) in message


def test_reject_schema_error_with_line_number(
    tmp_path: Path,
) -> None:
    """Schema 无效时应报告准确行号。"""

    path = tmp_path / "invalid_schema.jsonl"

    invalid_case = build_valid_case(
        "complex_002"
    )

    del invalid_case["gold_plan"]

    path.write_text(
        serialize_case(
            build_valid_case("complex_001")
        )
        + "\n"
        + serialize_case(invalid_case)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    message = str(exc_info.value)

    assert "第 2 行" in message
    assert str(path) in message


def test_reject_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    """整个 JSONL 中的 case_id 必须唯一。"""

    path = tmp_path / "duplicate_ids.jsonl"

    write_cases(
        path,
        build_valid_case("complex_001"),
        build_valid_case("complex_001"),
    )

    with pytest.raises(
        InvalidComplexPlanEvalDatasetError,
    ) as exc_info:
        load_complex_financial_eval_cases(
            path
        )

    message = str(exc_info.value)

    assert "重复 case_id" in message
    assert "complex_001" in message
    assert "第 1 行" in message
    assert "第 2 行" in message