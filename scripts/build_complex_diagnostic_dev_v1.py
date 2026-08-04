from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)
from app.schemas.enums import ValidationStatus
from app.services.complex_plan_eval_dataset import (
    load_complex_financial_eval_cases,
)
from app.services.complex_plan_eval_integrity import (
    validate_complex_plan_eval_integrity,
)
from app.services.registry_loader import (
    load_registry_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "registries"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_diagnostic_dev_v1.jsonl"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_diagnostic_dev_v1_manifest.json"
)

DEV_V2_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_dev_v2.jsonl"
)

FROZEN_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "complex_plan"
    / "complex_plan_test_v1.jsonl"
)

SOURCE_VERSION = "complex_plan_diagnostic_dev_v1"

CHINA_TIMEZONE = timezone(
    timedelta(hours=8)
)


def ordered_unique(
    values: list[str],
) -> list[str]:
    result: list[str] = []

    for value in values:
        if value not in result:
            result.append(value)

    return result


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_query(
    *,
    company_id: str,
    metric_id: str,
    statement_type: str,
    pdf_page: int,
    semantic_query: str,
) -> dict[str, Any]:
    report_id = f"{company_id}_2024"

    return {
        "company_id": company_id,
        "report_id": report_id,
        "metric_id": metric_id,
        "statement_type": statement_type,
        "statement_scope": "consolidated",
        "fiscal_year": 2024,
        "pdf_page": pdf_page,
        "semantic_query": semantic_query,
        "fact_id": (
            f"fact_{company_id}_2024_{metric_id}"
        ),
        "evidence_id": (
            f"evidence_{company_id}_2024_{metric_id}"
        ),
    }


def build_definitions() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "complex_031",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "medium",
            "question": (
                "美的集团2024年末合并口径非流动资产合计"
                "和资产总计分别是多少？哪一项金额更高？"
            ),
            "normalized_question": (
                "查询美的集团2024年末合并资产负债表中的"
                "非流动资产合计和资产总计，并比较金额。"
            ),
            "items": [
                atomic_query(
                    company_id="midea_group",
                    metric_id="non_current_assets",
                    statement_type="balance_sheet",
                    pdf_page=156,
                    semantic_query=(
                        "美的集团 2024年12月31日 "
                        "合并资产负债表 非流动资产合计"
                    ),
                ),
                atomic_query(
                    company_id="midea_group",
                    metric_id="total_assets",
                    statement_type="balance_sheet",
                    pdf_page=156,
                    semantic_query=(
                        "美的集团 2024年12月31日 "
                        "合并资产负债表 资产总计"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较非流动资产合计和资产总计。"
                    ),
                    "query_numbers": [1, 2],
                    "output_ref": (
                        "comparison_complex_031_assets"
                    ),
                },
            ],
            "answer_text": (
                "美的集团2024年末合并口径非流动资产合计"
                "为215,288,067,000元，资产总计为"
                "604,351,853,000元；资产总计更高。"
            ),
            "conclusion": (
                "美的集团2024年末资产总计高于"
                "非流动资产合计。"
            ),
        },
        {
            "case_id": "complex_032",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "medium",
            "question": (
                "格力电器2024年合并口径经营活动产生的"
                "现金流量净额和投资活动现金流出小计"
                "分别是多少？哪一项金额更高？"
            ),
            "normalized_question": (
                "查询格力电器2024年度合并现金流量表中的"
                "经营活动现金流量净额和投资活动现金流出"
                "小计，并比较金额。"
            ),
            "items": [
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "net_cash_flow_from_"
                        "operating_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=114,
                    semantic_query=(
                        "格力电器 2024年度 合并现金流量表 "
                        "经营活动产生的现金流量净额"
                    ),
                ),
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "cash_outflows_from_investing_"
                        "activities_subtotal"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=114,
                    semantic_query=(
                        "格力电器 2024年度 合并现金流量表 "
                        "投资活动现金流出小计"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较经营活动现金流量净额和"
                        "投资活动现金流出小计。"
                    ),
                    "query_numbers": [1, 2],
                    "output_ref": (
                        "comparison_complex_032_cash_flows"
                    ),
                },
            ],
            "answer_text": (
                "格力电器2024年合并口径经营活动产生的"
                "现金流量净额为29,369,250,570.66元，"
                "投资活动现金流出小计为"
                "50,412,775,368.32元；"
                "投资活动现金流出小计更高。"
            ),
            "conclusion": (
                "格力电器2024年投资活动现金流出小计"
                "高于经营活动产生的现金流量净额。"
            ),
        },
        {
            "case_id": "complex_033",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "medium",
            "question": (
                "海尔智家2024年合并口径营业收入和"
                "税金及附加分别是多少？哪一项金额更高？"
            ),
            "normalized_question": (
                "查询海尔智家2024年度合并利润表中的"
                "营业收入和税金及附加，并比较金额。"
            ),
            "items": [
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="revenue",
                    statement_type="income_statement",
                    pdf_page=121,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "营业收入"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="taxes_and_surcharges",
                    statement_type="income_statement",
                    pdf_page=122,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "税金及附加"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较营业收入和税金及附加。"
                    ),
                    "query_numbers": [1, 2],
                    "output_ref": (
                        "comparison_complex_033_income_items"
                    ),
                },
            ],
            "answer_text": (
                "海尔智家2024年合并口径营业收入为"
                "285,981,225,203.93元，税金及附加为"
                "1,276,040,830.27元；营业收入更高。"
            ),
            "conclusion": (
                "海尔智家2024年营业收入高于"
                "税金及附加。"
            ),
        },
        {
            "case_id": "complex_034",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "hard",
            "question": (
                "海信家电2024年合并口径销售商品、提供劳务"
                "收到的现金、收到的税费返还、经营活动产生的"
                "现金流量净额和投资活动产生的现金流量净额"
                "分别是多少？并判断后两项分别为净流入"
                "还是净流出。"
            ),
            "normalized_question": (
                "查询海信家电2024年度合并现金流量表中的"
                "销售商品提供劳务收到的现金、收到的税费返还、"
                "经营活动现金流量净额和投资活动现金流量净额，"
                "并根据正负号判断后两项的现金流方向。"
            ),
            "items": [
                atomic_query(
                    company_id="hisense_home",
                    metric_id=(
                        "cash_received_from_sales_of_goods_"
                        "and_rendering_of_services"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=119,
                    semantic_query=(
                        "海信家电 2024年度 合并现金流量表 "
                        "销售商品、提供劳务收到的现金"
                    ),
                ),
                atomic_query(
                    company_id="hisense_home",
                    metric_id="tax_refunds_received",
                    statement_type="cash_flow_statement",
                    pdf_page=120,
                    semantic_query=(
                        "海信家电 2024年度 合并现金流量表 "
                        "收到的税费返还"
                    ),
                ),
                atomic_query(
                    company_id="hisense_home",
                    metric_id=(
                        "net_cash_flow_from_"
                        "operating_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=120,
                    semantic_query=(
                        "海信家电 2024年度 合并现金流量表 "
                        "经营活动产生的现金流量净额"
                    ),
                ),
                atomic_query(
                    company_id="hisense_home",
                    metric_id=(
                        "net_cash_flow_from_"
                        "investing_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=120,
                    semantic_query=(
                        "海信家电 2024年度 合并现金流量表 "
                        "投资活动产生的现金流量净额"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "根据经营活动和投资活动现金流量"
                        "净额的正负号判断净流入或净流出。"
                    ),
                    "query_numbers": [3, 4],
                    "output_ref": (
                        "comparison_complex_034_flow_directions"
                    ),
                },
            ],
            "answer_text": (
                "海信家电2024年合并口径销售商品、提供劳务"
                "收到的现金为76,577,033,223.57元，"
                "收到的税费返还为3,516,048,170.30元，"
                "经营活动产生的现金流量净额为"
                "5,132,164,941.24元，为净流入；"
                "投资活动产生的现金流量净额为"
                "-619,437,088.11元，为净流出。"
            ),
            "conclusion": (
                "海信家电2024年经营活动产生现金净流入，"
                "投资活动产生现金净流出。"
            ),
        },
        {
            "case_id": "complex_035",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "medium",
            "question": (
                "格力电器2024年合并口径净利润和归属于"
                "母公司股东的净利润分别是多少？"
                "两者是否相同，哪一项更高？"
            ),
            "normalized_question": (
                "查询格力电器2024年度合并利润表中的"
                "净利润和归属于母公司股东的净利润，"
                "并比较两项指标。"
            ),
            "items": [
                atomic_query(
                    company_id="gree_electric",
                    metric_id="net_profit",
                    statement_type="income_statement",
                    pdf_page=113,
                    semantic_query=(
                        "格力电器 2024年度 合并利润表 净利润"
                    ),
                ),
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "net_profit_attributable_to_parent"
                    ),
                    statement_type="income_statement",
                    pdf_page=113,
                    semantic_query=(
                        "格力电器 2024年度 合并利润表 "
                        "归属于母公司股东的净利润"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较净利润和归属于母公司股东的"
                        "净利润是否相同以及哪项更高。"
                    ),
                    "query_numbers": [1, 2],
                    "output_ref": (
                        "comparison_complex_035_net_profits"
                    ),
                },
            ],
            "answer_text": (
                "格力电器2024年合并口径净利润为"
                "32,371,069,287.35元，归属于母公司股东的"
                "净利润为32,184,570,372.28元；"
                "两者不相同，净利润更高。"
            ),
            "conclusion": (
                "格力电器2024年合并净利润略高于"
                "归属于母公司股东的净利润。"
            ),
        },
        {
            "case_id": "complex_036",
            "question_type": (
                "cross_company_comparison"
            ),
            "difficulty": "hard",
            "question": (
                "比较格力电器和海尔智家2024年合并口径"
                "其他综合收益的税后净额及综合收益总额，"
                "分别判断哪家公司更高。"
            ),
            "normalized_question": (
                "分别查询格力电器和海尔智家2024年度"
                "合并利润表中的其他综合收益税后净额和"
                "综合收益总额，并按相同指标进行公司间比较。"
            ),
            "items": [
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "other_comprehensive_income_net_of_tax"
                    ),
                    statement_type="income_statement",
                    pdf_page=113,
                    semantic_query=(
                        "格力电器 2024年度 合并利润表 "
                        "其他综合收益的税后净额"
                    ),
                ),
                atomic_query(
                    company_id="gree_electric",
                    metric_id="total_comprehensive_income",
                    statement_type="income_statement",
                    pdf_page=113,
                    semantic_query=(
                        "格力电器 2024年度 合并利润表 "
                        "综合收益总额"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id=(
                        "other_comprehensive_income_net_of_tax"
                    ),
                    statement_type="income_statement",
                    pdf_page=122,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "其他综合收益的税后净额"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="total_comprehensive_income",
                    statement_type="income_statement",
                    pdf_page=123,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "综合收益总额"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较两家公司其他综合收益的"
                        "税后净额。"
                    ),
                    "query_numbers": [1, 3],
                    "output_ref": (
                        "comparison_complex_036_oci"
                    ),
                },
                {
                    "action": "compare",
                    "description": (
                        "比较两家公司综合收益总额。"
                    ),
                    "query_numbers": [2, 4],
                    "output_ref": (
                        "comparison_complex_036_total_income"
                    ),
                },
            ],
            "answer_text": (
                "格力电器2024年合并口径其他综合收益的"
                "税后净额为180,264,674.95元，综合收益总额"
                "为32,551,333,962.30元；海尔智家其他综合"
                "收益的税后净额为-1,173,713,256.17元，"
                "综合收益总额为18,401,899,245.51元。"
                "两项指标均为格力电器更高。"
            ),
            "conclusion": (
                "格力电器2024年其他综合收益税后净额和"
                "综合收益总额均高于海尔智家。"
            ),
        },
        {
            "case_id": "complex_037",
            "question_type": (
                "single_company_multi_metric"
            ),
            "difficulty": "hard",
            "question": (
                "海尔智家2024年末合并口径资产总计和"
                "非流动负债合计，以及2024年合并口径"
                "营业收入和净利润分别是多少？"
            ),
            "normalized_question": (
                "查询海尔智家2024年末合并资产负债表中的"
                "资产总计和非流动负债合计，以及2024年度"
                "合并利润表中的营业收入和净利润。"
            ),
            "items": [
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="total_assets",
                    statement_type="balance_sheet",
                    pdf_page=118,
                    semantic_query=(
                        "海尔智家 2024年12月31日 "
                        "合并资产负债表 资产总计"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="revenue",
                    statement_type="income_statement",
                    pdf_page=121,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "营业收入"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="net_profit",
                    statement_type="income_statement",
                    pdf_page=122,
                    semantic_query=(
                        "海尔智家 2024年度 合并利润表 "
                        "净利润"
                    ),
                ),
                atomic_query(
                    company_id="haier_smart_home",
                    metric_id="non_current_liabilities",
                    statement_type="balance_sheet",
                    pdf_page=119,
                    semantic_query=(
                        "海尔智家 2024年12月31日 "
                        "合并资产负债表 非流动负债合计"
                    ),
                ),
            ],
            "operations": [],
            "answer_text": (
                "海尔智家2024年末合并口径资产总计为"
                "290,113,822,824.61元，非流动负债合计为"
                "22,153,482,887.26元；2024年合并口径"
                "营业收入为285,981,225,203.93元，"
                "净利润为19,575,612,501.68元。"
            ),
            "conclusion": (
                "海尔智家四项指标分别来自2024年末"
                "合并资产负债表和2024年度合并利润表。"
            ),
        },
        {
            "case_id": "complex_038",
            "question_type": (
                "cross_company_comparison"
            ),
            "difficulty": "hard",
            "question": (
                "比较美的集团和格力电器2024年合并口径"
                "投资活动及筹资活动产生的现金流量净额，"
                "分别判断哪家公司更高，并说明净流入"
                "或净流出。"
            ),
            "normalized_question": (
                "分别查询美的集团和格力电器2024年度"
                "合并现金流量表中的投资活动及筹资活动"
                "现金流量净额，比较相同指标并根据正负号"
                "判断现金流方向。"
            ),
            "items": [
                atomic_query(
                    company_id="midea_group",
                    metric_id=(
                        "net_cash_flow_from_"
                        "investing_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=160,
                    semantic_query=(
                        "美的集团 2024年度 合并现金流量表 "
                        "投资活动使用的现金流量净额"
                    ),
                ),
                atomic_query(
                    company_id="midea_group",
                    metric_id=(
                        "net_cash_flow_from_"
                        "financing_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=160,
                    semantic_query=(
                        "美的集团 2024年度 合并现金流量表 "
                        "筹资活动产生或使用的现金流量净额"
                    ),
                ),
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "net_cash_flow_from_"
                        "investing_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=114,
                    semantic_query=(
                        "格力电器 2024年度 合并现金流量表 "
                        "投资活动产生的现金流量净额"
                    ),
                ),
                atomic_query(
                    company_id="gree_electric",
                    metric_id=(
                        "net_cash_flow_from_"
                        "financing_activities"
                    ),
                    statement_type="cash_flow_statement",
                    pdf_page=114,
                    semantic_query=(
                        "格力电器 2024年度 合并现金流量表 "
                        "筹资活动产生的现金流量净额"
                    ),
                ),
            ],
            "operations": [
                {
                    "action": "compare",
                    "description": (
                        "比较两家公司投资活动产生的"
                        "现金流量净额及现金流方向。"
                    ),
                    "query_numbers": [1, 3],
                    "output_ref": (
                        "comparison_complex_038_investing"
                    ),
                },
                {
                    "action": "compare",
                    "description": (
                        "比较两家公司筹资活动产生的"
                        "现金流量净额及现金流方向。"
                    ),
                    "query_numbers": [2, 4],
                    "output_ref": (
                        "comparison_complex_038_financing"
                    ),
                },
            ],
            "answer_text": (
                "美的集团2024年合并口径投资活动产生的"
                "现金流量净额为-87,901,802,000元，为净流出；"
                "格力电器为-15,557,909,615.57元，为净流出，"
                "格力电器更高。美的集团筹资活动产生的"
                "现金流量净额为22,697,954,000元，为净流入；"
                "格力电器为-23,703,212,908.16元，为净流出，"
                "美的集团更高。"
            ),
            "conclusion": (
                "投资活动现金流量净额为格力电器更高，"
                "筹资活动现金流量净额为美的集团更高。"
            ),
        },
    ]


def build_case(
    definition: dict[str, Any],
    timestamp: str,
) -> ComplexFinancialEvalCase:
    case_id = definition["case_id"]
    question = definition["question"]
    items = definition["items"]

    fact_ids = [
        item["fact_id"]
        for item in items
    ]

    evidence_ids = [
        item["evidence_id"]
        for item in items
    ]

    company_ids = ordered_unique(
        [
            item["company_id"]
            for item in items
        ]
    )

    report_ids = ordered_unique(
        [
            item["report_id"]
            for item in items
        ]
    )

    retrieval_queries: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    producer_step_by_output: dict[str, str] = {}

    for index, item in enumerate(
        items,
        start=1,
    ):
        query_id = f"q{index}"
        step_id = f"s{len(steps) + 1}"

        retrieval_queries.append(
            {
                "query_id": query_id,
                "target_fact_id": item["fact_id"],
                "baseline_query": question,
                "semantic_query": (
                    item["semantic_query"]
                ),
                "company_id": item["company_id"],
                "report_id": item["report_id"],
                "metric_id": item["metric_id"],
                "fiscal_year": item["fiscal_year"],
                "report_type": "annual_report",
                "statement_type": (
                    item["statement_type"]
                ),
                "statement_scope": (
                    item["statement_scope"]
                ),
                "gold_pdf_pages": [
                    item["pdf_page"],
                ],
            }
        )

        steps.append(
            {
                "step_id": step_id,
                "action": "retrieve",
                "description": (
                    f"检索原子事实：{item['semantic_query']}"
                ),
                "target_fact_ids": [
                    item["fact_id"],
                ],
                "input_refs": [],
                "depends_on": [],
                "output_ref": item["fact_id"],
            }
        )

        producer_step_by_output[
            item["fact_id"]
        ] = step_id

    operation_outputs: list[str] = []

    for operation in definition["operations"]:
        input_refs = [
            fact_ids[number - 1]
            for number
            in operation["query_numbers"]
        ]

        depends_on = ordered_unique(
            [
                producer_step_by_output[
                    input_ref
                ]
                for input_ref in input_refs
            ]
        )

        step_id = f"s{len(steps) + 1}"
        output_ref = operation["output_ref"]

        steps.append(
            {
                "step_id": step_id,
                "action": operation["action"],
                "description": (
                    operation["description"]
                ),
                "target_fact_ids": [],
                "input_refs": input_refs,
                "depends_on": depends_on,
                "output_ref": output_ref,
            }
        )

        producer_step_by_output[
            output_ref
        ] = step_id

        operation_outputs.append(
            output_ref
        )

    final_input_refs = ordered_unique(
        fact_ids + operation_outputs
    )

    final_dependencies = ordered_unique(
        [
            producer_step_by_output[input_ref]
            for input_ref in final_input_refs
        ]
    )

    final_step_id = f"s{len(steps) + 1}"

    steps.append(
        {
            "step_id": final_step_id,
            "action": "synthesize",
            "description": (
                "汇总全部已检索事实和比较结果，"
                "生成带有明确数值与结论的最终答案。"
            ),
            "target_fact_ids": [],
            "input_refs": final_input_refs,
            "depends_on": final_dependencies,
            "output_ref": f"answer_{case_id}",
        }
    )

    data = {
        "schema_version": 1,
        "case_id": case_id,
        "question": question,
        "question_type": (
            definition["question_type"]
        ),
        "difficulty": definition["difficulty"],
        "company_ids": company_ids,
        "report_ids": report_ids,
        "fiscal_years": [2024],
        "gold_rewrite": {
            "normalized_question": (
                definition["normalized_question"]
            ),
            "resolved_aliases": [],
            "retrieval_queries": (
                retrieval_queries
            ),
        },
        "gold_plan": {
            "steps": steps,
            "final_step_id": final_step_id,
        },
        "gold_fact_ids": fact_ids,
        "gold_evidence_ids": evidence_ids,
        "gold_calculation_ids": [],
        "gold_answer": {
            "answer_text": (
                definition["answer_text"]
            ),
            "conclusion": (
                definition["conclusion"]
            ),
            "supporting_fact_ids": fact_ids,
            "evidence_ids": evidence_ids,
            "supporting_calculation_ids": [],
        },
        "validation_status": "pending",
        "validated_by": None,
        "validated_at": None,
        "source_version": SOURCE_VERSION,
        "created_at": timestamp,
        "updated_at": timestamp,
        "review_notes": (
            "Diagnostic Dev v1：底层 Evidence 和 "
            "FinancialFact 已人工核验；当前 Case 等待"
            "Gold Rewrite、Gold Plan 和 Gold Answer "
            "最终人工语义核验。"
        ),
    }

    return (
        ComplexFinancialEvalCase
        .model_validate(data)
    )


TARGET_QUERY_KEYS = {
    ("complex_031", "q1"): (
        "fact_midea_group_2024_non_current_assets"
    ),
    ("complex_032", "q2"): (
        "fact_gree_electric_2024_"
        "cash_outflows_from_investing_activities_subtotal"
    ),
    ("complex_033", "q2"): (
        "fact_haier_smart_home_2024_taxes_and_surcharges"
    ),
    ("complex_034", "q2"): (
        "fact_hisense_home_2024_tax_refunds_received"
    ),
    ("complex_035", "q2"): (
        "fact_gree_electric_2024_"
        "net_profit_attributable_to_parent"
    ),
    ("complex_036", "q4"): (
        "fact_haier_smart_home_2024_"
        "total_comprehensive_income"
    ),
    ("complex_037", "q4"): (
        "fact_haier_smart_home_2024_"
        "non_current_liabilities"
    ),
    ("complex_038", "q2"): (
        "fact_midea_group_2024_"
        "net_cash_flow_from_financing_activities"
    ),
}


DIAGNOSTIC_CATEGORIES = {
    "complex_031": "same_page_sibling",
    "complex_032": "same_page_sibling",
    "complex_033": "adjacent_page_continued_table",
    "complex_034": "adjacent_page_continued_table",
    "complex_035": "confusable_labels",
    "complex_036": "confusable_labels_and_adjacent_page",
    "complex_037": "page_window_coverage_gap",
    "complex_038": "multi_fact_structural_retrieval",
}


NEW_SOURCE_FACT_IDS = {
    "fact_midea_group_2024_non_current_assets",
    (
        "fact_gree_electric_2024_"
        "cash_outflows_from_investing_activities_subtotal"
    ),
    "fact_haier_smart_home_2024_taxes_and_surcharges",
    (
        "fact_hisense_home_2024_"
        "cash_received_from_sales_of_goods_"
        "and_rendering_of_services"
    ),
    "fact_hisense_home_2024_tax_refunds_received",
    (
        "fact_hisense_home_2024_"
        "net_cash_flow_from_operating_activities"
    ),
    (
        "fact_hisense_home_2024_"
        "net_cash_flow_from_investing_activities"
    ),
    (
        "fact_gree_electric_2024_"
        "net_profit_attributable_to_parent"
    ),
    (
        "fact_gree_electric_2024_"
        "other_comprehensive_income_net_of_tax"
    ),
    (
        "fact_gree_electric_2024_"
        "total_comprehensive_income"
    ),
    (
        "fact_haier_smart_home_2024_"
        "other_comprehensive_income_net_of_tax"
    ),
    (
        "fact_haier_smart_home_2024_"
        "total_comprehensive_income"
    ),
    (
        "fact_haier_smart_home_2024_"
        "non_current_liabilities"
    ),
    (
        "fact_midea_group_2024_"
        "net_cash_flow_from_investing_activities"
    ),
    (
        "fact_midea_group_2024_"
        "net_cash_flow_from_financing_activities"
    ),
    (
        "fact_gree_electric_2024_"
        "net_cash_flow_from_investing_activities"
    ),
    (
        "fact_gree_electric_2024_"
        "net_cash_flow_from_financing_activities"
    ),
}


def validate_registry(
    registry_bundle: Any,
) -> None:
    if len(registry_bundle.evidences) != 78:
        raise SystemExit(
            "evidence_count_not_78="
            f"{len(registry_bundle.evidences)}"
        )

    if len(registry_bundle.financial_facts) != 78:
        raise SystemExit(
            "financial_fact_count_not_78="
            f"{len(registry_bundle.financial_facts)}"
        )

    for fact_id in sorted(
        NEW_SOURCE_FACT_IDS
    ):
        fact = (
            registry_bundle
            .financial_facts
            .get(fact_id)
        )

        if fact is None:
            raise SystemExit(
                f"missing_source_fact={fact_id}"
            )

        if (
            fact.validation_status
            != ValidationStatus.VERIFIED
        ):
            raise SystemExit(
                f"source_fact_not_verified={fact_id}"
            )

        evidence_id = (
            fact.primary_evidence_id
        )

        evidence = (
            registry_bundle
            .evidences
            .get(evidence_id)
        )

        if evidence is None:
            raise SystemExit(
                f"missing_source_evidence={evidence_id}"
            )

        if (
            evidence.validation_status
            != ValidationStatus.VERIFIED
        ):
            raise SystemExit(
                "source_evidence_not_verified="
                f"{evidence_id}"
            )


def validate_cases(
    *,
    cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ],
    dev_cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ],
    test_cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ],
) -> dict[str, Any]:
    expected_case_ids = [
        f"complex_{number:03d}"
        for number in range(31, 39)
    ]

    actual_case_ids = [
        case.case_id
        for case in cases
    ]

    if actual_case_ids != expected_case_ids:
        raise SystemExit(
            f"unexpected_case_ids={actual_case_ids}"
        )

    if len(cases) != 8:
        raise SystemExit(
            f"case_count_not_8={len(cases)}"
        )

    query_count = sum(
        len(
            case.gold_rewrite.retrieval_queries
        )
        for case in cases
    )

    if query_count != 24:
        raise SystemExit(
            f"query_count_not_24={query_count}"
        )

    calculation_count = sum(
        len(case.gold_calculation_ids)
        for case in cases
    )

    if calculation_count != 0:
        raise SystemExit(
            "unexpected_calculation_count="
            f"{calculation_count}"
        )

    status_counts = Counter(
        case.validation_status.value
        for case in cases
    )

    if status_counts != Counter(
        {"pending": 8}
    ):
        raise SystemExit(
            f"unexpected_status_counts={dict(status_counts)}"
        )

    type_difficulty_counts = Counter(
        (
            case.question_type,
            case.difficulty,
        )
        for case in cases
    )

    expected_distribution = Counter(
        {
            (
                "single_company_multi_metric",
                "medium",
            ): 4,
            (
                "single_company_multi_metric",
                "hard",
            ): 2,
            (
                "cross_company_comparison",
                "hard",
            ): 2,
        }
    )

    if (
        type_difficulty_counts
        != expected_distribution
    ):
        raise SystemExit(
            "unexpected_type_difficulty_counts="
            f"{dict(type_difficulty_counts)}"
        )

    fact_occurrences = [
        fact_id
        for case in cases
        for fact_id in case.gold_fact_ids
    ]

    evidence_occurrences = [
        evidence_id
        for case in cases
        for evidence_id
        in case.gold_evidence_ids
    ]

    if len(fact_occurrences) != 24:
        raise SystemExit(
            "fact_reference_count_not_24="
            f"{len(fact_occurrences)}"
        )

    if len(evidence_occurrences) != 24:
        raise SystemExit(
            "evidence_reference_count_not_24="
            f"{len(evidence_occurrences)}"
        )

    if len(set(fact_occurrences)) != 23:
        raise SystemExit(
            "unique_fact_reference_count_not_23="
            f"{len(set(fact_occurrences))}"
        )

    query_by_key = {
        (
            case.case_id,
            query.query_id,
        ): query
        for case in cases
        for query
        in case.gold_rewrite.retrieval_queries
    }

    for key, expected_fact_id in (
        TARGET_QUERY_KEYS.items()
    ):
        query = query_by_key.get(key)

        if query is None:
            raise SystemExit(
                f"missing_target_query={key}"
            )

        if (
            query.target_fact_id
            != expected_fact_id
        ):
            raise SystemExit(
                f"wrong_target_fact={key}:"
                f"{query.target_fact_id}"
            )

    existing_case_ids = {
        case.case_id
        for case in dev_cases + test_cases
    }

    overlapping_case_ids = sorted(
        set(actual_case_ids)
        & existing_case_ids
    )

    if overlapping_case_ids:
        raise SystemExit(
            "existing_case_id_overlap="
            + ",".join(overlapping_case_ids)
        )

    existing_questions = {
        case.question.strip()
        for case in dev_cases + test_cases
    }

    duplicated_questions = sorted(
        case.case_id
        for case in cases
        if case.question.strip()
        in existing_questions
    )

    if duplicated_questions:
        raise SystemExit(
            "existing_question_duplicates="
            + ",".join(duplicated_questions)
        )

    frozen_test_fact_ids = {
        fact_id
        for case in test_cases
        for fact_id in case.gold_fact_ids
    }

    frozen_test_evidence_ids = {
        evidence_id
        for case in test_cases
        for evidence_id
        in case.gold_evidence_ids
    }

    diagnostic_target_fact_ids = set(
        TARGET_QUERY_KEYS.values()
    )

    diagnostic_target_evidence_ids = {
        fact_id.replace(
            "fact_",
            "evidence_",
            1,
        )
        for fact_id
        in diagnostic_target_fact_ids
    }

    target_fact_overlap = sorted(
        diagnostic_target_fact_ids
        & frozen_test_fact_ids
    )

    target_evidence_overlap = sorted(
        diagnostic_target_evidence_ids
        & frozen_test_evidence_ids
    )

    if target_fact_overlap:
        raise SystemExit(
            "frozen_test_target_fact_overlap="
            + ",".join(target_fact_overlap)
        )

    if target_evidence_overlap:
        raise SystemExit(
            "frozen_test_target_evidence_overlap="
            + ",".join(target_evidence_overlap)
        )

    return {
        "query_count": query_count,
        "calculation_count": (
            calculation_count
        ),
        "status_counts": dict(
            sorted(status_counts.items())
        ),
        "type_difficulty_counts": {
            f"{key[0]}:{key[1]}": value
            for key, value in sorted(
                type_difficulty_counts.items()
            )
        },
        "fact_reference_count": (
            len(fact_occurrences)
        ),
        "unique_fact_reference_count": (
            len(set(fact_occurrences))
        ),
        "evidence_reference_count": (
            len(evidence_occurrences)
        ),
        "diagnostic_target_count": (
            len(diagnostic_target_fact_ids)
        ),
        "frozen_test_target_fact_overlap": 0,
        "frozen_test_target_evidence_overlap": 0,
    }


def write_dataset_safely(
    cases: tuple[
        ComplexFinancialEvalCase,
        ...,
    ],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_text = "\n".join(
        case.model_dump_json()
        for case in cases
    ) + "\n"

    temporary_path = (
        OUTPUT_PATH.with_suffix(
            ".jsonl.tmp"
        )
    )

    temporary_path.write_text(
        output_text,
        encoding="utf-8",
    )

    loaded_temporary_cases = (
        load_complex_financial_eval_cases(
            temporary_path
        )
    )

    if loaded_temporary_cases != cases:
        temporary_path.unlink(
            missing_ok=True
        )
        raise SystemExit(
            "temporary_dataset_reload_mismatch=true"
        )

    if OUTPUT_PATH.exists():
        existing_text = OUTPUT_PATH.read_text(
            encoding="utf-8"
        )

        if existing_text != output_text:
            backup_path = (
                OUTPUT_PATH.with_name(
                    "complex_plan_diagnostic_dev_v1"
                    ".before_rebuild.jsonl"
                )
            )

            if not backup_path.exists():
                shutil.copy2(
                    OUTPUT_PATH,
                    backup_path,
                )

            print(
                f"existing_dataset_backup={backup_path}"
            )

    temporary_path.replace(
        OUTPUT_PATH
    )


def main() -> None:
    if not DEV_V2_PATH.is_file():
        raise SystemExit(
            f"missing_dev_v2={DEV_V2_PATH}"
        )

    if not FROZEN_TEST_PATH.is_file():
        raise SystemExit(
            f"missing_frozen_test={FROZEN_TEST_PATH}"
        )

    dev_hash_before = sha256_file(
        DEV_V2_PATH
    )

    test_hash_before = sha256_file(
        FROZEN_TEST_PATH
    )

    (
        registry_bundle,
        _page_mappings,
        _metric_aliases,
        fact_evidence_links,
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

    validate_registry(
        registry_bundle
    )

    if len(fact_evidence_links) != 78:
        raise SystemExit(
            "fact_evidence_link_count_not_78="
            f"{len(fact_evidence_links)}"
        )

    timestamp = datetime.now(
        CHINA_TIMEZONE
    ).isoformat()

    definitions = build_definitions()

    cases = tuple(
        build_case(
            definition,
            timestamp,
        )
        for definition in definitions
    )

    dev_cases = (
        load_complex_financial_eval_cases(
            DEV_V2_PATH
        )
    )

    test_cases = (
        load_complex_financial_eval_cases(
            FROZEN_TEST_PATH
        )
    )

    summary = validate_cases(
        cases=cases,
        dev_cases=dev_cases,
        test_cases=test_cases,
    )

    validate_complex_plan_eval_integrity(
        cases=cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )

    write_dataset_safely(
        cases
    )

    reloaded_cases = (
        load_complex_financial_eval_cases(
            OUTPUT_PATH
        )
    )

    validate_cases(
        cases=reloaded_cases,
        dev_cases=dev_cases,
        test_cases=test_cases,
    )

    validate_complex_plan_eval_integrity(
        cases=reloaded_cases,
        calculations=[],
        registry_bundle=registry_bundle,
    )

    dev_hash_after = sha256_file(
        DEV_V2_PATH
    )

    test_hash_after = sha256_file(
        FROZEN_TEST_PATH
    )

    if dev_hash_after != dev_hash_before:
        raise SystemExit(
            "complex_plan_dev_v2_was_modified=true"
        )

    if test_hash_after != test_hash_before:
        raise SystemExit(
            "complex_plan_test_v1_was_modified=true"
        )

    output_hash = sha256_file(
        OUTPUT_PATH
    )

    target_records = []

    query_by_key = {
        (
            case.case_id,
            query.query_id,
        ): query
        for case in reloaded_cases
        for query
        in case.gold_rewrite.retrieval_queries
    }

    for (
        case_id,
        query_id,
    ), fact_id in sorted(
        TARGET_QUERY_KEYS.items()
    ):
        target_records.append(
            {
                "case_id": case_id,
                "query_id": query_id,
                "diagnostic_category": (
                    DIAGNOSTIC_CATEGORIES[
                        case_id
                    ]
                ),
                "target_fact_id": fact_id,
                "target_evidence_id": (
                    fact_id.replace(
                        "fact_",
                        "evidence_",
                        1,
                    )
                ),
                "semantic_query": (
                    query_by_key[
                        (case_id, query_id)
                    ].semantic_query
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_id": (
            "complex_plan_diagnostic_dev_v1"
        ),
        "split": "diagnostic_dev",
        "source_version": SOURCE_VERSION,
        "created_at": timestamp,
        "case_count": len(
            reloaded_cases
        ),
        "case_ids": [
            case.case_id
            for case in reloaded_cases
        ],
        **summary,
        "diagnostic_targets": (
            target_records
        ),
        "source_registry": {
            "evidence_count": (
                len(registry_bundle.evidences)
            ),
            "financial_fact_count": (
                len(
                    registry_bundle
                    .financial_facts
                )
            ),
            "fact_evidence_link_count": (
                len(fact_evidence_links)
            ),
            "new_verified_source_count": (
                len(NEW_SOURCE_FACT_IDS)
            ),
        },
        "frozen_inputs": {
            "complex_plan_dev_v2_sha256": (
                dev_hash_after
            ),
            "complex_plan_test_v1_sha256": (
                test_hash_after
            ),
        },
        "dataset_sha256": output_hash,
        "manual_semantic_review_required": True,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 80)
    print(f"output_path={OUTPUT_PATH}")
    print(f"manifest_path={MANIFEST_PATH}")
    print(f"case_count={len(reloaded_cases)}")
    print(
        "case_ids="
        f"{[case.case_id for case in reloaded_cases]}"
    )
    print(
        "planned_query_count="
        f"{summary['query_count']}"
    )
    print(
        "fact_reference_count="
        f"{summary['fact_reference_count']}"
    )
    print(
        "unique_fact_reference_count="
        f"{summary['unique_fact_reference_count']}"
    )
    print(
        "evidence_reference_count="
        f"{summary['evidence_reference_count']}"
    )
    print(
        "diagnostic_target_count="
        f"{summary['diagnostic_target_count']}"
    )
    print(
        "calculation_count="
        f"{summary['calculation_count']}"
    )
    print(
        "status_counts="
        f"{summary['status_counts']}"
    )
    print(
        "type_difficulty_counts="
        f"{summary['type_difficulty_counts']}"
    )
    print(
        "registry_evidence_count="
        f"{len(registry_bundle.evidences)}"
    )
    print(
        "registry_financial_fact_count="
        f"{len(registry_bundle.financial_facts)}"
    )
    print(
        "fact_evidence_link_count="
        f"{len(fact_evidence_links)}"
    )
    print(
        "frozen_test_target_fact_overlap=0"
    )
    print(
        "frozen_test_target_evidence_overlap=0"
    )
    print("dev_v2_preserved=true")
    print("test_v1_preserved=true")
    print(
        "complex_diagnostic_dev_integrity_passed=true"
    )
    print(
        "complex_plan_diagnostic_dev_v1_created=true"
    )
    print(
        "manual_semantic_review_required=true"
    )


if __name__ == "__main__":
    main()