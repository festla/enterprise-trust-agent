from pathlib import Path

import pytest

from app.services.retrieval_eval_dataset import (
    InvalidRetrievalEvalDatasetError,
    load_financial_fact_retrieval_cases,
)


VALID_CASE = (
    '{"schema_version":1,'
    '"case_id":"fact_001",'
    '"question":"营业收入是多少？",'
    '"metric_name":"营业收入",'
    '"company_id":"midea_group",'
    '"report_id":"midea_group_2024",'
    '"fiscal_year":2024,'
    '"report_type":"annual_report",'
    '"statement_type":"income_statement",'
    '"statement_scope":"consolidated",'
    '"gold_pdf_pages":[158]}'
)


def test_load_retrieval_eval_cases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.jsonl"

    path.write_text(
        VALID_CASE + "\n",
        encoding="utf-8",
    )

    cases = (
        load_financial_fact_retrieval_cases(
            path
        )
    )

    assert len(cases) == 1
    assert cases[0].case_id == "fact_001"


def test_reject_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.jsonl"

    path.write_text(
        VALID_CASE
        + "\n"
        + VALID_CASE
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidRetrievalEvalDatasetError,
        match="重复 case_id",
    ):
        load_financial_fact_retrieval_cases(
            path
        )