from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.evidence_context import (
    assess_financial_fact_evidence,
    build_evidence_context,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
    RetrievalEvalResult,
)
from app.services.registry_loader import (
    load_reports,
)


class EvidenceInspectionError(ValueError):
    """真实 Evidence Context 验收基础异常。"""


def _read_non_empty_lines(
    path: Path,
) -> tuple[str, ...]:
    if not path.is_file():
        raise EvidenceInspectionError(
            f"文件不存在：{path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise EvidenceInspectionError(
            f"无法读取文件：{path}"
        ) from exc

    lines = tuple(
        line
        for line in text.splitlines()
        if line.strip()
    )

    if not lines:
        raise EvidenceInspectionError(
            f"文件没有有效记录：{path}"
        )

    return lines


def _load_case(
    *,
    path: Path,
    case_id: str,
) -> FinancialFactRetrievalEvalCase:
    cases = tuple(
        FinancialFactRetrievalEvalCase
        .model_validate_json(line)
        for line in _read_non_empty_lines(
            path
        )
    )

    matches = tuple(
        case
        for case in cases
        if case.case_id == case_id
    )

    if len(matches) != 1:
        raise EvidenceInspectionError(
            "评测集中必须恰好存在一条 "
            f"case_id={case_id!r}，"
            f"实际数量={len(matches)}"
        )

    return matches[0]


def _load_result(
    *,
    path: Path,
    case_id: str,
) -> RetrievalEvalResult:
    results = tuple(
        RetrievalEvalResult
        .model_validate_json(line)
        for line in _read_non_empty_lines(
            path
        )
    )

    matches = tuple(
        result
        for result in results
        if result.case_id == case_id
    )

    if len(matches) != 1:
        raise EvidenceInspectionError(
            "评测结果中必须恰好存在一条 "
            f"case_id={case_id!r}，"
            f"实际数量={len(matches)}"
        )

    return matches[0]


def _validate_case_and_result(
    *,
    case: FinancialFactRetrievalEvalCase,
    result: RetrievalEvalResult,
) -> None:
    if result.question != case.question:
        raise EvidenceInspectionError(
            "评测题和评测结果的 question 不一致"
        )

    if result.company_id != case.company_id:
        raise EvidenceInspectionError(
            "评测题和评测结果的 company_id 不一致"
        )

    if result.report_id != case.report_id:
        raise EvidenceInspectionError(
            "评测题和评测结果的 report_id 不一致"
        )

    if result.fiscal_year != case.fiscal_year:
        raise EvidenceInspectionError(
            "评测题和评测结果的 fiscal_year 不一致"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "使用真实检索结果构建 Evidence Context，"
            "并检查是否可以进入回答生成"
        )
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reports-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--case-id",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-hits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
    )

    args = parser.parse_args()

    case = _load_case(
        path=args.cases_path,
        case_id=args.case_id,
    )

    result = _load_result(
        path=args.results_path,
        case_id=args.case_id,
    )

    _validate_case_and_result(
        case=case,
        result=result,
    )

    report_registry, _ = load_reports(
        args.reports_path
    )

    report = report_registry.require(
        case.report_id
    )

    plan = build_financial_fact_query_plan(
        original_query=case.question,
        metric_name=case.metric_name,
        fiscal_year=case.fiscal_year,
        company_id=case.company_id,
        report_id=case.report_id,
        report_type=case.report_type,
        statement_type=(
            case.statement_type
        ),
        statement_scope=(
            case.statement_scope
        ),
    )

    context = build_evidence_context(
        report=report,
        plan=plan,
        hits=result.top_hits,
        max_hits=args.max_hits,
        max_chars=args.max_chars,
    )

    decision = (
        assess_financial_fact_evidence(
            plan=plan,
            context=context,
        )
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_path.write_text(
        decision.model_dump_json(
            indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"case_id={case.case_id}"
    )

    print(
        f"question={case.question}"
    )

    print(
        f"metric_name={case.metric_name}"
    )

    print(
        "retrieval_first_relevant_rank="
        f"{result.first_relevant_rank}"
    )

    print(
        f"context_item_count={len(context.items)}"
    )

    print(
        f"context_used_chars={context.used_chars}"
    )

    print(
        "context_truncated="
        f"{context.truncated}"
    )

    print(
        f"status={decision.status}"
    )

    print(
        "supporting_citation_ids="
        f"{decision.supporting_citation_ids}"
    )

    print(
        f"reason={decision.reason}"
    )

    print(
        f"output_path={args.output_path}"
    )

    for item in context.items:
        citation = item.citation

        preview = item.text.replace(
            "\n",
            " ",
        )[:300]

        is_supporting = (
            citation.citation_id
            in decision.supporting_citation_ids
        )

        print("-" * 80)

        print(
            "citation_id="
            f"{citation.citation_id}"
        )

        print(
            f"supporting={is_supporting}"
        )

        print(
            f"rank={citation.retrieval_rank}"
        )

        print(
            f"score={citation.retrieval_score:.6f}"
        )

        print(
            f"pdf_page={citation.pdf_page}"
        )

        print(
            "printed_page="
            f"{citation.printed_page}"
        )

        print(
            f"chunk_id={citation.chunk_id}"
        )

        print(
            f"text={preview}"
        )


if __name__ == "__main__":
    main()