from __future__ import annotations

import argparse
from pathlib import Path

from app.rag.answer_control import (
    build_financial_fact_answer_packet,
)
from app.rag.query_planning import (
    build_financial_fact_query_plan,
)
from app.schemas.evidence_context import (
    EvidenceReadinessResult,
)
from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
)


class AnswerPacketBuildError(ValueError):
    """真实 Answer Packet 构建异常。"""


def _load_case(
    *,
    path: Path,
    case_id: str,
) -> FinancialFactRetrievalEvalCase:
    if not path.is_file():
        raise AnswerPacketBuildError(
            f"评测集文件不存在：{path}"
        )

    cases = tuple(
        FinancialFactRetrievalEvalCase
        .model_validate_json(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )

    matches = tuple(
        case
        for case in cases
        if case.case_id == case_id
    )

    if len(matches) != 1:
        raise AnswerPacketBuildError(
            "必须恰好找到一条评测题，"
            f"实际数量={len(matches)}"
        )

    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "根据真实 Evidence Readiness "
            "构造回答或拒答控制结果"
        )
    )

    parser.add_argument(
        "--cases-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--readiness-path",
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

    args = parser.parse_args()

    case = _load_case(
        path=args.cases_path,
        case_id=args.case_id,
    )

    readiness = (
        EvidenceReadinessResult
        .model_validate_json(
            args.readiness_path.read_bytes()
        )
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

    packet = (
        build_financial_fact_answer_packet(
            plan=plan,
            readiness=readiness,
        )
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_path.write_text(
        packet.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"case_id={case.case_id}")
    print(f"status={packet.status}")
    print(f"action={packet.action}")
    print(
        "supporting_citation_ids="
        f"{packet.supporting_citation_ids}"
    )
    print(
        "used_chunk_ids="
        f"{packet.used_chunk_ids}"
    )
    print(
        "generation_context_chars="
        f"{len(packet.generation_context)}"
    )
    print(f"message={packet.message}")
    print(
        f"output_path={args.output_path}"
    )


if __name__ == "__main__":
    main()