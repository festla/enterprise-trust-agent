from __future__ import annotations

import argparse
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_lookup import (
    solve_excel_table_lookup,
)
from app.services.competition_excel_parser import (
    parse_competition_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qa",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--attachments",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--case-id",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_competition_qa_excel(
        args.qa
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    case = case_by_id[
        args.case_id
    ]

    if case.qa_type != "表格取数":
        raise SystemExit(
            f"{case.case_id} "
            "不是表格取数题"
        )

    manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    resolver = CompetitionSourceResolver(
        manifest
    )

    resolution = resolver.resolve(
        case
    )

    source_by_id = {
        source.source_id: source
        for source in manifest
    }

    source = source_by_id[
        resolution.source_id
    ]

    workbook = (
        parse_competition_excel(
            attachments_root=(
                args.attachments
            ),
            source=source,
        )
    )

    question = (
        build_competition_question(
            case
        )
    )

    result = (
        solve_excel_table_lookup(
            question=question,
            workbook=workbook,
        )
    )

    print(
        "=== Excel Table Lookup ==="
    )

    print(
        "Case:",
        case.case_id,
    )

    print(
        "Question:",
        question.question,
    )

    print(
        "Predicted option:",
        result.answer_option,
    )

    print(
        "Predicted value:",
        result.answer_text,
    )

    print(
        "Confidence:",
        result.confidence,
    )

    print(
        "Sheet:",
        result.evidence.sheet_name,
    )

    print(
        "Value cell:",
        result.evidence.value_coordinate,
    )

    print(
        "Row label:",
        (
            result.evidence
            .row_label_coordinate
        ),
        result.evidence.row_label,
    )

    print(
        "Column label:",
        (
            result.evidence
            .column_label_coordinate
        ),
        result.evidence.column_label,
    )

    # ========================================================
    # 仅用于本地开发诊断。
    #
    # Gold 不进入 Solver。
    # Solver 完成后才比较答案。
    # ========================================================

    print()
    print(
        "Gold option:",
        case.answer,
    )

    print(
        "Gold answer:",
        case.answer_text,
    )

    print(
        "Correct:",
        (
            result.answer_option
            == case.answer
        ),
    )


if __name__ == "__main__":
    main()