from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from app.services.competition_dataset import (
    load_competition_qa_excel,
)
from app.services.competition_excel_parser import (
    parse_competition_excel,
)
from app.services.competition_source_resolver import (
    CompetitionSourceResolver,
    build_competition_source_manifest,
)


def _normalize_text(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    return value


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
        "--split",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_competition_qa_excel(
        args.qa
    )

    split = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_ids = set(
        split["dev_case_ids"]
    )

    calculation_cases = [
        case
        for case in cases
        if (
            case.case_id in dev_ids
            and case.source_type
            == "excel"
            and case.qa_type
            == "表格计算"
        )
    ]

    manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    resolver = CompetitionSourceResolver(
        manifest
    )

    source_by_id = {
        source.source_id: source
        for source in manifest
    }

    print(
        "=== Dev Excel Calculation Cases ==="
    )

    print(
        "Count:",
        len(calculation_cases),
    )

    for index, case in enumerate(
        calculation_cases,
        start=1,
    ):
        resolution = resolver.resolve(
            case
        )

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

        print()
        print(
            "=" * 72
        )

        print(
            f"[{index}/"
            f"{len(calculation_cases)}] "
            f"{case.case_id}"
        )

        print(
            "Source:",
            source.relative_path,
        )

        print(
            "Sheets:",
            workbook.sheet_names,
        )

        print()

        print(
            "Question:",
            case.question,
        )

        print()

        print(
            "A:",
            case.option_a,
        )

        print(
            "B:",
            case.option_b,
        )

        print(
            "C:",
            case.option_c,
        )

        print(
            "D:",
            case.option_d,
        )

        # ================================================
        # Dev diagnostic only.
        # Gold 不会进入后续 Solver。
        # ================================================

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
            "Gold evidence:",
            case.evidence,
        )

        print()
        print(
            "Question-mentioned cells:"
        )

        question_key = _normalize_text(
            case.question
        )

        found = 0

        for sheet in workbook.sheets:
            for cell in sheet.cells:
                if (
                    cell.cell_type
                    != "text"
                ):
                    continue

                text = (
                    cell.text_value.strip()
                )

                if len(text) < 2:
                    continue

                text_key = (
                    _normalize_text(
                        text
                    )
                )

                if (
                    text_key
                    and text_key
                    in question_key
                ):
                    print(
                        f"- {sheet.name} "
                        f"{cell.coordinate}: "
                        f"{repr(text)}"
                    )

                    # 顺便打印同一行数字，
                    # 方便判断计算输入。
                    same_row_numbers = [
                        (
                            candidate.coordinate,
                            candidate.numeric_value,
                        )
                        for candidate
                        in sheet.cells
                        if (
                            candidate.numeric_value
                            is not None
                            and candidate.row_index
                            == cell.row_index
                        )
                    ]

                    if same_row_numbers:
                        print(
                            "    same-row:",
                            same_row_numbers,
                        )

                    # 以及同列数字。
                    same_column_numbers = [
                        (
                            candidate.coordinate,
                            candidate.numeric_value,
                        )
                        for candidate
                        in sheet.cells
                        if (
                            candidate.numeric_value
                            is not None
                            and candidate.column_index
                            == cell.column_index
                        )
                    ]

                    if same_column_numbers:
                        print(
                            "    same-column:",
                            same_column_numbers[
                                :20
                            ],
                        )

                    found += 1

        if found == 0:
            print(
                "- none by exact "
                "text containment"
            )


if __name__ == "__main__":
    main()