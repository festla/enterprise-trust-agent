from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
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

    value = re.sub(
        r"[，。；：、,.!?！？"
        r"（）()《》“”\"'：:/\\_-]",
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
        "--case-id",
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

    question = (
        build_competition_question(
            case
        )
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

    print(
        "=== Excel Compare Diagnostic ==="
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

    for sheet in workbook.sheets:
        print()
        print(
            f"--- Sheet: {sheet.name} ---"
        )

        text_cells = [
            cell
            for cell in sheet.cells
            if (
                cell.cell_type
                == "text"
                and cell.text_value
            )
        ]

        numeric_cells = [
            cell
            for cell in sheet.cells
            if (
                cell.numeric_value
                is not None
            )
        ]

        for (
            option,
            option_text,
        ) in question.options.items():
            option_key = (
                _normalize_text(
                    option_text
                )
            )

            print()
            print(
                f"{option}: "
                f"{option_text!r}"
            )

            exact = []

            containing = []

            contained_by = []

            for cell in text_cells:
                cell_key = (
                    _normalize_text(
                        cell.text_value
                    )
                )

                if cell_key == option_key:
                    exact.append(cell)

                elif (
                    option_key
                    and option_key
                    in cell_key
                ):
                    containing.append(
                        cell
                    )

                elif (
                    cell_key
                    and cell_key
                    in option_key
                ):
                    contained_by.append(
                        cell
                    )

            print(
                "  exact:",
                [
                    (
                        cell.coordinate,
                        cell.text_value,
                    )
                    for cell in exact
                ],
            )

            print(
                "  option-in-cell:",
                [
                    (
                        cell.coordinate,
                        cell.text_value,
                    )
                    for cell in containing[
                        :10
                    ]
                ],
            )

            print(
                "  cell-in-option:",
                [
                    (
                        cell.coordinate,
                        cell.text_value,
                    )
                    for cell in contained_by[
                        :10
                    ]
                ],
            )

            candidate_labels = (
                exact
                or containing
                or contained_by
            )

            for label in (
                candidate_labels[:5]
            ):
                same_row = [
                    (
                        cell.coordinate,
                        cell.numeric_value,
                    )
                    for cell
                    in numeric_cells
                    if (
                        cell.row_index
                        == label.row_index
                    )
                ]

                same_column = [
                    (
                        cell.coordinate,
                        cell.numeric_value,
                    )
                    for cell
                    in numeric_cells
                    if (
                        cell.column_index
                        == label.column_index
                    )
                ]

                print(
                    "  candidate:",
                    label.coordinate,
                    repr(
                        label.text_value
                    ),
                )

                print(
                    "    same-row numbers:",
                    same_row[:20],
                )

                print(
                    "    same-column numbers:",
                    same_column[
                        :20
                    ],
                )


if __name__ == "__main__":
    main()