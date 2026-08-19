from __future__ import annotations

import argparse
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

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
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

    if args.case_id not in case_by_id:
        raise SystemExit(
            "Unknown case id: "
            f"{args.case_id}"
        )

    case = case_by_id[
        args.case_id
    ]

    if case.source_type != "excel":
        raise SystemExit(
            f"{args.case_id} "
            "不是 Excel QA"
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
        record.source_id: record
        for record in manifest
    }

    source = source_by_id[
        resolution.source_id
    ]

    workbook = parse_competition_excel(
        attachments_root=(
            args.attachments
        ),
        source=source,
    )

    print(
        "=== Competition Excel Case ==="
    )

    print(
        "Case:",
        case.case_id,
    )

    print(
        "Source:",
        source.relative_path,
    )

    print(
        "Format:",
        workbook.excel_format,
    )

    print(
        "Sheets:",
        workbook.sheet_names,
    )

    print(
        "Total sparse cells:",
        workbook.total_cell_count,
    )

    for sheet in workbook.sheets:
        print()

        print(
            f"--- Sheet: {sheet.name} ---"
        )

        print(
            "Visible:",
            sheet.visible,
        )

        print(
            "Declared size:",
            (
                sheet.declared_max_row,
                sheet.declared_max_column,
            ),
        )

        print(
            "Sparse cells:",
            sheet.non_empty_cell_count,
        )

        print(
            "Merged ranges:",
            len(sheet.merged_ranges),
        )

        print()

        for cell in (
            sheet.cells[
                : args.limit
            ]
        ):
            print(
                cell.coordinate,
                "|",
                cell.cell_type,
                "|",
                repr(
                    cell.text_value
                ),
                "| numeric=",
                cell.numeric_value,
                "| format=",
                repr(
                    cell.number_format
                ),
                "| merged=",
                cell.merged_range,
            )


if __name__ == "__main__":
    main()