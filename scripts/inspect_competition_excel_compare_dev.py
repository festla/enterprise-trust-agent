from __future__ import annotations

import argparse
import json
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

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    split_payload = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_ids = set(
        split_payload["dev_case_ids"]
    )

    compare_cases = [
        case
        for case in cases
        if (
            case.case_id in dev_ids
            and case.source_type
            == "excel"
            and case.qa_type
            == "表格比较"
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
        "=== Dev Excel Compare Cases ==="
    )

    print(
        "Count:",
        len(compare_cases),
    )

    for index, case in enumerate(
        compare_cases,
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
            f"{len(compare_cases)}] "
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

        # ====================================================
        # Dev-only diagnostic.
        #
        # 这里只用于理解题型和后续验证，
        # 不进入 Solver。
        # ====================================================

        print()
        print(
            "Gold option:",
            case.answer,
        )

        print(
            "Gold answer:",
            case.answer_text,
        )

        print()
        print(
            "Relevant text cells:"
        )

        question_key = (
            case.question
            .replace(" ", "")
        )

        shown = 0

        for sheet in workbook.sheets:
            for cell in sheet.cells:
                if (
                    cell.cell_type
                    != "text"
                ):
                    continue

                text = (
                    cell.text_value
                    .strip()
                )

                if len(text) < 2:
                    continue

                normalized = (
                    text.replace(
                        " ",
                        "",
                    )
                )

                if (
                    normalized
                    in question_key
                ):
                    print(
                        f"- {sheet.name} "
                        f"{cell.coordinate}: "
                        f"{repr(text)}"
                    )

                    shown += 1

        if shown == 0:
            print(
                "- none by exact "
                "text containment"
            )


if __name__ == "__main__":
    main()