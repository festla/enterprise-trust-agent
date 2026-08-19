from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    load_competition_qa_excel,
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

    split_payload = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_ids = set(
        split_payload["dev_case_ids"]
    )

    test_ids = set(
        split_payload["test_case_ids"]
    )

    overall: Counter[
        tuple[str, str]
    ] = Counter()

    dev: Counter[
        tuple[str, str]
    ] = Counter()

    test: Counter[
        tuple[str, str]
    ] = Counter()

    format_totals = Counter()

    for case in cases:
        if case.source_type != "excel":
            continue

        resolution = resolver.resolve(
            case
        )

        source = source_by_id[
            resolution.source_id
        ]

        excel_format = (
            source.extension
            .removeprefix(".")
            .lower()
        )

        key = (
            case.qa_type,
            excel_format,
        )

        overall[key] += 1
        format_totals[
            excel_format
        ] += 1

        if case.case_id in dev_ids:
            dev[key] += 1

        elif case.case_id in test_ids:
            test[key] += 1

        else:
            raise RuntimeError(
                "Case 不在 frozen split 中: "
                f"{case.case_id}"
            )

    qa_types = (
        "表格取数",
        "表格比较",
        "表格计算",
    )

    formats = (
        "xls",
        "xlsx",
    )

    def print_section(
        name: str,
        counter: Counter[
            tuple[str, str]
        ],
    ) -> None:
        print()
        print(
            f"=== {name} ==="
        )

        for qa_type in qa_types:
            values = {
                excel_format:
                    counter[
                        (
                            qa_type,
                            excel_format,
                        )
                    ]
                for excel_format
                in formats
            }

            total = sum(
                values.values()
            )

            print(
                qa_type,
                values,
                "total=",
                total,
            )

    print(
        "=== Competition Excel "
        "QA Format Audit ==="
    )

    print(
        "Format totals:",
        dict(format_totals),
    )

    print_section(
        "Overall",
        overall,
    )

    print_section(
        "Dev",
        dev,
    )

    print_section(
        "Test",
        test,
    )


if __name__ == "__main__":
    main()