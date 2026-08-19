from __future__ import annotations

import argparse
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
        "--sheet",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_competition_qa_excel(
        args.qa,
        worksheet_name=args.sheet,
    )

    manifest = (
        build_competition_source_manifest(
            args.attachments
        )
    )

    resolver = CompetitionSourceResolver(
        manifest
    )

    failures: list[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = []

    strategies: Counter[str] = Counter()

    for case in cases:
        try:
            resolution = resolver.resolve(
                case
            )

            strategies[
                resolution.strategy
            ] += 1

        except Exception as exc:
            failures.append(
                (
                    case.case_id,
                    case.source_type,
                    case.qa_type,
                    str(exc),
                )
            )

    source_counts = Counter(
        case.source_type
        for case in cases
    )

    qa_type_counts = Counter(
        case.qa_type
        for case in cases
    )

    difficulty_counts = Counter(
        case.difficulty
        for case in cases
    )

    attachment_counts = Counter(
        record.source_type
        for record in manifest
    )

    failure_source_counts = Counter(
        source_type
        for (
            _,
            source_type,
            _,
            _,
        )
        in failures
    )

    failure_qa_type_counts = Counter(
        qa_type
        for (
            _,
            _,
            qa_type,
            _,
        )
        in failures
    )

    print(
        "=== Competition Dataset Gate ==="
    )

    print(
        f"QA cases: {len(cases)}"
    )

    print(
        "QA source types:",
        dict(source_counts),
    )

    print(
        "QA types:",
        dict(qa_type_counts),
    )

    print(
        "Difficulty:",
        dict(difficulty_counts),
    )

    print(
        f"Attachments: {len(manifest)}"
    )

    print(
        "Attachment types:",
        dict(attachment_counts),
    )

    print(
        "Resolution strategies:",
        dict(strategies),
    )

    print(
        "Resolved:",
        f"{len(cases) - len(failures)}"
        f"/{len(cases)}",
    )

    if failures:
        print()
        print(
            "Failure source types:",
            dict(
                failure_source_counts
            ),
        )

        print(
            "Failure QA types:",
            dict(
                failure_qa_type_counts
            ),
        )

        print()
        print("Failures:")

        for (
            case_id,
            source_type,
            qa_type,
            message,
        ) in failures:
            print(
                f"- {case_id} "
                f"[{source_type}/"
                f"{qa_type}]: "
                f"{message}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()