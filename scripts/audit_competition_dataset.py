from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

    source_to_cases: dict[
        str,
        list[str],
    ] = defaultdict(list)

    source_type_by_id: dict[
        str,
        str,
    ] = {}

    for case in cases:
        resolution = resolver.resolve(
            case
        )

        source_to_cases[
            resolution.source_id
        ].append(
            case.case_id
        )

        source_type_by_id[
            resolution.source_id
        ] = case.source_type

    used_source_ids = set(
        source_to_cases
    )

    unused_source_ids = {
        record.source_id
        for record in manifest
        if (
            record.source_id
            not in used_source_ids
        )
    }

    used_source_type_counts = Counter(
        source_type_by_id.values()
    )

    questions_per_source = Counter(
        len(case_ids)
        for case_ids
        in source_to_cases.values()
    )

    multi_question_sources = {
        source_id: case_ids
        for (
            source_id,
            case_ids,
        )
        in source_to_cases.items()
        if len(case_ids) > 1
    }

    source_case_details: dict[
        str,
        list[object],
    ] = defaultdict(list)

    for case in cases:
        resolution = resolver.resolve(
            case
        )

        source_case_details[
            resolution.source_id
        ].append(case)

    mixed_qa_type_sources = {}

    mixed_difficulty_sources = {}

    for (
        source_id,
        source_cases,
    ) in source_case_details.items():

        qa_types = {
            case.qa_type
            for case in source_cases
        }

        difficulties = {
            case.difficulty
            for case in source_cases
        }

        if len(qa_types) > 1:
            mixed_qa_type_sources[
                source_id
            ] = sorted(qa_types)

        if len(difficulties) > 1:
            mixed_difficulty_sources[
                source_id
            ] = sorted(difficulties)

    print(
        "=== Competition Dataset Audit ==="
    )

    print(
        f"QA cases: {len(cases)}"
    )

    print(
        f"All attachments: {len(manifest)}"
    )

    print(
        "Unique used sources:",
        len(used_source_ids),
    )

    print(
        "Unused attachments:",
        len(unused_source_ids),
    )

    print(
        "Used source types:",
        dict(
            used_source_type_counts
        ),
    )

    print(
        "Questions per source:",
        dict(
            sorted(
                questions_per_source.items()
            )
        ),
    )

    print(
        "Sources with multiple questions:",
        len(
            multi_question_sources
        ),
    )

    if multi_question_sources:
        print()
        print(
            "Example multi-question sources:"
        )

        for (
            source_id,
            case_ids,
        ) in list(
            sorted(
                multi_question_sources.items()
            )
        )[:20]:
            print(
                f"- {source_id}: "
                + ", ".join(case_ids)
            )

    print()
    print(
        "Sources with mixed QA types:",
        len(mixed_qa_type_sources),
    )

    print(
        "Sources with mixed difficulties:",
        len(mixed_difficulty_sources),
    )

    if mixed_qa_type_sources:
        print()
        print(
            "Example mixed QA type sources:"
        )

        for (
            source_id,
            qa_types,
        ) in list(
            sorted(
                mixed_qa_type_sources.items()
            )
        )[:10]:
            print(
                f"- {source_id}: "
                f"{qa_types}"
            )


    if mixed_difficulty_sources:
        print()
        print(
            "Example mixed difficulty sources:"
        )

        for (
            source_id,
            difficulties,
        ) in list(
            sorted(
                mixed_difficulty_sources.items()
            )
        )[:10]:
            print(
                f"- {source_id}: "
                f"{difficulties}"
            )

if __name__ == "__main__":
    main()