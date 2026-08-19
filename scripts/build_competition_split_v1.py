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
from app.services.competition_split import (
    build_grouped_balanced_dev_test_split,
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
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
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

    resolutions = tuple(
        resolver.resolve(case)
        for case in cases
    )

    dev_ids, test_ids = (
        build_grouped_balanced_dev_test_split(
            cases=cases,
            resolutions=resolutions,
            dev_ratio=1.0 / 3.0,
            seed=args.seed,
            search_iterations=25_000,
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    resolution_by_case = {
        resolution.case_id:
        resolution
        for resolution
        in resolutions
    }

    def case_stats(
        ids: tuple[str, ...],
    ) -> dict[str, object]:
        selected = [
            case_by_id[case_id]
            for case_id in ids
        ]

        return {
            "count": len(selected),
            "source_type": dict(
                Counter(
                    case.source_type
                    for case in selected
                )
            ),
            "qa_type": dict(
                Counter(
                    case.qa_type
                    for case in selected
                )
            ),
            "difficulty": dict(
                Counter(
                    case.difficulty
                    for case in selected
                )
            ),
        }

    def source_stats(
        ids: tuple[str, ...],
    ) -> dict[str, object]:
        source_ids = {
            resolution_by_case[
                case_id
            ].source_id
            for case_id in ids
        }

        return {
            "unique_source_count":
                len(source_ids),
        }

    dev_source_ids = {
        resolution_by_case[
            case_id
        ].source_id
        for case_id in dev_ids
    }

    test_source_ids = {
        resolution_by_case[
            case_id
        ].source_id
        for case_id in test_ids
    }

    source_overlap = (
        dev_source_ids
        & test_source_ids
    )

    if source_overlap:
        raise RuntimeError(
            "Source Leakage detected: "
            f"{sorted(source_overlap)}"
        )

    payload = {
        "schema_version": 1,
        "name":
            "competition_eval_split_v1",
        "seed": args.seed,
        "dev_ratio": 1.0 / 3.0,
        "dev_case_ids": list(dev_ids),
        "test_case_ids":
            list(test_ids),
        "dev_stats":
            case_stats(dev_ids),
        "test_stats":
            case_stats(test_ids),
        "dev_source_stats":
            source_stats(dev_ids),
        "test_source_stats":
            source_stats(test_ids),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== Competition Split V1 ==="
    )

    print(
        "Dev:",
        payload["dev_stats"],
    )

    print(
        "Dev sources:",
        payload[
            "dev_source_stats"
        ],
    )

    print(
        "Test:",
        payload["test_stats"],
    )

    print(
        "Test sources:",
        payload[
            "test_source_stats"
        ],
    )

    print(
        "Source overlap:",
        len(source_overlap),
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()