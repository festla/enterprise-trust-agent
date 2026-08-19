from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_compare import (
    solve_excel_table_compare,
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

    parser.add_argument(
        "--output",
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

    split = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_ids = set(
        split["dev_case_ids"]
    )

    compare_cases = [
        case
        for case in cases
        if (
            case.case_id in dev_ids
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

    workbook_cache = {}

    results = []

    correct_count = 0

    for index, case in enumerate(
        compare_cases,
        start=1,
    ):
        print()
        print(
            f"[{index}/"
            f"{len(compare_cases)}] "
            f"{case.case_id}"
        )

        try:
            question = (
                build_competition_question(
                    case
                )
            )

            resolution = resolver.resolve(
                case
            )

            source = source_by_id[
                resolution.source_id
            ]

            if (
                source.source_id
                not in workbook_cache
            ):
                workbook_cache[
                    source.source_id
                ] = (
                    parse_competition_excel(
                        attachments_root=(
                            args.attachments
                        ),
                        source=source,
                    )
                )

            workbook = workbook_cache[
                source.source_id
            ]

            prediction = (
                solve_excel_table_compare(
                    question=question,
                    workbook=workbook,
                )
            )

            # Gold 从这里才进入。
            correct = (
                prediction.answer_option
                == case.answer
            )

            if correct:
                correct_count += 1

            print(
                "Prediction:",
                prediction.answer_option,
                prediction.answer_text,
                prediction.winning_value,
            )

            print(
                "Gold:",
                case.answer,
                case.answer_text,
            )

            for item in (
                prediction.items
            ):
                print(
                    " ",
                    item.option,
                    item.label_text,
                    "=",
                    item.numeric_value,
                    "@",
                    item.value_coordinate,
                    "context=",
                    item.context_labels,
                )

            print(
                "Correct:",
                correct,
            )

            results.append(
                {
                    "case_id":
                        case.case_id,
                    "predicted_option":
                        prediction.answer_option,
                    "gold_option":
                        case.answer,
                    "correct":
                        correct,
                    "answer_text":
                        prediction.answer_text,
                    "winning_value":
                        prediction.winning_value,
                    "sheet_name":
                        prediction.sheet_name,
                    "confidence":
                        prediction.confidence,
                    "items": [
                        item.model_dump()
                        for item
                        in prediction.items
                    ],
                    "error_type":
                        None,
                    "error_message":
                        None,
                }
            )

        except Exception as exc:
            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

            results.append(
                {
                    "case_id":
                        case.case_id,
                    "predicted_option":
                        None,
                    "gold_option":
                        case.answer,
                    "correct":
                        False,
                    "answer_text":
                        None,
                    "winning_value":
                        None,
                    "sheet_name":
                        None,
                    "confidence":
                        None,
                    "items": [],
                    "error_type":
                        type(exc).__name__,
                    "error_message":
                        str(exc),
                }
            )

    count = len(
        compare_cases
    )

    accuracy = (
        correct_count / count
        if count
        else 0.0
    )

    payload = {
        "summary": {
            "eval_name":
                "competition_excel_compare_dev_v1",
            "split": "dev",
            "qa_type": "表格比较",
            "case_count": count,
            "correct_count":
                correct_count,
            "accuracy": accuracy,
        },
        "results": results,
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

    print()
    print(
        "=== Excel Compare "
        "Dev Baseline V1 ==="
    )

    print(
        f"Correct: "
        f"{correct_count}/{count}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.3f}"
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()