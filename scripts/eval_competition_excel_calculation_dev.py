from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_calculation import (
    solve_excel_table_calculation,
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
            and case.source_type == "excel"
            and case.qa_type == "表格计算"
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

    resolution_modes = Counter()

    errors = Counter()

    print(
        "Dev calculation cases:",
        len(calculation_cases),
    )

    for index, case in enumerate(
        calculation_cases,
        start=1,
    ):
        print()
        print(
            f"[{index}/"
            f"{len(calculation_cases)}] "
            f"{case.case_id}"
        )

        try:
            # =================================================
            # Solver 输入：
            # 不包含 answer / answer_text / evidence。
            # =================================================

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
                solve_excel_table_calculation(
                    question=question,
                    workbook=workbook,
                )
            )

            # =================================================
            # Gold 只在 prediction 完成之后用于评估。
            # =================================================

            correct = (
                prediction.answer_option
                == case.answer
            )

            if correct:
                correct_count += 1

            resolution_modes[
                prediction.resolution_mode
            ] += 1

            print(
                "Entity:",
                prediction.entity_text,
                "@",
                prediction.entity_coordinate,
            )

            print(
                "Start:",
                prediction.start.scope_text,
                "=",
                prediction.start.numeric_value,
                "@",
                prediction.start.coordinate,
                "headers=",
                prediction.start.header_labels,
            )

            print(
                "End:",
                prediction.end.scope_text,
                "=",
                prediction.end.numeric_value,
                "@",
                prediction.end.coordinate,
                "headers=",
                prediction.end.header_labels,
            )

            print(
                "Formula:",
                prediction.formula,
            )

            print(
                "Result:",
                prediction.result,
            )

            print(
                "Resolution mode:",
                prediction.resolution_mode,
            )

            print(
                "Prediction:",
                prediction.answer_option,
                prediction.answer_text,
            )

            print(
                "Gold:",
                case.answer,
                case.answer_text,
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
                    "entity_text":
                        prediction.entity_text,
                    "entity_coordinate":
                        prediction.entity_coordinate,
                    "start":
                        prediction.start.model_dump(),
                    "end":
                        prediction.end.model_dump(),
                    "formula":
                        prediction.formula,
                    "result":
                        prediction.result,
                    "resolution_mode":
                        prediction.resolution_mode,
                    "confidence":
                        prediction.confidence,
                    "error_type":
                        None,
                    "error_message":
                        None,
                }
            )

        except Exception as exc:
            error_type = type(
                exc
            ).__name__

            errors[
                error_type
            ] += 1

            print(
                "ERROR:",
                error_type,
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
                    "entity_text":
                        None,
                    "entity_coordinate":
                        None,
                    "start":
                        None,
                    "end":
                        None,
                    "formula":
                        None,
                    "result":
                        None,
                    "resolution_mode":
                        None,
                    "confidence":
                        None,
                    "error_type":
                        error_type,
                    "error_message":
                        str(exc),
                }
            )

    count = len(
        calculation_cases
    )

    accuracy = (
        correct_count / count
        if count
        else 0.0
    )

    payload = {
        "summary": {
            "eval_name":
                "competition_excel_calculation_dev_v1",
            "split":
                "dev",
            "qa_type":
                "表格计算",
            "case_count":
                count,
            "correct_count":
                correct_count,
            "accuracy":
                accuracy,
            "resolution_modes":
                dict(
                    resolution_modes
                ),
            "errors":
                dict(
                    errors
                ),
        },
        "results":
            results,
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
        "=== Excel Calculation "
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
        "Resolution modes:",
        dict(
            resolution_modes
        ),
    )

    print(
        "Errors:",
        dict(
            errors
        ),
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()