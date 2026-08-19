from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_lookup import (
    solve_excel_table_lookup,
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

    # ========================================================
    # Load Dataset
    # ========================================================

    cases = load_competition_qa_excel(
        args.qa
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    # ========================================================
    # Load Frozen Split
    # ========================================================

    split_payload = json.loads(
        args.split.read_text(
            encoding="utf-8"
        )
    )

    dev_case_ids = tuple(
        split_payload[
            "dev_case_ids"
        ]
    )

    lookup_case_ids = [
        case_id
        for case_id
        in dev_case_ids
        if (
            case_by_id[
                case_id
            ].qa_type
            == "表格取数"
        )
    ]

    # 你的当前 frozen split
    # 应该有 12 个 Dev 表格取数 Case。
    print(
        "Dev table lookup cases:",
        len(lookup_case_ids),
    )

    # ========================================================
    # Source Infrastructure
    # ========================================================

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

    # ========================================================
    # Evaluation
    # ========================================================

    results: list[
        dict[str, object]
    ] = []

    correct_count = 0

    error_counts: Counter[
        str
    ] = Counter()

    for index, case_id in enumerate(
        lookup_case_ids,
        start=1,
    ):
        case = case_by_id[
            case_id
        ]

        print()
        print(
            f"[{index}/"
            f"{len(lookup_case_ids)}] "
            f"{case_id}"
        )

        try:
            # ------------------------------------------------
            # Gold-free Solver Input
            # ------------------------------------------------

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

            workbook = (
                parse_competition_excel(
                    attachments_root=(
                        args.attachments
                    ),
                    source=source,
                )
            )

            prediction = (
                solve_excel_table_lookup(
                    question=question,
                    workbook=workbook,
                )
            )

            # ------------------------------------------------
            # Evaluation Boundary
            #
            # Gold 只从这里开始使用。
            # ------------------------------------------------

            correct = (
                prediction.answer_option
                == case.answer
            )

            if correct:
                correct_count += 1

            result = {
                "case_id": case.case_id,
                "source_id": (
                    resolution.source_id
                ),
                "source_format": (
                    workbook.excel_format
                ),
                "predicted_option": (
                    prediction.answer_option
                ),
                "gold_option": case.answer,
                "correct": correct,
                "predicted_value": (
                    prediction.answer_text
                ),
                "gold_answer_text": (
                    case.answer_text
                ),
                "confidence": (
                    prediction.confidence
                ),
                "sheet_name": (
                    prediction.evidence
                    .sheet_name
                ),
                "value_coordinate": (
                    prediction.evidence
                    .value_coordinate
                ),
                "row_label_coordinate": (
                    prediction.evidence
                    .row_label_coordinate
                ),
                "row_label": (
                    prediction.evidence
                    .row_label
                ),
                "column_label_coordinate": (
                    prediction.evidence
                    .column_label_coordinate
                ),
                "column_label": (
                    prediction.evidence
                    .column_label
                ),
                "error_type": None,
                "error_message": None,
            }

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
                "Cell:",
                prediction.evidence
                .value_coordinate,
            )

            print(
                "Correct:",
                correct,
            )

        except Exception as exc:
            error_type = (
                type(exc).__name__
            )

            error_counts[
                error_type
            ] += 1

            result = {
                "case_id": case.case_id,
                "source_id": None,
                "source_format": None,
                "predicted_option": None,
                "gold_option": case.answer,
                "correct": False,
                "predicted_value": None,
                "gold_answer_text": (
                    case.answer_text
                ),
                "confidence": None,
                "sheet_name": None,
                "value_coordinate": None,
                "row_label_coordinate": None,
                "row_label": None,
                "column_label_coordinate":
                    None,
                "column_label": None,
                "error_type": error_type,
                "error_message": str(exc),
            }

            print(
                "ERROR:",
                error_type,
                str(exc),
            )

        results.append(result)

    # ========================================================
    # Summary
    # ========================================================

    case_count = len(
        lookup_case_ids
    )

    accuracy = (
        correct_count / case_count
        if case_count
        else 0.0
    )

    xls_results = [
        result
        for result in results
        if (
            result[
                "source_format"
            ]
            == "xls"
        )
    ]

    xlsx_results = [
        result
        for result in results
        if (
            result[
                "source_format"
            ]
            == "xlsx"
        )
    ]

    def subset_accuracy(
        subset: list[
            dict[str, object]
        ],
    ) -> float | None:
        if not subset:
            return None

        return (
            sum(
                bool(
                    result["correct"]
                )
                for result in subset
            )
            / len(subset)
        )

    summary = {
        "eval_name": (
            "competition_excel_lookup_dev_v1"
        ),
        "split": "dev",
        "qa_type": "表格取数",
        "case_count": case_count,
        "correct_count": (
            correct_count
        ),
        "accuracy": accuracy,
        "xls_case_count": len(
            xls_results
        ),
        "xls_accuracy": (
            subset_accuracy(
                xls_results
            )
        ),
        "xlsx_case_count": len(
            xlsx_results
        ),
        "xlsx_accuracy": (
            subset_accuracy(
                xlsx_results
            )
        ),
        "error_counts": dict(
            error_counts
        ),
    }

    output_payload = {
        "summary": summary,
        "results": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=== Excel Table Lookup "
        "Dev Baseline ==="
    )

    print(
        f"Correct: "
        f"{correct_count}/"
        f"{case_count}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.3f}"
    )

    print(
        "XLS:",
        len(xls_results),
        subset_accuracy(
            xls_results
        ),
    )

    print(
        "XLSX:",
        len(xlsx_results),
        subset_accuracy(
            xlsx_results
        ),
    )

    print(
        "Errors:",
        dict(error_counts),
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()