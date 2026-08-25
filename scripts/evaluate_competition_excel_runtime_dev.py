from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.services.competition_dataset import (
    build_competition_question,
    load_competition_qa_excel,
)
from app.services.competition_excel_runtime import (
    load_competition_excel_runtime_resources,
    solve_competition_excel_question,
)


QA_FILE = Path(
    "data/competition/private/qa/QA数据.xlsx"
)

SPLIT_FILE = Path(
    "data/competition/processed/"
    "competition_eval_split_v1.json"
)

ATTACHMENTS_ROOT = Path(
    "data/competition/private/attachments"
)

OUTPUT_FILE = Path(
    "data/competition/processed/eval/"
    "competition_excel_runtime_dev_v1.json"
)


EXPECTED_COUNTS = {
    "表格取数": 12,
    "表格比较": 11,
    "表格计算": 10,
}

EXPECTED_TOTAL = 33


def main() -> None:
    # ========================================================
    # Dataset
    # ========================================================

    cases = tuple(
        load_competition_qa_excel(
            QA_FILE
        )
    )

    case_by_id = {
        case.case_id: case
        for case in cases
    }

    split = json.loads(
        SPLIT_FILE.read_text(
            encoding="utf-8"
        )
    )

    dev_case_ids = tuple(
        split["dev_case_ids"]
    )

    excel_cases = [
        case_by_id[case_id]
        for case_id in dev_case_ids
        if (
            case_by_id[
                case_id
            ].source_type
            == "excel"
        )
    ]

    excel_cases.sort(
        key=lambda case: (
            case.case_id
        )
    )

    # ========================================================
    # Frozen Split Invariants
    # ========================================================

    if (
        len(excel_cases)
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Frozen Dev Excel Case "
            "数量发生变化："
            f"{len(excel_cases)}"
        )

    actual_counts = Counter(
        case.qa_type
        for case in excel_cases
    )

    for (
        qa_type,
        expected_count,
    ) in EXPECTED_COUNTS.items():
        actual_count = (
            actual_counts[
                qa_type
            ]
        )

        if (
            actual_count
            != expected_count
        ):
            raise RuntimeError(
                f"{qa_type} Case 数量"
                "发生变化："
                f"{actual_count} "
                f"!= {expected_count}"
            )

    # ========================================================
    # Process-level Runtime
    # ========================================================

    resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            )
        )
    )

    print()
    print(
        "===== Competition Excel "
        "Runtime Frozen Dev ====="
    )

    print(
        "Cases:",
        len(excel_cases),
    )

    print(
        "By QA type:",
        dict(actual_counts),
    )

    # ========================================================
    # Evaluation
    # ========================================================

    results: list[
        dict[str, object]
    ] = []

    correct_count = 0

    correct_by_type = Counter()
    error_counts = Counter()
    format_counts = Counter()
    solver_counts = Counter()

    for (
        index,
        case,
    ) in enumerate(
        excel_cases,
        start=1,
    ):
        try:
            # ------------------------------------------------
            # Gold-safe Runtime Boundary
            # ------------------------------------------------

            question = (
                build_competition_question(
                    case
                )
            )

            runtime_result = (
                solve_competition_excel_question(
                    question=question,
                    resources=resources,
                )
            )

            prediction = (
                runtime_result
                .prediction
            )

            predicted_option = (
                prediction
                .answer_option
            )

            # ------------------------------------------------
            # Gold 只从这里开始用于 Evaluation
            # ------------------------------------------------

            correct = (
                predicted_option
                == case.answer
            )

            if correct:
                correct_count += 1

                correct_by_type[
                    case.qa_type
                ] += 1

            format_counts[
                runtime_result
                .excel_format
            ] += 1

            solver_counts[
                runtime_result
                .solver_type
            ] += 1

            row = {
                "case_id": (
                    case.case_id
                ),
                "qa_type": (
                    case.qa_type
                ),
                "source_id": (
                    runtime_result
                    .resolution
                    .source_id
                ),
                "resolution_strategy": (
                    runtime_result
                    .resolution
                    .strategy
                ),
                "excel_format": (
                    runtime_result
                    .excel_format
                ),
                "solver_type": (
                    runtime_result
                    .solver_type
                ),
                "predicted_option": (
                    predicted_option
                ),
                "gold_option": (
                    case.answer
                ),
                "correct": correct,
                "answer_text": (
                    prediction
                    .answer_text
                ),
                "confidence": (
                    prediction
                    .confidence
                ),
                "prediction": (
                    prediction
                    .model_dump(
                        mode="json"
                    )
                ),
                "error_type": None,
                "error_message": None,
            }

            print(
                f"[{index:02d}/"
                f"{len(excel_cases):02d}] "
                f"{case.case_id} "
                f"{case.qa_type} "
                f"solver="
                f"{runtime_result.solver_type} "
                f"pred={predicted_option} "
                f"gold={case.answer} "
                f"correct={correct}"
            )

        except Exception as exc:
            error_type = (
                type(exc).__name__
            )

            error_counts[
                error_type
            ] += 1

            row = {
                "case_id": (
                    case.case_id
                ),
                "qa_type": (
                    case.qa_type
                ),
                "source_id": None,
                "resolution_strategy": None,
                "excel_format": None,
                "solver_type": None,
                "predicted_option": None,
                "gold_option": (
                    case.answer
                ),
                "correct": False,
                "answer_text": None,
                "confidence": None,
                "prediction": None,
                "error_type": (
                    error_type
                ),
                "error_message": (
                    str(exc)
                ),
            }

            print(
                f"[{index:02d}/"
                f"{len(excel_cases):02d}] "
                f"{case.case_id} "
                f"{case.qa_type} "
                "ERROR "
                f"{error_type}: "
                f"{exc}"
            )

        results.append(row)

    # ========================================================
    # Summary
    # ========================================================

    qa_type_summary = {}

    for (
        qa_type,
        expected_count,
    ) in EXPECTED_COUNTS.items():
        subset = [
            row
            for row in results
            if (
                row["qa_type"]
                == qa_type
            )
        ]

        correct = sum(
            int(
                bool(
                    row["correct"]
                )
            )
            for row in subset
        )

        errors = sum(
            int(
                row["error_type"]
                is not None
            )
            for row in subset
        )

        qa_type_summary[
            qa_type
        ] = {
            "case_count": (
                len(subset)
            ),
            "expected_count": (
                expected_count
            ),
            "correct_count": (
                correct
            ),
            "accuracy": (
                correct
                / len(subset)
                if subset
                else 0.0
            ),
            "error_count": (
                errors
            ),
        }

    accuracy = (
        correct_count
        / len(excel_cases)
    )

    summary = {
        "case_count": (
            len(excel_cases)
        ),
        "correct_count": (
            correct_count
        ),
        "accuracy": accuracy,
        "error_count": (
            sum(
                error_counts.values()
            )
        ),
        "error_counts": dict(
            sorted(
                error_counts.items()
            )
        ),
        "format_counts": dict(
            sorted(
                format_counts.items()
            )
        ),
        "solver_counts": dict(
            sorted(
                solver_counts.items()
            )
        ),
        "workbook_cache_size": len(
            resources
            .workbook_cache
        ),
        "by_qa_type": (
            qa_type_summary
        ),
    }

    # ========================================================
    # Console Summary
    # ========================================================

    print()
    print(
        "===== Excel Runtime Summary ====="
    )

    print(
        "Correct:",
        f"{correct_count}/"
        f"{len(excel_cases)}",
    )

    print(
        "Accuracy:",
        f"{accuracy:.4f}",
    )

    print(
        "Errors:",
        dict(error_counts),
    )

    print(
        "Formats:",
        dict(format_counts),
    )

    print(
        "Solvers:",
        dict(solver_counts),
    )

    print(
        "Workbook cache size:",
        len(
            resources
            .workbook_cache
        ),
    )

    print()
    print(
        "By QA type:"
    )

    for (
        qa_type,
        group,
    ) in (
        qa_type_summary.items()
    ):
        print(
            f"  {qa_type}: "
            f"{group['correct_count']}/"
            f"{group['case_count']} "
            f"accuracy="
            f"{group['accuracy']:.4f} "
            f"errors="
            f"{group['error_count']}"
        )

    problematic = [
        row
        for row in results
        if not bool(
            row["correct"]
        )
    ]

    if problematic:
        print()
        print(
            "Problematic Cases:"
        )

        for row in problematic:
            print(
                " ",
                row["case_id"],
                row["qa_type"],
                "pred=",
                row[
                    "predicted_option"
                ],
                "gold=",
                row[
                    "gold_option"
                ],
                "error=",
                row[
                    "error_type"
                ],
            )

    # ========================================================
    # Artifact
    # ========================================================

    payload = {
        "schema_version": 1,
        "evaluation_version": (
            "competition_excel_"
            "runtime_dev_v1"
        ),
        "summary": summary,
        "cases": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE,
    )

    # ========================================================
    # Acceptance Gate
    # ========================================================

    if error_counts:
        raise RuntimeError(
            "Excel Runtime Dev "
            "存在执行错误"
        )

    for (
        qa_type,
        expected_count,
    ) in EXPECTED_COUNTS.items():
        actual_correct = (
            correct_by_type[
                qa_type
            ]
        )

        if (
            actual_correct
            != expected_count
        ):
            raise RuntimeError(
                f"{qa_type} 未保持 "
                "Frozen Baseline："
                f"{actual_correct}/"
                f"{expected_count}"
            )

    if (
        correct_count
        != EXPECTED_TOTAL
    ):
        raise RuntimeError(
            "Excel Runtime Dev "
            "未达到 33/33"
        )

    print()
    print(
        "EXCEL RUNTIME DEV ACCEPTED"
    )


if __name__ == "__main__":
    main()