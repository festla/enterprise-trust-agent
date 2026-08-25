from __future__ import annotations

import json
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


TARGET_QA_TYPES = (
    "表格取数",
    "表格比较",
    "表格计算",
)


def main() -> None:
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

    dev_ids = tuple(
        split["dev_case_ids"]
    )

    selected = {}

    for case_id in dev_ids:
        case = case_by_id[
            case_id
        ]

        if (
            case.source_type
            != "excel"
        ):
            continue

        if (
            case.qa_type
            not in TARGET_QA_TYPES
        ):
            continue

        selected.setdefault(
            case.qa_type,
            case,
        )

    missing = (
        set(TARGET_QA_TYPES)
        - set(selected)
    )

    if missing:
        raise RuntimeError(
            "Dev Split 缺少 Excel QA 类型："
            f"{tuple(sorted(missing))}"
        )

    resources = (
        load_competition_excel_runtime_resources(
            attachments_root=(
                ATTACHMENTS_ROOT
            )
        )
    )

    print()
    print(
        "===== Competition "
        "Excel Runtime Smoke ====="
    )

    for qa_type in TARGET_QA_TYPES:
        case = selected[
            qa_type
        ]

        #
        # Gold-safe boundary
        #
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

        correct = (
            predicted_option
            == case.answer
        )

        print()
        print(
            "Case:",
            case.case_id,
        )

        print(
            "QA type:",
            qa_type,
        )

        print(
            "Source:",
            runtime_result
            .resolution
            .source_id,
        )

        print(
            "Excel format:",
            runtime_result
            .excel_format,
        )

        print(
            "Solver:",
            runtime_result
            .solver_type,
        )

        print(
            "Prediction:",
            predicted_option,
        )

        print(
            "Gold:",
            case.answer,
        )

        print(
            "Correct:",
            correct,
        )

        if not correct:
            raise RuntimeError(
                f"{case.case_id} "
                "Excel Runtime Smoke "
                "未保持 baseline"
            )

    print()
    print(
        "Workbook cache size:",
        len(
            resources
            .workbook_cache
        ),
    )

    print()
    print(
        "EXCEL RUNTIME PASS"
    )


if __name__ == "__main__":
    main()