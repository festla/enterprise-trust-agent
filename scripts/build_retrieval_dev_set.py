from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
)


class RetrievalDevSetBuildError(ValueError):
    """Retrieval Dev Set 构建异常。"""


def _load_cases(
    path: Path,
) -> tuple[
    FinancialFactRetrievalEvalCase,
    ...,
]:
    """读取并验证一个 Retrieval JSONL 文件。"""

    if not path.is_file():
        raise RetrievalDevSetBuildError(
            f"输入文件不存在：{path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise RetrievalDevSetBuildError(
            f"文件不是合法 UTF-8：{path}"
        ) from exc
    except OSError as exc:
        raise RetrievalDevSetBuildError(
            f"无法读取输入文件：{path}"
        ) from exc

    cases: list[
        FinancialFactRetrievalEvalCase
    ] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            case = (
                FinancialFactRetrievalEvalCase
                .model_validate_json(line)
            )
        except ValidationError as exc:
            raise RetrievalDevSetBuildError(
                "Retrieval Case 校验失败："
                f"path={path}，"
                f"line={line_number}"
            ) from exc

        cases.append(case)

    if not cases:
        raise RetrievalDevSetBuildError(
            f"输入文件没有有效记录：{path}"
        )

    return tuple(cases)


def _write_cases(
    *,
    output_path: Path,
    cases: tuple[
        FinancialFactRetrievalEvalCase,
        ...,
    ],
) -> None:
    """以 UTF-8 JSONL 原子写入开发集。"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = "\n".join(
        case.model_dump_json()
        for case in cases
    ) + "\n"

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "合并多个经过人工核验的 Retrieval "
            "Case 文件，生成统一开发集"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="input_paths",
        help=(
            "输入 JSONL，可以多次指定"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    all_cases: list[
        FinancialFactRetrievalEvalCase
    ] = []

    source_by_case_id: dict[
        str,
        Path,
    ] = {}

    for input_path in args.input_paths:
        for case in _load_cases(input_path):
            previous_source = (
                source_by_case_id.get(
                    case.case_id
                )
            )

            if previous_source is not None:
                raise RetrievalDevSetBuildError(
                    "发现重复 case_id："
                    f"{case.case_id}；"
                    f"source_1={previous_source}；"
                    f"source_2={input_path}"
                )

            source_by_case_id[
                case.case_id
            ] = input_path

            all_cases.append(case)

    ordered_cases = tuple(
        sorted(
            all_cases,
            key=lambda case: (
                case.company_id,
                case.report_id,
                case.case_id,
            ),
        )
    )

    _write_cases(
        output_path=args.output,
        cases=ordered_cases,
    )

    report_counts = Counter(
        case.report_id
        for case in ordered_cases
    )

    print(
        f"output_path={args.output}"
    )

    print(
        f"case_count={len(ordered_cases)}"
    )

    print(
        "unique_case_id_count="
        f"{len(source_by_case_id)}"
    )

    for report_id in sorted(
        report_counts
    ):
        print(
            f"report={report_id} "
            f"case_count="
            f"{report_counts[report_id]}"
        )

    print("cases:")

    for case in ordered_cases:
        print(
            f"- {case.case_id} | "
            f"{case.report_id} | "
            f"{case.metric_name} | "
            f"gold={case.gold_pdf_pages}"
        )


if __name__ == "__main__":
    main()