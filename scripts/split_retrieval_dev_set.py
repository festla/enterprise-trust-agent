from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from app.schemas.retrieval_eval import (
    FinancialFactRetrievalEvalCase,
)


class RetrievalDevSetSplitError(ValueError):
    """Retrieval Dev Set 拆分异常。"""


def _load_cases(
    path: Path,
) -> tuple[
    FinancialFactRetrievalEvalCase,
    ...,
]:
    if not path.is_file():
        raise RetrievalDevSetSplitError(
            f"输入文件不存在：{path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise RetrievalDevSetSplitError(
            f"输入文件不是合法 UTF-8：{path}"
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
            raise RetrievalDevSetSplitError(
                "Case 校验失败："
                f"path={path}，"
                f"line={line_number}"
            ) from exc

        cases.append(case)

    if not cases:
        raise RetrievalDevSetSplitError(
            "输入文件没有有效 Case"
        )

    case_ids = [
        case.case_id
        for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise RetrievalDevSetSplitError(
            "输入文件包含重复 case_id"
        )

    return tuple(cases)


def _write_cases(
    *,
    path: Path,
    cases: tuple[
        FinancialFactRetrievalEvalCase,
        ...,
    ],
) -> None:
    content = "\n".join(
        case.model_dump_json()
        for case in cases
    ) + "\n"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content,
        encoding="utf-8",
    )

    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "按照 report_id 拆分统一的 "
            "Retrieval Dev Set"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    cases = _load_cases(args.input)

    groups: dict[
        str,
        list[FinancialFactRetrievalEvalCase],
    ] = defaultdict(list)

    for case in cases:
        groups[case.report_id].append(case)

    for report_id in sorted(groups):
        ordered_cases = tuple(
            sorted(
                groups[report_id],
                key=lambda case: case.case_id,
            )
        )

        output_path = (
            args.output_dir
            / (
                f"{args.input.stem}_"
                f"{report_id}.jsonl"
            )
        )

        _write_cases(
            path=output_path,
            cases=ordered_cases,
        )

        print(
            f"report_id={report_id} "
            f"case_count={len(ordered_cases)} "
            f"output_path={output_path}"
        )

    print(
        f"source_case_count={len(cases)}"
    )

    print(
        f"report_count={len(groups)}"
    )


if __name__ == "__main__":
    main()