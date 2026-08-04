from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.schemas.complex_plan_eval import (
    ComplexFinancialEvalCase,
)


class ComplexPlanEvalDatasetError(
    ValueError
):
    """复杂规划评测数据集基础异常。"""


class ComplexPlanEvalDatasetNotFoundError(
    ComplexPlanEvalDatasetError
):
    """复杂规划评测数据集文件不存在。"""


class InvalidComplexPlanEvalDatasetError(
    ComplexPlanEvalDatasetError
):
    """复杂规划评测数据集内容无效。"""


def load_complex_financial_eval_cases(
    path: Path,
) -> tuple[
    ComplexFinancialEvalCase,
    ...,
]:
    """从 JSONL 文件加载复杂财务评测 Case。"""

    if not path.is_file():
        raise ComplexPlanEvalDatasetNotFoundError(
            f"复杂评测数据集文件不存在：{path}"
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ComplexPlanEvalDatasetError(
            f"无法读取复杂评测数据集：{path}"
        ) from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidComplexPlanEvalDatasetError(
            "复杂评测数据集必须使用 UTF-8 编码："
            f"{path}"
        ) from exc

    lines = text.splitlines()

    if not lines:
        raise InvalidComplexPlanEvalDatasetError(
            f"复杂评测数据集不能为空：{path}"
        )

    cases: list[ComplexFinancialEvalCase] = []

    first_line_by_case_id: dict[str, int] = {}

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            raise InvalidComplexPlanEvalDatasetError(
                "JSONL 不能包含空记录："
                f"{path}，第 {line_number} 行"
            )

        try:
            case = (
                ComplexFinancialEvalCase
                .model_validate_json(line)
            )
        except ValidationError as exc:
            raise InvalidComplexPlanEvalDatasetError(
                "复杂评测数据集包含无效记录："
                f"{path}，第 {line_number} 行"
            ) from exc

        previous_line = (
            first_line_by_case_id.get(
                case.case_id
            )
        )

        if previous_line is not None:
            raise InvalidComplexPlanEvalDatasetError(
                f"重复 case_id={case.case_id!r}："
                f"{path}，首次出现在第 "
                f"{previous_line} 行，"
                f"再次出现在第 {line_number} 行"
            )

        first_line_by_case_id[
            case.case_id
        ] = line_number

        cases.append(case)

    return tuple(cases)