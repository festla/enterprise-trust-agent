from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.schemas.calculation import (
    DerivedCalculation,
)


class DerivedCalculationDatasetError(
    ValueError
):
    """派生计算数据集基础异常。"""


class DerivedCalculationDatasetNotFoundError(
    DerivedCalculationDatasetError
):
    """派生计算数据集文件不存在。"""


class InvalidDerivedCalculationDatasetError(
    DerivedCalculationDatasetError
):
    """派生计算数据集内容无效。"""


def load_derived_calculations(
    path: Path,
) -> tuple[DerivedCalculation, ...]:
    """从JSONL文件加载派生计算结果。"""

    if not path.is_file():
        raise (
            DerivedCalculationDatasetNotFoundError(
                f"派生计算数据集文件不存在：{path}"
            )
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise DerivedCalculationDatasetError(
            f"无法读取派生计算数据集：{path}"
        ) from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise (
            InvalidDerivedCalculationDatasetError(
                "派生计算数据集必须使用"
                f"UTF-8编码：{path}"
            )
        ) from exc

    lines = text.splitlines()

    if not lines:
        raise (
            InvalidDerivedCalculationDatasetError(
                f"派生计算数据集不能为空：{path}"
            )
        )

    calculations: list[
        DerivedCalculation
    ] = []

    first_line_by_id: dict[str, int] = {}

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            raise (
                InvalidDerivedCalculationDatasetError(
                    "JSONL不能包含空记录："
                    f"{path}，第 {line_number} 行"
                )
            )

        try:
            calculation = (
                DerivedCalculation
                .model_validate_json(line)
            )
        except ValidationError as exc:
            raise (
                InvalidDerivedCalculationDatasetError(
                    "派生计算数据集包含"
                    "无效记录："
                    f"{path}，第 {line_number} 行"
                )
            ) from exc

        previous_line = (
            first_line_by_id.get(
                calculation.calculation_id
            )
        )

        if previous_line is not None:
            raise (
                InvalidDerivedCalculationDatasetError(
                    "重复 calculation_id="
                    f"{calculation.calculation_id!r}："
                    f"{path}，首次出现在第 "
                    f"{previous_line} 行，"
                    f"再次出现在第 "
                    f"{line_number} 行"
                )
            )

        first_line_by_id[
            calculation.calculation_id
        ] = line_number

        calculations.append(calculation)

    return tuple(calculations)