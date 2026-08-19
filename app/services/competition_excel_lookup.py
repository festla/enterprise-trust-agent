from __future__ import annotations

import math
import re
import unicodedata

from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)

from app.schemas.competition import (
    CompetitionQuestion,
)
from app.schemas.competition_excel import (
    CompetitionExcelCell,
    CompetitionExcelSheet,
    CompetitionExcelWorkbook,
)
from app.schemas.competition_excel_solver import (
    CompetitionExcelEvidence,
    CompetitionExcelLookupResult,
)


class CompetitionExcelLookupError(
    RuntimeError
):
    pass


def _normalize_text(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    value = re.sub(
        r"[，。；：、,.!?！？（）()《》“”\"'：:/\\_-]",
        "",
        value,
    )

    return value


def _text_mentioned_in_question(
    *,
    cell_text: str,
    question: str,
) -> bool:
    cell_key = _normalize_text(
        cell_text
    )

    question_key = _normalize_text(
        question
    )

    # 太短的文本例如：
    # 年、月、1
    # 不适合直接作为语义标签。
    if len(cell_key) < 2:
        return False

    return (
        cell_key
        in question_key
    )


def _build_cell_map(
    sheet: CompetitionExcelSheet,
) -> dict[
    tuple[int, int],
    CompetitionExcelCell,
]:
    return {
        (
            cell.row_index,
            cell.column_index,
        ): cell
        for cell in sheet.cells
    }


def _find_label_cells(
    *,
    sheet: CompetitionExcelSheet,
    question: str,
) -> list[
    CompetitionExcelCell
]:
    matches = []

    for cell in sheet.cells:
        if cell.cell_type != "text":
            continue

        if not cell.text_value:
            continue

        if _text_mentioned_in_question(
            cell_text=cell.text_value,
            question=question,
        ):
            matches.append(cell)

    return matches


def _parse_option_number(
    value: str,
) -> tuple[
    Decimal,
    int,
] | None:
    """
    从选择项中提取数字，同时保留题目使用的小数精度。

    示例：

    "259039.99亿元"
        ->
    Decimal("259039.99"), 2

    "5155"
        ->
    Decimal("5155"), 0
    """

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    normalized = normalized.replace(
        ",",
        "",
    )

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        normalized,
    )

    if not match:
        return None

    number_text = match.group(0)

    try:
        number = Decimal(
            number_text
        )
    except InvalidOperation:
        return None

    if "." in number_text:
        decimal_places = len(
            number_text.split(
                ".",
                maxsplit=1,
            )[1]
        )
    else:
        decimal_places = 0

    return (
        number,
        decimal_places,
    )


def _quantize_to_places(
    value: Decimal,
    decimal_places: int,
) -> Decimal:
    quantizer = Decimal(1).scaleb(
        -decimal_places
    )

    return value.quantize(
        quantizer,
        rounding=ROUND_HALF_UP,
    )


def _resolve_answer_option(
    *,
    question: CompetitionQuestion,
    numeric_value: int | float,
) -> str | None:
    raw_value = Decimal(
        str(numeric_value)
    )

    matched_options: list[str] = []

    for option, text in (
        question.options.items()
    ):
        parsed = _parse_option_number(
            text
        )

        if parsed is None:
            continue

        (
            option_value,
            decimal_places,
        ) = parsed

        normalized_prediction = (
            _quantize_to_places(
                raw_value,
                decimal_places,
            )
        )

        normalized_option = (
            _quantize_to_places(
                option_value,
                decimal_places,
            )
        )

        if (
            normalized_prediction
            == normalized_option
        ):
            matched_options.append(
                option
            )

    # 必须唯一匹配。
    #
    # 如果两个选项经过舍入后都一样，
    # Solver 不应该自己猜。
    if len(matched_options) == 1:
        return matched_options[0]

    return None


def _candidate_numeric_cells(
    *,
    sheet: CompetitionExcelSheet,
) -> list[
    CompetitionExcelCell
]:
    return [
        cell
        for cell in sheet.cells
        if (
            cell.numeric_value
            is not None
        )
    ]


def _score_candidate(
    *,
    candidate: CompetitionExcelCell,
    label_cells: list[
        CompetitionExcelCell
    ],
) -> tuple[
    float,
    CompetitionExcelCell | None,
    CompetitionExcelCell | None,
]:
    """
    通用二维表格启发式。

    优先：
    1. 同一行左侧存在问题中出现的文本标签；
    2. 同一列上方存在问题中出现的文本标签；
    3. 标签距离越近越好。
    """

    best_row_label = None
    best_column_label = None

    row_score = 0.0
    column_score = 0.0

    for label in label_cells:

        # --------------------------------------------
        # Row Label
        # --------------------------------------------

        if (
            label.row_index
            == candidate.row_index
            and label.column_index
            < candidate.column_index
        ):
            distance = (
                candidate.column_index
                - label.column_index
            )

            score = (
                4.0
                + 1.0 / distance
            )

            if score > row_score:
                row_score = score
                best_row_label = label

        # --------------------------------------------
        # Column Label
        # --------------------------------------------

        if (
            label.column_index
            == candidate.column_index
            and label.row_index
            < candidate.row_index
        ):
            distance = (
                candidate.row_index
                - label.row_index
            )

            score = (
                4.0
                + 1.0 / distance
            )

            if score > column_score:
                column_score = score
                best_column_label = label

    score = (
        row_score
        + column_score
    )

    # 行列语义同时命中，
    # 额外奖励。
    if (
        best_row_label is not None
        and best_column_label
        is not None
    ):
        score += 5.0

    return (
        score,
        best_row_label,
        best_column_label,
    )


def solve_excel_table_lookup(
    *,
    question: CompetitionQuestion,
    workbook: CompetitionExcelWorkbook,
) -> CompetitionExcelLookupResult:
    if question.qa_type != "表格取数":
        raise CompetitionExcelLookupError(
            "该 Solver 只处理表格取数"
        )

    best: tuple[
        float,
        CompetitionExcelSheet,
        CompetitionExcelCell,
        CompetitionExcelCell | None,
        CompetitionExcelCell | None,
    ] | None = None

    for sheet in workbook.sheets:
        if not sheet.visible:
            continue

        label_cells = (
            _find_label_cells(
                sheet=sheet,
                question=question.question,
            )
        )

        if not label_cells:
            continue

        for candidate in (
            _candidate_numeric_cells(
                sheet=sheet
            )
        ):
            (
                score,
                row_label,
                column_label,
            ) = _score_candidate(
                candidate=candidate,
                label_cells=label_cells,
            )

            if score <= 0:
                continue

            if (
                best is None
                or score > best[0]
            ):
                best = (
                    score,
                    sheet,
                    candidate,
                    row_label,
                    column_label,
                )

    if best is None:
        raise CompetitionExcelLookupError(
            "无法根据问题定位表格数值"
        )

    (
        score,
        sheet,
        value_cell,
        row_label,
        column_label,
    ) = best

    if value_cell.numeric_value is None:
        raise CompetitionExcelLookupError(
            "定位结果不是数值 Cell"
        )

    answer_option = (
        _resolve_answer_option(
            question=question,
            numeric_value=(
                value_cell.numeric_value
            ),
        )
    )

    # 当前只是启发式 baseline。
    # 行列都命中时给较高 confidence。
    if (
        row_label is not None
        and column_label is not None
    ):
        confidence = 0.95

    elif row_label is not None:
        confidence = 0.70

    elif column_label is not None:
        confidence = 0.60

    else:
        confidence = 0.30

    return CompetitionExcelLookupResult(
        answer_option=answer_option,
        answer_text=(
            value_cell.text_value
        ),
        evidence=(
            CompetitionExcelEvidence(
                sheet_name=sheet.name,
                value_coordinate=(
                    value_cell.coordinate
                ),
                value_text=(
                    value_cell.text_value
                ),
                numeric_value=(
                    value_cell.numeric_value
                ),
                number_format=(
                    value_cell.number_format
                ),
                row_label_coordinate=(
                    row_label.coordinate
                    if row_label
                    else None
                ),
                row_label=(
                    row_label.text_value
                    if row_label
                    else None
                ),
                column_label_coordinate=(
                    column_label.coordinate
                    if column_label
                    else None
                ),
                column_label=(
                    column_label.text_value
                    if column_label
                    else None
                ),
            )
        ),
        confidence=confidence,
    )