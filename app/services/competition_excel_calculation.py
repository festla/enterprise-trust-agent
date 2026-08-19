from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)

from app.schemas.competition import (
    CompetitionAnswerOption,
    CompetitionQuestion,
)
from app.schemas.competition_excel import (
    CompetitionExcelCell,
    CompetitionExcelMergedRange,
    CompetitionExcelSheet,
    CompetitionExcelWorkbook,
)
from app.schemas.competition_excel_solver import (
    CompetitionExcelCalculationOperand,
    CompetitionExcelCalculationResult,
)


class CompetitionExcelCalculationError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class _CalculationIntent:
    entity_text: str

    start_scope: str

    end_scope: str


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
        r"[，。；：、,.!?！？"
        r"（）()《》“”\"'：:/\\_-]",
        "",
        value,
    )

    return value


def _strip_label_qualifier(
    value: str,
) -> str:
    key = _normalize_text(
        value
    )

    key = re.sub(
        r"^\d+[、.．]",
        "",
        key,
    )

    prefixes = (
        "其中包括",
        "其中",
    )

    for prefix in prefixes:
        if (
            key.startswith(prefix)
            and len(key) > len(prefix)
        ):
            return key[
                len(prefix):
            ]

    return key


def _parse_intent(
    question: str,
) -> _CalculationIntent:
    match = re.search(
        r"“(?P<entity>[^”]+)”"
        r"\s*从\s*"
        r"“(?P<start>[^”]+)”"
        r"\s*到\s*"
        r"“(?P<end>[^”]+)”"
        r".*?数值变化",
        question,
    )

    if match is None:
        raise CompetitionExcelCalculationError(
            "无法解析计算题中的 "
            "entity/start/end"
        )

    return _CalculationIntent(
        entity_text=(
            match.group(
                "entity"
            ).strip()
        ),
        start_scope=(
            match.group(
                "start"
            ).strip()
        ),
        end_scope=(
            match.group(
                "end"
            ).strip()
        ),
    )


def _find_entity_cell(
    *,
    sheet: CompetitionExcelSheet,
    entity_text: str,
) -> CompetitionExcelCell:
    entity_key = _normalize_text(
        entity_text
    )

    text_cells = [
        cell
        for cell in sheet.cells
        if (
            cell.cell_type == "text"
            and cell.text_value
        )
    ]

    exact = [
        cell
        for cell in text_cells
        if (
            _normalize_text(
                cell.text_value
            )
            == entity_key
        )
    ]

    if exact:
        return min(
            exact,
            key=lambda cell: (
                cell.row_index,
                cell.column_index,
            ),
        )

    qualified = [
        cell
        for cell in text_cells
        if (
            _strip_label_qualifier(
                cell.text_value
            )
            == entity_key
        )
    ]

    if qualified:
        return min(
            qualified,
            key=lambda cell: (
                cell.row_index,
                cell.column_index,
            ),
        )

    raise CompetitionExcelCalculationError(
        f"找不到实体行标签: "
        f"{entity_text}"
    )


def _merged_map(
    sheet: CompetitionExcelSheet,
) -> dict[
    str,
    CompetitionExcelMergedRange,
]:
    return {
        item.anchor_coordinate:
        item
        for item
        in sheet.merged_ranges
    }


def _header_applies_to_column(
    *,
    cell: CompetitionExcelCell,
    column_index: int,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> bool:
    if (
        cell.column_index
        == column_index
    ):
        return True

    merged = merged_map.get(
        cell.coordinate
    )

    if merged is None:
        return False

    return (
        merged.min_column
        <= column_index
        <= merged.max_column
    )


def _header_labels_for_column(
    *,
    sheet: CompetitionExcelSheet,
    column_index: int,
    before_row: int,
) -> tuple[str, ...]:
    merged_map = _merged_map(
        sheet
    )

    labels = []

    for cell in sheet.cells:
        if (
            cell.cell_type
            != "text"
        ):
            continue

        if (
            cell.row_index
            >= before_row
        ):
            continue

        if not _header_applies_to_column(
            cell=cell,
            column_index=column_index,
            merged_map=merged_map,
        ):
            continue

        labels.append(
            cell.text_value
        )

    return tuple(
        dict.fromkeys(
            labels
        )
    )


def _resolve_direct_header_column(
    *,
    sheet: CompetitionExcelSheet,
    scope_text: str,
    before_row: int,
) -> int | None:
    scope_key = _normalize_text(
        scope_text
    )

    matches = [
        cell
        for cell in sheet.cells
        if (
            cell.cell_type == "text"
            and cell.row_index
            < before_row
            and _normalize_text(
                cell.text_value
            )
            == scope_key
        )
    ]

    columns = {
        cell.column_index
        for cell in matches
    }

    if len(columns) == 1:
        return next(
            iter(columns)
        )

    return None


_QUARTER_KEYS = {
    "一季度",
    "二季度",
    "三季度",
    "四季度",
    "1季度",
    "2季度",
    "3季度",
    "4季度",
}


def _resolve_quarter_boundary_column(
    *,
    sheet: CompetitionExcelSheet,
    scope_text: str,
    before_row: int,
    role: str,
) -> int | None:
    scope_key = _normalize_text(
        scope_text
    )

    if "季度" not in scope_key:
        return None

    quarter_cells = [
        cell
        for cell in sheet.cells
        if (
            cell.cell_type == "text"
            and cell.row_index
            < before_row
            and _normalize_text(
                cell.text_value
            )
            in _QUARTER_KEYS
        )
    ]

    if len(
        quarter_cells
    ) < 2:
        return None

    # 选择 quarter header 最密集的那一行。
    by_row: dict[
        int,
        list[
            CompetitionExcelCell
        ],
    ] = {}

    for cell in quarter_cells:
        by_row.setdefault(
            cell.row_index,
            [],
        ).append(
            cell
        )

    best_row_cells = max(
        by_row.values(),
        key=lambda cells: len(
            cells
        ),
    )

    columns = sorted(
        {
            cell.column_index
            for cell
            in best_row_cells
        }
    )

    if len(columns) < 2:
        return None

    if role == "start":
        return columns[0]

    if role == "end":
        return columns[-1]

    return None


def _resolve_scope_column(
    *,
    sheet: CompetitionExcelSheet,
    scope_text: str,
    before_row: int,
    role: str,
) -> int | None:
    direct = (
        _resolve_direct_header_column(
            sheet=sheet,
            scope_text=scope_text,
            before_row=before_row,
        )
    )

    if direct is not None:
        return direct

    return (
        _resolve_quarter_boundary_column(
            sheet=sheet,
            scope_text=scope_text,
            before_row=before_row,
            role=role,
        )
    )


def _numeric_cell_at(
    *,
    sheet: CompetitionExcelSheet,
    row_index: int,
    column_index: int,
) -> CompetitionExcelCell | None:
    for cell in sheet.cells:
        if (
            cell.row_index
            == row_index
            and cell.column_index
            == column_index
            and cell.numeric_value
            is not None
        ):
            return cell

    return None


def _row_numeric_cells(
    *,
    sheet: CompetitionExcelSheet,
    row_index: int,
) -> list[
    CompetitionExcelCell
]:
    return sorted(
        [
            cell
            for cell in sheet.cells
            if (
                cell.row_index
                == row_index
                and cell.numeric_value
                is not None
            )
        ],
        key=lambda cell:
            cell.column_index,
    )


def _parse_option_number(
    value: str,
) -> tuple[
    Decimal,
    int,
] | None:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).replace(
        ",",
        "",
    ).strip()

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        normalized,
    )

    if match is None:
        return None

    token = match.group(0)

    try:
        number = Decimal(
            token
        )
    except InvalidOperation:
        return None

    decimal_places = (
        len(
            token.split(
                ".",
                1,
            )[1]
        )
        if "." in token
        else 0
    )

    return (
        number,
        decimal_places,
    )


def _quantize(
    value: Decimal,
    decimal_places: int,
) -> Decimal:
    quantizer = (
        Decimal(1).scaleb(
            -decimal_places
        )
    )

    return value.quantize(
        quantizer,
        rounding=ROUND_HALF_UP,
    )


def _match_result_option(
    *,
    question: CompetitionQuestion,
    result: Decimal,
) -> CompetitionAnswerOption | None:
    matches = []

    for (
        option,
        option_text,
    ) in question.options.items():
        parsed = (
            _parse_option_number(
                option_text
            )
        )

        if parsed is None:
            continue

        (
            option_value,
            decimal_places,
        ) = parsed

        if (
            _quantize(
                result,
                decimal_places,
            )
            == _quantize(
                option_value,
                decimal_places,
            )
        ):
            matches.append(
                option
            )

    if len(matches) == 1:
        return matches[0]

    return None


def _difference(
    *,
    start: CompetitionExcelCell,
    end: CompetitionExcelCell,
) -> Decimal:
    return (
        Decimal(
            str(
                end.numeric_value
            )
        )
        - Decimal(
            str(
                start.numeric_value
            )
        )
    )


def _option_constrained_fallback(
    *,
    question: CompetitionQuestion,
    sheet: CompetitionExcelSheet,
    entity_row: int,
    start_column: int | None,
    end_column: int | None,
) -> tuple[
    CompetitionExcelCell,
    CompetitionExcelCell,
    CompetitionAnswerOption,
]:
    row_cells = (
        _row_numeric_cells(
            sheet=sheet,
            row_index=entity_row,
        )
    )

    if len(row_cells) < 2:
        raise CompetitionExcelCalculationError(
            "实体行中数值 Cell 不足"
        )

    matches = []

    for start in row_cells:
        if (
            start_column is not None
            and start.column_index
            != start_column
        ):
            continue

        for end in row_cells:
            if (
                end_column is not None
                and end.column_index
                != end_column
            ):
                continue

            # 当前 benchmark 的
            # “从 X 到 Y”对应表格中
            # 从左向右的变化。
            if (
                end.column_index
                <= start.column_index
            ):
                continue

            result = _difference(
                start=start,
                end=end,
            )

            option = (
                _match_result_option(
                    question=question,
                    result=result,
                )
            )

            if option is None:
                continue

            matches.append(
                (
                    start,
                    end,
                    option,
                )
            )

    unique = {
        (
            start.coordinate,
            end.coordinate,
            option,
        ): (
            start,
            end,
            option,
        )
        for (
            start,
            end,
            option,
        )
        in matches
    }

    if len(unique) != 1:
        raise CompetitionExcelCalculationError(
            "Option-constrained fallback "
            "无法得到唯一 Operand Pair；"
            f"candidate_count="
            f"{len(unique)}"
        )

    return next(
        iter(
            unique.values()
        )
    )


def _solve_sheet(
    *,
    question: CompetitionQuestion,
    sheet: CompetitionExcelSheet,
    intent: _CalculationIntent,
) -> CompetitionExcelCalculationResult:
    entity = _find_entity_cell(
        sheet=sheet,
        entity_text=(
            intent.entity_text
        ),
    )

    start_column = (
        _resolve_scope_column(
            sheet=sheet,
            scope_text=(
                intent.start_scope
            ),
            before_row=(
                entity.row_index
            ),
            role="start",
        )
    )

    end_column = (
        _resolve_scope_column(
            sheet=sheet,
            scope_text=(
                intent.end_scope
            ),
            before_row=(
                entity.row_index
            ),
            role="end",
        )
    )

    start_cell = None
    end_cell = None

    if start_column is not None:
        start_cell = (
            _numeric_cell_at(
                sheet=sheet,
                row_index=(
                    entity.row_index
                ),
                column_index=(
                    start_column
                ),
            )
        )

    if end_column is not None:
        end_cell = (
            _numeric_cell_at(
                sheet=sheet,
                row_index=(
                    entity.row_index
                ),
                column_index=(
                    end_column
                ),
            )
        )

    resolution_mode = (
        "semantic_headers"
    )

    if (
        start_cell is None
        or end_cell is None
    ):
        (
            start_cell,
            end_cell,
            answer_option,
        ) = (
            _option_constrained_fallback(
                question=question,
                sheet=sheet,
                entity_row=(
                    entity.row_index
                ),
                start_column=(
                    start_column
                ),
                end_column=(
                    end_column
                ),
            )
        )

        resolution_mode = (
            "option_constrained_fallback"
        )

    else:
        result_decimal = _difference(
            start=start_cell,
            end=end_cell,
        )

        answer_option = (
            _match_result_option(
                question=question,
                result=(
                    result_decimal
                ),
            )
        )

        if answer_option is None:
            raise CompetitionExcelCalculationError(
                "计算结果无法唯一匹配选项"
            )

    result_decimal = _difference(
        start=start_cell,
        end=end_cell,
    )

    start_headers = (
        _header_labels_for_column(
            sheet=sheet,
            column_index=(
                start_cell
                .column_index
            ),
            before_row=(
                entity.row_index
            ),
        )
    )

    end_headers = (
        _header_labels_for_column(
            sheet=sheet,
            column_index=(
                end_cell
                .column_index
            ),
            before_row=(
                entity.row_index
            ),
        )
    )

    confidence = (
        0.98
        if (
            resolution_mode
            == "semantic_headers"
        )
        else 0.82
    )

    return (
        CompetitionExcelCalculationResult(
            operation="difference",
            entity_text=(
                entity.text_value
            ),
            entity_coordinate=(
                entity.coordinate
            ),
            start=(
                CompetitionExcelCalculationOperand(
                    role="start",
                    scope_text=(
                        intent.start_scope
                    ),
                    coordinate=(
                        start_cell
                        .coordinate
                    ),
                    numeric_value=(
                        start_cell
                        .numeric_value
                    ),
                    header_labels=(
                        start_headers
                    ),
                )
            ),
            end=(
                CompetitionExcelCalculationOperand(
                    role="end",
                    scope_text=(
                        intent.end_scope
                    ),
                    coordinate=(
                        end_cell
                        .coordinate
                    ),
                    numeric_value=(
                        end_cell
                        .numeric_value
                    ),
                    header_labels=(
                        end_headers
                    ),
                )
            ),
            formula=(
                f"{end_cell.coordinate} "
                f"- "
                f"{start_cell.coordinate}"
            ),
            result=float(
                result_decimal
            ),
            answer_option=(
                answer_option
            ),
            answer_text=(
                question.options[
                    answer_option
                ]
            ),
            resolution_mode=(
                resolution_mode
            ),
            confidence=(
                confidence
            ),
        )
    )


def solve_excel_table_calculation(
    *,
    question: CompetitionQuestion,
    workbook: CompetitionExcelWorkbook,
) -> CompetitionExcelCalculationResult:
    if (
        question.qa_type
        != "表格计算"
    ):
        raise CompetitionExcelCalculationError(
            "该 Solver 只处理表格计算"
        )

    intent = _parse_intent(
        question.question
    )

    question_key = (
        _normalize_text(
            question.question
        )
    )

    visible_sheets = [
        sheet
        for sheet
        in workbook.sheets
        if sheet.visible
    ]

    mentioned = [
        sheet
        for sheet
        in visible_sheets
        if (
            _normalize_text(
                sheet.name
            )
            in question_key
        )
    ]

    candidate_sheets = (
        mentioned
        or visible_sheets
    )

    errors = []

    for sheet in candidate_sheets:
        try:
            return _solve_sheet(
                question=question,
                sheet=sheet,
                intent=intent,
            )

        except (
            CompetitionExcelCalculationError
        ) as exc:
            errors.append(
                f"{sheet.name}: {exc}"
            )

    raise CompetitionExcelCalculationError(
        "没有 Sheet 能完成计算；"
        + " | ".join(
            errors
        )
    )