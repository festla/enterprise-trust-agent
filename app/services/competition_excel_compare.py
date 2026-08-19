from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from itertools import (
    combinations,
    product,
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
    CompetitionExcelCompareItem,
    CompetitionExcelCompareOperation,
    CompetitionExcelCompareResult,
)


class CompetitionExcelCompareError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class _ResolvedOptionValue:
    option: CompetitionAnswerOption

    label_cell: CompetitionExcelCell

    value_cell: CompetitionExcelCell

    score: float

    context_labels: tuple[
        str,
        ...
    ]

def _select_comparable_option_labels(
    *,
    question: CompetitionQuestion,
    sheet: CompetitionExcelSheet,
) -> dict[
    CompetitionAnswerOption,
    CompetitionExcelCell,
]:
    """
    从 A/B/C/D 的 Label 候选中寻找一个空间连续的
    comparable section。

    当前比赛 Excel 的典型结构：

        原保险保费收入
        ├── 财产险
        ├── 人身险
        └── ...

        ...

        总资产

    或：

        第一组统计
        ...
        第二组统计
        ...
        第三组统计

    同一语义区块中的 Option Label 通常在 Excel 中
    空间上非常接近。

    策略：

    1. 优先寻找包含 4 个 Option 的紧凑区块；
    2. 否则允许 3/4 Option 构成多数可比较区块；
    3. Row-oriented 区块允许最大跨度 4 行；
    4. Column-oriented 区块允许最大跨度 4 列；
    5. 多个区块同分时，优先匹配质量更高、
       更紧凑、且更靠前的区块。
    """

    labels_by_option: dict[
        CompetitionAnswerOption,
        list[CompetitionExcelCell],
    ] = {}

    for option, option_text in (
        question.options.items()
    ):
        labels = _find_option_labels(
            sheet=sheet,
            option_text=option_text,
        )

        if labels:
            labels_by_option[
                option
            ] = labels

    if len(labels_by_option) < 3:
        raise CompetitionExcelCompareError(
            "不足 3 个 Option 能定位到 "
            "Excel Label"
        )

    option_order = (
        "A",
        "B",
        "C",
        "D",
    )

    candidates: list[
        tuple[
            tuple[
                int,
                int,
                int,
                int,
                int,
            ],
            dict[
                CompetitionAnswerOption,
                CompetitionExcelCell,
            ],
        ]
    ] = []

    # ========================================================
    # 优先尝试 4-option section，
    # 找不到再允许 3-option majority section。
    # ========================================================

    for subset_size in (
        4,
        3,
    ):
        available_options = [
            option
            for option in option_order
            if option
            in labels_by_option
        ]

        if (
            len(available_options)
            < subset_size
        ):
            continue

        for option_subset in combinations(
            available_options,
            subset_size,
        ):
            label_lists = [
                labels_by_option[
                    option
                ]
                for option
                in option_subset
            ]

            for selected_labels in product(
                *label_lists
            ):
                rows = [
                    cell.row_index
                    for cell
                    in selected_labels
                ]

                columns = [
                    cell.column_index
                    for cell
                    in selected_labels
                ]

                row_span = (
                    max(rows)
                    - min(rows)
                )

                column_span = (
                    max(columns)
                    - min(columns)
                )

                # --------------------------------------------
                # Row-oriented section
                #
                # B6 root
                # B7 child
                # B8 child
                # --------------------------------------------

                row_oriented = (
                    row_span <= 4
                    and column_span <= 1
                )

                # --------------------------------------------
                # Column-oriented section
                #
                # B3 C3 D3 E3
                # --------------------------------------------

                column_oriented = (
                    column_span <= 4
                    and row_span <= 1
                )

                if not (
                    row_oriented
                    or column_oriented
                ):
                    continue

                selected = {
                    option: cell
                    for option, cell
                    in zip(
                        option_subset,
                        selected_labels,
                        strict=True,
                    )
                }

                quality = sum(
                    _label_match_quality(
                        option_text=(
                            question.options[
                                option
                            ]
                        ),
                        cell_text=(
                            selected[
                                option
                            ].text_value
                        ),
                    )
                    for option
                    in option_subset
                )

                if row_oriented:
                    span = row_span

                    first_position = min(
                        rows
                    )

                    # 一个 major metric 往往位于
                    # block 的第一行。
                    anchor_quality = max(
                        _label_match_quality(
                            option_text=(
                                question.options[
                                    option
                                ]
                            ),
                            cell_text=(
                                selected[
                                    option
                                ].text_value
                            ),
                        )
                        for option
                        in option_subset
                        if (
                            selected[
                                option
                            ].row_index
                            == first_position
                        )
                    )

                else:
                    span = column_span

                    first_position = min(
                        columns
                    )

                    anchor_quality = max(
                        _label_match_quality(
                            option_text=(
                                question.options[
                                    option
                                ]
                            ),
                            cell_text=(
                                selected[
                                    option
                                ].text_value
                            ),
                        )
                        for option
                        in option_subset
                        if (
                            selected[
                                option
                            ].column_index
                            == first_position
                        )
                    )

                # tuple 越大越好：
                #
                # 1. Option 数量
                # 2. 区块起始位置匹配质量
                # 3. 总 Label 匹配质量
                # 4. 区块越紧凑越好
                # 5. 越靠前越好
                score = (
                    subset_size,
                    anchor_quality,
                    quality,
                    -span,
                    -first_position,
                )

                candidates.append(
                    (
                        score,
                        selected,
                    )
                )

        # 如果已经找到了完整 4-option 区块，
        # 就没必要用 3-option 区块覆盖它。
        if (
            subset_size == 4
            and candidates
        ):
            break

    if not candidates:
        raise CompetitionExcelCompareError(
            "无法找到空间连续的 "
            "Comparable Section"
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]

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
    """
    去除表格行标签中常见的结构性前缀。

    例如：

    其中：寿险
        -> 寿险

    其中：机动车辆保险
        -> 机动车辆保险

    注意：
    这里只处理非常保守的表格前缀，
    不进行语义改写或模糊匹配。
    """

    key = _normalize_text(
        value
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

def _parse_operation(
    question: str,
) -> CompetitionExcelCompareOperation:
    key = _normalize_text(
        question
    )

    max_words = (
        "最高",
        "最大",
        "最多",
    )

    min_words = (
        "最低",
        "最小",
        "最少",
    )

    has_max = any(
        word in key
        for word in max_words
    )

    has_min = any(
        word in key
        for word in min_words
    )

    if has_max and not has_min:
        return "max"

    if has_min and not has_max:
        return "min"

    raise CompetitionExcelCompareError(
        "无法确定比较操作："
        "目前仅支持 max / min"
    )


def _candidate_sheets(
    *,
    question: CompetitionQuestion,
    workbook: CompetitionExcelWorkbook,
) -> tuple[
    CompetitionExcelSheet,
    ...
]:
    question_key = _normalize_text(
        question.question
    )

    visible = tuple(
        sheet
        for sheet in workbook.sheets
        if sheet.visible
    )

    explicitly_mentioned = tuple(
        sheet
        for sheet in visible
        if (
            _normalize_text(
                sheet.name
            )
            in question_key
        )
    )

    if explicitly_mentioned:
        return explicitly_mentioned

    return visible


def _merged_by_anchor(
    sheet: CompetitionExcelSheet,
) -> dict[
    str,
    CompetitionExcelMergedRange,
]:
    return {
        merged.anchor_coordinate:
        merged
        for merged
        in sheet.merged_ranges
    }


def _text_applies_to_column(
    *,
    text_cell: CompetitionExcelCell,
    target_column: int,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> bool:
    if (
        text_cell.column_index
        == target_column
    ):
        return True

    merged = merged_map.get(
        text_cell.coordinate
    )

    if merged is None:
        return False

    return (
        merged.min_column
        <= target_column
        <= merged.max_column
    )


def _text_applies_to_row(
    *,
    text_cell: CompetitionExcelCell,
    target_row: int,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> bool:
    if (
        text_cell.row_index
        == target_row
    ):
        return True

    merged = merged_map.get(
        text_cell.coordinate
    )

    if merged is None:
        return False

    return (
        merged.min_row
        <= target_row
        <= merged.max_row
    )


def _question_context_above(
    *,
    sheet: CompetitionExcelSheet,
    candidate: CompetitionExcelCell,
    question_key: str,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> tuple[str, ...]:
    matches: list[str] = []

    for cell in sheet.cells:
        if cell.cell_type != "text":
            continue

        if (
            cell.row_index
            >= candidate.row_index
        ):
            continue

        text_key = _normalize_text(
            cell.text_value
        )

        if len(text_key) < 2:
            continue

        if text_key not in question_key:
            continue

        if not _text_applies_to_column(
            text_cell=cell,
            target_column=(
                candidate.column_index
            ),
            merged_map=merged_map,
        ):
            continue

        matches.append(
            cell.text_value
        )

    return tuple(
        dict.fromkeys(matches)
    )


def _question_context_left(
    *,
    sheet: CompetitionExcelSheet,
    candidate: CompetitionExcelCell,
    question_key: str,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> tuple[str, ...]:
    matches: list[str] = []

    for cell in sheet.cells:
        if cell.cell_type != "text":
            continue

        if (
            cell.column_index
            >= candidate.column_index
        ):
            continue

        text_key = _normalize_text(
            cell.text_value
        )

        if len(text_key) < 2:
            continue

        if text_key not in question_key:
            continue

        if not _text_applies_to_row(
            text_cell=cell,
            target_row=(
                candidate.row_index
            ),
            merged_map=merged_map,
        ):
            continue

        matches.append(
            cell.text_value
        )

    return tuple(
        dict.fromkeys(matches)
    )


def _score_value_candidate(
    *,
    sheet: CompetitionExcelSheet,
    label: CompetitionExcelCell,
    value: CompetitionExcelCell,
    question_key: str,
    merged_map: dict[
        str,
        CompetitionExcelMergedRange,
    ],
) -> tuple[
    float,
    tuple[str, ...],
]:
    # ========================================================
    # 情形 A：
    # Option 是行标签，数字在同一行。
    #
    # B5 原保险保费收入 | C5 12345
    # ========================================================

    if (
        label.row_index
        == value.row_index
    ):
        distance = abs(
            value.column_index
            - label.column_index
        )

        if distance == 0:
            return (
                -math.inf,
                (),
            )

        context = (
            _question_context_above(
                sheet=sheet,
                candidate=value,
                question_key=(
                    question_key
                ),
                merged_map=(
                    merged_map
                ),
            )
        )

        # 问题口径命中的表头，
        # 权重要远高于“离得近”。
        score = (
            10.0 * len(context)
            + (
                2.0
                if (
                    value.column_index
                    > label.column_index
                )
                else 0.25
            )
            + 1.0 / distance
        )

        return (
            score,
            context,
        )

    # ========================================================
    # 情形 B：
    # Option 是列标签，数字在同一列。
    #
    # 后续如果出现横向表也可以处理。
    # ========================================================

    if (
        label.column_index
        == value.column_index
    ):
        distance = abs(
            value.row_index
            - label.row_index
        )

        if distance == 0:
            return (
                -math.inf,
                (),
            )

        context = (
            _question_context_left(
                sheet=sheet,
                candidate=value,
                question_key=(
                    question_key
                ),
                merged_map=(
                    merged_map
                ),
            )
        )

        score = (
            10.0 * len(context)
            + (
                2.0
                if (
                    value.row_index
                    > label.row_index
                )
                else 0.25
            )
            + 1.0 / distance
        )

        return (
            score,
            context,
        )

    return (
        -math.inf,
        (),
    )


def _find_option_labels(
    *,
    sheet: CompetitionExcelSheet,
    option_text: str,
) -> list[
    CompetitionExcelCell
]:
    """
    按保守优先级解析 Option Label。

    Priority 1:
        normalized exact

    Priority 2:
        去除“其中”等结构性前缀后 exact

    Priority 3:
        很短的唯一 containment variant

    不使用：
        fuzzy similarity
        embedding
        LLM
    """

    option_key = _normalize_text(
        option_text
    )

    text_cells = [
        cell
        for cell in sheet.cells
        if (
            cell.cell_type == "text"
            and cell.text_value
        )
    ]

    # ========================================================
    # Level 1
    # Exact normalized match
    #
    # Q035 会保留多个 exact candidate，
    # 后续由 section context 消歧。
    # ========================================================

    exact_matches = [
        cell
        for cell in text_cells
        if (
            _normalize_text(
                cell.text_value
            )
            == option_key
        )
    ]

    if exact_matches:
        return exact_matches

    # ========================================================
    # Level 2
    # Structural-prefix match
    #
    # 寿险
    #   <- 其中：寿险
    #
    # 财产险
    #   <- 其中：财产险
    # ========================================================

    qualified_matches = [
        cell
        for cell in text_cells
        if (
            _strip_label_qualifier(
                cell.text_value
            )
            == option_key
        )
    ]

    if qualified_matches:
        return qualified_matches

    # ========================================================
    # Level 3
    # Conservative containment
    #
    # 只接受“非常接近”的短标签。
    #
    # 避免：
    #
    # 原保险保费收入
    #   匹配到
    # “原保险保费收入为按企业会计准则……”
    # ========================================================

    containment_matches: list[
        CompetitionExcelCell
    ] = []

    for cell in text_cells:
        cell_key = _normalize_text(
            cell.text_value
        )

        if not option_key:
            continue

        if option_key not in cell_key:
            continue

        extra_length = (
            len(cell_key)
            - len(option_key)
        )

        # 最多只允许增加少量修饰文字。
        #
        # 例如：
        # 大型商业银行（含...）
        #
        # 但不会允许几十上百字备注。
        max_extra = max(
            6,
            len(option_key),
        )

        if extra_length <= max_extra:
            containment_matches.append(
                cell
            )

    return containment_matches

def _label_match_quality(
    *,
    option_text: str,
    cell_text: str,
) -> int:
    """
    Label 与 Option 的结构匹配质量。

    3:
        normalized exact

    2:
        去掉“其中”等结构前缀后 exact

    1:
        conservative containment
    """

    option_key = _normalize_text(
        option_text
    )

    cell_key = _normalize_text(
        cell_text
    )

    if cell_key == option_key:
        return 3

    if (
        _strip_label_qualifier(
            cell_text
        )
        == option_key
    ):
        return 2

    if (
        option_key
        and option_key in cell_key
    ):
        return 1

    return 0

def _resolve_option_value(
    *,
    option: CompetitionAnswerOption,
    option_text: str,
    sheet: CompetitionExcelSheet,
    question_key: str,
    label_override: (
        CompetitionExcelCell | None
    ) = None,
) -> _ResolvedOptionValue:
    if label_override is not None:
        labels = [
            label_override
        ]
    else:
        labels = _find_option_labels(
            sheet=sheet,
            option_text=option_text,
        )

    if not labels:
        raise CompetitionExcelCompareError(
            f"找不到选项标签: "
            f"{option}={option_text}"
        )

    numeric_cells = [
        cell
        for cell in sheet.cells
        if (
            cell.numeric_value
            is not None
        )
    ]

    merged_map = (
        _merged_by_anchor(
            sheet
        )
    )

    candidates: list[
        _ResolvedOptionValue
    ] = []

    for label in labels:
        for value in numeric_cells:
            (
                score,
                context,
            ) = _score_value_candidate(
                sheet=sheet,
                label=label,
                value=value,
                question_key=(
                    question_key
                ),
                merged_map=(
                    merged_map
                ),
            )

            if not math.isfinite(
                score
            ):
                continue

            candidates.append(
                _ResolvedOptionValue(
                    option=option,
                    label_cell=label,
                    value_cell=value,
                    score=score,
                    context_labels=(
                        context
                    ),
                )
            )

    if not candidates:
        raise CompetitionExcelCompareError(
            f"找到了选项 {option_text}，"
            "但没有定位到对应数值"
        )

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.value_cell.row_index,
            item.value_cell.column_index,
        )
    )

    best = candidates[0]

    if (
        len(candidates) > 1
        and math.isclose(
            candidates[0].score,
            candidates[1].score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and (
            candidates[0]
            .value_cell.coordinate
            != candidates[1]
            .value_cell.coordinate
        )
    ):
        raise CompetitionExcelCompareError(
            f"选项 {option_text} "
            "存在多个同分数值候选"
        )

    return best


def _solve_sheet(
    *,
    question: CompetitionQuestion,
    sheet: CompetitionExcelSheet,
) -> tuple[
    float,
    tuple[
        _ResolvedOptionValue,
        ...
    ],
]:
    question_key = _normalize_text(
        question.question
    )

    selected_labels = (
        _select_comparable_option_labels(
            question=question,
            sheet=sheet,
        )
    )

    resolved: list[
        _ResolvedOptionValue
    ] = []

    total_score = 0.0

    for option in (
        "A",
        "B",
        "C",
        "D",
    ):
        if option not in selected_labels:
            continue

        option_text = (
            question.options[
                option
            ]
        )

        try:
            result = (
                _resolve_option_value(
                    option=option,
                    option_text=(
                        option_text
                    ),
                    sheet=sheet,
                    question_key=(
                        question_key
                    ),
                    label_override=(
                        selected_labels[
                            option
                        ]
                    ),
                )
            )

        except CompetitionExcelCompareError as exc:
            raise CompetitionExcelCompareError(
                f"Sheet={sheet.name!r}, "
                f"option={option}, "
                f"text={option_text!r}: "
                f"{exc}"
            ) from exc

        resolved.append(
            result
        )

        total_score += (
            result.score
        )

    if len(resolved) < 2:
        raise CompetitionExcelCompareError(
            "Comparable Section 中"
            "有效比较项不足"
        )

    return (
        total_score,
        tuple(resolved),
    )

def solve_excel_table_compare(
    *,
    question: CompetitionQuestion,
    workbook: CompetitionExcelWorkbook,
) -> CompetitionExcelCompareResult:
    if (
        question.qa_type
        != "表格比较"
    ):
        raise CompetitionExcelCompareError(
            "该 Solver 只处理表格比较"
        )

    operation = _parse_operation(
        question.question
    )

    candidate_sheets = (
        _candidate_sheets(
            question=question,
            workbook=workbook,
        )
    )

    if not candidate_sheets:
        raise CompetitionExcelCompareError(
            "没有可用 Sheet"
        )

    sheet_results: list[
        tuple[
            float,
            CompetitionExcelSheet,
            tuple[
                _ResolvedOptionValue,
                ...
            ],
        ]
    ] = []

    sheet_errors: list[str] = []

    for sheet in candidate_sheets:
        try:
            (
                score,
                resolved,
            ) = _solve_sheet(
                question=question,
                sheet=sheet,
            )

        except CompetitionExcelCompareError as exc:
            sheet_errors.append(
                str(exc)
            )
            continue

        sheet_results.append(
            (
                score,
                sheet,
                resolved,
            )
        )

    if not sheet_results:
        detail = (
            " | ".join(sheet_errors)
            if sheet_errors
            else "无候选 Sheet"
        )

        raise CompetitionExcelCompareError(
            "没有任何 Sheet 能完整解析 "
            "A/B/C/D 四个比较项；"
            f"details: {detail}"
        )

    sheet_results.sort(
        key=lambda item: (
            -item[0],
            item[1].sheet_index,
        )
    )

    (
        _sheet_score,
        sheet,
        resolved,
    ) = sheet_results[0]

    values = [
        float(
            item.value_cell
            .numeric_value
        )
        for item in resolved
    ]

    if operation == "max":
        target_value = max(values)
    else:
        target_value = min(values)

    winners = [
        item
        for item in resolved
        if math.isclose(
            float(
                item.value_cell
                .numeric_value
            ),
            target_value,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ]

    if len(winners) != 1:
        raise CompetitionExcelCompareError(
            "比较结果存在并列，"
            "无法唯一选择答案"
        )

    winner = winners[0]

    evidence_items = tuple(
        CompetitionExcelCompareItem(
            option=item.option,
            label_text=(
                item.label_cell
                .text_value
            ),
            label_coordinate=(
                item.label_cell
                .coordinate
            ),
            value_coordinate=(
                item.value_cell
                .coordinate
            ),
            value_text=(
                item.value_cell
                .text_value
            ),
            numeric_value=(
                item.value_cell
                .numeric_value
            ),
            context_labels=(
                item.context_labels
            ),
        )
        for item in resolved
    )

    context_hit_count = sum(
        1
        for item in resolved
        if item.context_labels
    )

    confidence = min(
        0.99,
        0.75
        + 0.05
        * context_hit_count,
    )

    return CompetitionExcelCompareResult(
        operation=operation,
        answer_option=(
            winner.option
        ),
        answer_text=(
            winner.label_cell
            .text_value
        ),
        winning_value=(
            winner.value_cell
            .numeric_value
        ),
        sheet_name=sheet.name,
        items=evidence_items,
        confidence=confidence,
    )