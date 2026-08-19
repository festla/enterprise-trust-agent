import pytest
from pydantic import ValidationError

from app.schemas.competition_excel import (
    CompetitionExcelCell,
    CompetitionExcelMergedRange,
    CompetitionExcelSheet,
    CompetitionExcelWorkbook,
)


def test_excel_workbook_sparse_structure(
) -> None:
    merged = CompetitionExcelMergedRange(
        range_ref="A1:C1",
        min_row=1,
        max_row=1,
        min_column=1,
        max_column=3,
        anchor_coordinate="A1",
    )

    cells = (
        CompetitionExcelCell(
            coordinate="A1",
            row_index=1,
            column_index=1,
            cell_type="text",
            text_value="原保险保费收入",
            merged_range="A1:C1",
        ),
        CompetitionExcelCell(
            coordinate="C5",
            row_index=5,
            column_index=3,
            cell_type="number",
            text_value="123.45",
            numeric_value=123.45,
        ),
    )

    sheet = CompetitionExcelSheet(
        name="Sheet1",
        sheet_index=0,
        visible=True,
        declared_max_row=10007,
        declared_max_column=16384,
        cells=cells,
        merged_ranges=(merged,),
    )

    workbook = CompetitionExcelWorkbook(
        source_id=(
            "src_0123456789abcdef"
        ),
        relative_path="test.xlsx",
        excel_format="xlsx",
        sheets=(sheet,),
    )

    # 声明范围很大，
    # 但实际上只保存两个有意义 Cell。
    assert (
        workbook.total_cell_count
        == 2
    )

    assert workbook.sheet_names == (
        "Sheet1",
    )

    assert (
        sheet.non_empty_cell_count
        == 2
    )


def test_formula_cell_keeps_formula_and_value(
) -> None:
    cell = CompetitionExcelCell(
        coordinate="C6",
        row_index=6,
        column_index=3,
        cell_type="formula",
        text_value="123.45",
        numeric_value=123.45,
        formula="=SUM(C1:C5)",
        formula_cached_value=123.45,
    )

    assert (
        cell.formula
        == "=SUM(C1:C5)"
    )

    assert (
        cell.formula_cached_value
        == 123.45
    )


def test_formula_requires_formula_cell_type(
) -> None:
    with pytest.raises(
        ValidationError
    ):
        CompetitionExcelCell(
            coordinate="C6",
            row_index=6,
            column_index=3,
            cell_type="number",
            text_value="123.45",
            numeric_value=123.45,
            formula="=SUM(C1:C5)",
        )


def test_merged_range_validates_bounds(
) -> None:
    with pytest.raises(
        ValidationError
    ):
        CompetitionExcelMergedRange(
            range_ref="A5:A1",
            min_row=5,
            max_row=1,
            min_column=1,
            max_column=1,
            anchor_coordinate="A5",
        )