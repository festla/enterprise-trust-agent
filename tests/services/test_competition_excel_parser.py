from pathlib import Path

import pytest
from openpyxl import Workbook

from app.schemas.competition import (
    CompetitionSourceRecord,
)
from app.services.competition_excel_parser import (
    CompetitionExcelParserError,
    parse_competition_excel,
)


def test_parse_sparse_xlsx(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.xlsx"

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = "人身保险公司（月度）"

    worksheet.merge_cells(
        "A1:C1"
    )

    worksheet["A1"] = "原保险保费收入"

    worksheet["B2"] = "本年累计"

    worksheet["C5"] = 123.45

    worksheet["C5"].number_format = (
        "0.00"
    )

    worksheet["D5"] = (
        "=SUM(C5:C5)"
    )

    workbook.save(path)

    source = CompetitionSourceRecord(
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="excel",
        actual_filename="test.xlsx",
        relative_path="test.xlsx",
        extension=".xlsx",
        size_bytes=(
            path.stat().st_size
        ),
    )

    parsed = parse_competition_excel(
        attachments_root=tmp_path,
        source=source,
    )

    assert (
        parsed.excel_format
        == "xlsx"
    )

    assert parsed.sheet_names == (
        "人身保险公司（月度）",
    )

    sheet = parsed.sheets[0]

    cell_by_coordinate = {
        cell.coordinate: cell
        for cell in sheet.cells
    }

    assert (
        "A1"
        in cell_by_coordinate
    )

    assert (
        cell_by_coordinate[
            "A1"
        ].merged_range
        == "A1:C1"
    )

    assert (
        cell_by_coordinate[
            "C5"
        ].numeric_value
        == 123.45
    )

    assert (
        cell_by_coordinate[
            "C5"
        ].number_format
        == "0.00"
    )

    assert (
        cell_by_coordinate[
            "D5"
        ].cell_type
        == "formula"
    )

    assert (
        cell_by_coordinate[
            "D5"
        ].formula
        == "=SUM(C5:C5)"
    )

    # openpyxl 不会主动计算公式，
    # 因此新建测试文件没有 cached value。
    assert (
        cell_by_coordinate[
            "D5"
        ].formula_cached_value
        is None
    )


def test_excel_parser_rejects_non_excel_source(
    tmp_path: Path,
) -> None:
    source = CompetitionSourceRecord(
        source_id=(
            "src_0123456789abcdef"
        ),
        source_type="pdf",
        actual_filename="test.pdf",
        relative_path="test.pdf",
        extension=".pdf",
        size_bytes=1,
    )

    with pytest.raises(
        CompetitionExcelParserError
    ):
        parse_competition_excel(
            attachments_root=tmp_path,
            source=source,
        )